import copy
import os
import queue
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from qwen3_tts_gguf.inference import TTSEngine, TTSConfig

SAMPLE_RATE = 24000
MODEL_DIR = os.environ.get("QWEN_TTS_MODEL_DIR", "model-base-small")
VOICE_DIR = Path(os.environ.get("QWEN_TTS_VOICE_DIR", os.path.join("output", "elaborate")))
EXTRA_VOICE_DIR = Path(os.environ.get("QWEN_TTS_EXTRA_VOICE_DIR", os.path.join("output", "voices")))
DEFAULT_VOICE = os.environ.get("QWEN_TTS_VOICE", "Vivian")
ONNX_PROVIDER = os.environ.get("QWEN_TTS_ONNX_PROVIDER", "CUDA")
MAX_CONCURRENT_STREAMS = int(os.environ.get("QWEN_TTS_STREAM_WORKERS", "2"))
CACHE_DEFAULT_VOICE = os.environ.get("QWEN_TTS_CACHE_DEFAULT_VOICE", "0") == "1"

engine: Optional[TTSEngine] = None
voice_map: dict[str, "VoiceItem"] = {}
request_queue: "queue.Queue[Optional[StreamTask]]"
worker_threads: list[threading.Thread] = []
default_voice_result = None
worker_default_streams = []
service_task_counter = 0
service_task_counter_lock = threading.Lock()
decoder_play_q_original = None
decoder_play_q_silent_users = 0
decoder_play_q_lock = threading.Lock()
STREAM_END = object()


@dataclass(frozen=True)
class VoiceItem:
    name: str
    path: Path
    kind: str
    ref_text: Optional[str] = None


@dataclass
class StreamTask:
    text: str
    speed: float
    voice: VoiceItem
    chunks: "queue.Queue[object]"


class StreamRequest(BaseModel):
    text: str = Field(..., min_length=1, description="待合成的文本")
    seed: float = Field(1.0, gt=0, description="语速倍率，1.0 为正常速度")
    voice: Optional[str] = Field(None, description="音色名称")


class _SilentPlayQueue:
    def put(self, *_args, **_kwargs) -> None:
        return None


def load_voice_map() -> dict[str, VoiceItem]:
    voices: dict[str, VoiceItem] = {}

    if VOICE_DIR.is_dir():
        for path in sorted(VOICE_DIR.glob("*.json")):
            voices[path.stem] = VoiceItem(path.stem, path.resolve(), "json")

    if EXTRA_VOICE_DIR.is_dir():
        for ext in ("*.mp3", "*.m4a", "*.wav", "*.flac", "*.opus"):
            for path in sorted(EXTRA_VOICE_DIR.glob(ext)):
                txt_path = path.with_suffix(".txt")
                ref_text = txt_path.read_text(encoding="utf-8").strip() if txt_path.is_file() else None
                # Prefer elaborate JSON when both JSON and raw reference audio share the same name.
                voices.setdefault(path.stem, VoiceItem(path.stem, path.resolve(), "audio", ref_text))

    if not voices:
        raise RuntimeError(
            f"No voice files found in: {VOICE_DIR.resolve()} or {EXTRA_VOICE_DIR.resolve()}"
        )
    return voices


def resolve_voice(voice_name: Optional[str]) -> VoiceItem:
    selected = (voice_name or DEFAULT_VOICE).strip()
    if selected not in voice_map:
        raise RuntimeError(
            f"Unknown voice: {selected}. Available voices: {', '.join(sorted(voice_map))}"
        )
    return voice_map[selected]


def pcm_to_int16_bytes(audio: np.ndarray) -> bytes:
    pcm = np.asarray(audio, dtype=np.float32)
    if pcm.size == 0:
        return b""
    pcm = np.clip(pcm, -1.0, 1.0)
    return (pcm * 32767.0).astype("<i2").tobytes()


def wav_stream_header(sample_rate: int, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = 0x7FFFFFFF
    riff_size = 36 + data_size
    return (
        b"RIFF"
        + int(riff_size).to_bytes(4, "little", signed=False)
        + b"WAVE"
        + b"fmt "
        + (16).to_bytes(4, "little", signed=False)
        + (1).to_bytes(2, "little", signed=False)
        + int(channels).to_bytes(2, "little", signed=False)
        + int(sample_rate).to_bytes(4, "little", signed=False)
        + int(byte_rate).to_bytes(4, "little", signed=False)
        + int(block_align).to_bytes(2, "little", signed=False)
        + int(bits_per_sample).to_bytes(2, "little", signed=False)
        + b"data"
        + int(data_size).to_bytes(4, "little", signed=False)
    )


def log_stream_timing(label: str, started_at: float) -> float:
    now = time.perf_counter()
    elapsed_ms = (now - started_at) * 1000.0
    print(f"[web_stream_service] {label} elapsed_ms={elapsed_ms:.1f}", flush=True)
    return now


def next_service_task_id(stream) -> str:
    global service_task_counter
    with service_task_counter_lock:
        task_number = service_task_counter
        service_task_counter += 1
    stream.task_counter = task_number
    return f"task_{task_number}"


def acquire_decoder_silence():
    global decoder_play_q_silent_users
    if engine is None or engine.decoder is None:
        return
    with decoder_play_q_lock:
        if decoder_play_q_silent_users == 0 and decoder_play_q_original is not None:
            engine.decoder.play_q = _SilentPlayQueue()
        decoder_play_q_silent_users += 1


def release_decoder_silence():
    global decoder_play_q_silent_users
    if engine is None or engine.decoder is None:
        return
    with decoder_play_q_lock:
        if decoder_play_q_silent_users > 0:
            decoder_play_q_silent_users -= 1
        if decoder_play_q_silent_users == 0 and decoder_play_q_original is not None:
            engine.decoder.play_q = decoder_play_q_original


def process_stream_task(task: StreamTask, worker_index: int) -> None:
    if engine is None:
        raise RuntimeError("TTS engine is not initialized.")

    total_started_at = time.perf_counter()
    using_cached_default_stream = (
        CACHE_DEFAULT_VOICE
        and task.voice.name == DEFAULT_VOICE
        and worker_index < len(worker_default_streams)
        and worker_default_streams[worker_index] is not None
    )
    print(
        f"[web_stream_service] task start worker={worker_index} voice={task.voice.name} cached_stream={using_cached_default_stream}",
        flush=True,
    )

    stream_started_at = time.perf_counter()
    if using_cached_default_stream:
        stream = worker_default_streams[worker_index]
    else:
        stream = engine.create_stream()
        if stream is None:
            raise RuntimeError("Unable to create TTS stream.")
    log_stream_timing("stream ready", stream_started_at)

    set_voice_started_at = time.perf_counter()
    if using_cached_default_stream:
        voice_loaded = True
    elif CACHE_DEFAULT_VOICE and task.voice.name == DEFAULT_VOICE and default_voice_result is not None:
        voice_loaded = stream.set_voice(copy.deepcopy(default_voice_result))
    else:
        if task.voice.kind == "audio":
            if not task.voice.ref_text:
                raise RuntimeError(f"Missing reference text for audio voice: {task.voice.name}")
            voice_loaded = stream.set_voice(str(task.voice.path), task.voice.ref_text)
        else:
            voice_loaded = stream.set_voice(str(task.voice.path))
    log_stream_timing(
        f"set_voice done ok={bool(voice_loaded)} mode={'fixed-stream' if using_cached_default_stream else 'dynamic'}",
        set_voice_started_at,
    )

    if not voice_loaded:
        raise RuntimeError(f"Failed to load voice anchor: {task.voice.path}")

    config = TTSConfig(
        max_steps=400,
        temperature=0.6,
        sub_temperature=0.6,
        seed=42,
        sub_seed=45,
        streaming=True,
        playback=False,
    )
    sample_rate = max(1000, int(round(SAMPLE_RATE * task.speed)))
    task_id = next_service_task_id(stream)
    print(
        f"[web_stream_service] clone start worker={worker_index} task_id={task_id} sample_rate={sample_rate}",
        flush=True,
    )
    result_holder: dict[str, object] = {"result": None, "error": None, "done": False}

    def run_clone() -> None:
        try:
            result_holder["result"] = stream.clone(text=task.text, language="Chinese", config=config)
        except Exception as exc:
            result_holder["error"] = exc
        finally:
            result_holder["done"] = True

    clone_thread = threading.Thread(target=run_clone, daemon=True)
    clone_started_at = time.perf_counter()
    acquire_decoder_silence()
    clone_thread.start()
    emitted_count = 0
    emitted_samples = 0
    header_sent = False
    first_chunk_logged = False

    try:
        while True:
            responses = stream.decoder.get_decode_result(task_id).responses
            new_responses = responses[emitted_count:]
            emitted_count = len(responses)

            if not header_sent and (new_responses or result_holder["done"]):
                task.chunks.put(wav_stream_header(sample_rate))
                header_sent = True

            wrote_chunk = False
            for response in new_responses:
                if response.msg_type != "AUDIO" or response.audio is None or len(response.audio) == 0:
                    continue
                payload = pcm_to_int16_bytes(response.audio)
                if payload:
                    if not first_chunk_logged:
                        log_stream_timing("first audio chunk ready", clone_started_at)
                        first_chunk_logged = True
                    emitted_samples += len(response.audio)
                    wrote_chunk = True
                    task.chunks.put(payload)

            if result_holder["done"]:
                break

            if not wrote_chunk:
                time.sleep(0.02)

        clone_thread.join()
        log_stream_timing("clone thread finished", clone_started_at)
        if result_holder["error"] is not None:
            raise RuntimeError(str(result_holder["error"]))

        result = result_holder["result"]
        if result is None or result.audio is None:
            raise RuntimeError("Synthesis returned no audio.")

        result.print_stats()

        full_audio = np.asarray(result.audio, dtype=np.float32)
        if not header_sent:
            task.chunks.put(wav_stream_header(sample_rate))
        if emitted_samples < full_audio.shape[0]:
            tail = pcm_to_int16_bytes(full_audio[emitted_samples:])
            if tail:
                task.chunks.put(tail)
        print(
            f"[web_stream_service] task finished worker={worker_index} task_id={task_id} total_ms={(time.perf_counter() - total_started_at) * 1000.0:.1f}",
            flush=True,
        )
    finally:
        try:
            stream.join(timeout=5.0)
        except Exception:
            pass
        release_decoder_silence()


def stream_worker(worker_index: int) -> None:
    print(f"[web_stream_service] worker ready index={worker_index}", flush=True)
    while True:
        try:
            task = request_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if task is None:
            request_queue.task_done()
            print(f"[web_stream_service] worker exit index={worker_index}", flush=True)
            break

        try:
            process_stream_task(task, worker_index)
        except Exception as exc:
            task.chunks.put(exc)
        finally:
            task.chunks.put(STREAM_END)
            request_queue.task_done()


def chunk_generator(task: StreamTask):
    while True:
        item = task.chunks.get()
        if item is STREAM_END:
            break
        if isinstance(item, Exception):
            raise item
        yield item


@asynccontextmanager
async def lifespan(_: FastAPI):
    global decoder_play_q_original, default_voice_result, engine, voice_map, request_queue, worker_default_streams, worker_threads

    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError(f"Model directory not found: {os.path.abspath(MODEL_DIR)}")

    voice_map = load_voice_map()
    if DEFAULT_VOICE not in voice_map:
        first_voice = sorted(voice_map)[0]
        raise RuntimeError(
            f"Default voice `{DEFAULT_VOICE}` not found. Try one of: {', '.join(sorted(voice_map))}. "
            f"For example, set QWEN_TTS_VOICE={first_voice}"
        )

    engine = TTSEngine(model_dir=MODEL_DIR, onnx_provider=ONNX_PROVIDER)
    if not engine or not engine.ready:
        raise RuntimeError("TTS engine failed to initialize.")

    print(
        f"[web_stream_service] startup workers={MAX_CONCURRENT_STREAMS} cache_default_voice={CACHE_DEFAULT_VOICE}",
        flush=True,
    )
    decoder_play_q_original = engine.decoder.play_q

    default_voice_result = None
    worker_default_streams = []
    if CACHE_DEFAULT_VOICE:
        default_voice_item = resolve_voice(DEFAULT_VOICE)
        warmup_stream = engine.create_stream()
        if warmup_stream is None:
            raise RuntimeError("Unable to create warmup stream for default voice cache.")

        if default_voice_item.kind == "audio":
            if not default_voice_item.ref_text:
                raise RuntimeError(f"Missing reference text for default voice: {default_voice_item.name}")
            default_voice_result = warmup_stream.set_voice(
                str(default_voice_item.path),
                default_voice_item.ref_text,
            )
        else:
            default_voice_result = warmup_stream.set_voice(str(default_voice_item.path))

        if not default_voice_result:
            raise RuntimeError(f"Failed to warm up default voice cache: {default_voice_item.path}")

        print(
            f"[web_stream_service] default voice cache ready name={DEFAULT_VOICE}",
            flush=True,
        )

        worker_default_streams = []
        for idx in range(MAX_CONCURRENT_STREAMS):
            worker_stream = engine.create_stream()
            if worker_stream is None:
                raise RuntimeError(f"Unable to create cached stream for worker {idx}.")
            voice_loaded = worker_stream.set_voice(copy.deepcopy(default_voice_result))
            if not voice_loaded:
                raise RuntimeError(f"Failed to bind cached default voice to worker stream {idx}.")
            worker_default_streams.append(worker_stream)

        print(
            f"[web_stream_service] worker default streams ready count={len(worker_default_streams)} name={DEFAULT_VOICE}",
            flush=True,
        )

    request_queue = queue.Queue()
    worker_threads = []
    for worker_index in range(MAX_CONCURRENT_STREAMS):
        worker = threading.Thread(target=stream_worker, args=(worker_index,), daemon=True)
        worker.start()
        worker_threads.append(worker)
    yield

    for _ in worker_threads:
        request_queue.put(None)
    for worker in worker_threads:
        worker.join(timeout=2.0)
    worker_threads = []

    if engine is not None:
        engine.shutdown()
        engine = None
    default_voice_result = None
    worker_default_streams = []
    decoder_play_q_original = None


app = FastAPI(title="Qwen3-TTS Stream Service", lifespan=lifespan)


@app.post("/api/stream")
async def api_stream(request: StreamRequest):
    request_started_at = time.perf_counter()
    try:
        voice = resolve_voice(request.voice)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    task = StreamTask(
        text=request.text,
        speed=request.seed,
        voice=voice,
        chunks=queue.Queue(maxsize=32),
    )
    request_queue.put(task)
    print(
        f"[web_stream_service] enqueue voice={voice.name} queue_size={request_queue.qsize()} total_ms={(time.perf_counter() - request_started_at) * 1000.0:.1f}",
        flush=True,
    )

    return StreamingResponse(
        content=chunk_generator(task),
        media_type="audio/wav",
        headers={
            "Content-Type": "audio/wav",
            "X-Voice-Name": voice.name,
            "X-Voice-Cache": "default" if CACHE_DEFAULT_VOICE and voice.name == DEFAULT_VOICE else "none",
            "X-Queue-Mode": f"fifo-{MAX_CONCURRENT_STREAMS}",
            "Cache-Control": "no-store",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_stream_service:app", host="0.0.0.0", port=8210, reload=False)
