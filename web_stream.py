import copy
import asyncio
import os
import queue
import threading
import time
import wave
from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from qwen3_tts_gguf.inference import TTSEngine, TTSConfig

SAMPLE_RATE = 24000
# Default runtime tuning from benchmark best-RFT case:
# chunk_size=24, workers=1, ORT intra=0, inter=0
os.environ.setdefault("QWEN_TTS_ORT_INTRA_OP_THREADS", "0")
os.environ.setdefault("QWEN_TTS_ORT_INTER_OP_THREADS", "0")
MODEL_DIR = os.environ.get("QWEN_TTS_MODEL_DIR", "model-base-small")
VOICE_DIR = Path(os.environ.get("QWEN_TTS_VOICE_DIR", os.path.join("output", "elaborate")))
EXTRA_VOICE_DIR = Path(os.environ.get("QWEN_TTS_EXTRA_VOICE_DIR", os.path.join("output", "voices")))
DEFAULT_VOICE = os.environ.get("QWEN_TTS_VOICE", "Vivian")
ONNX_PROVIDER = os.environ.get("QWEN_TTS_ONNX_PROVIDER", "CUDA")
MAX_CONCURRENT_STREAMS = 1
CHUNK_SIZE = int(os.environ.get("QWEN_TTS_CHUNK_SIZE", "24"))
STARTUP_WARMUP = os.environ.get("QWEN_TTS_STARTUP_WARMUP", "1") == "1"
STARTUP_WARMUP_TEXT = os.environ.get("QWEN_TTS_STARTUP_WARMUP_TEXT", "hi")
API_STREAM_TOKEN = os.environ.get("QWEN_TTS_STREAM_TOKEN", "zengleqi")
OUTPUT_GAIN = float(os.environ.get("QWEN_TTS_OUTPUT_GAIN", "1.0"))
UVICORN_KEEP_ALIVE_SECONDS = 2147483647

engine: Optional[TTSEngine] = None
voice_map: dict[str, "VoiceItem"] = {}
request_queue: "queue.Queue[Optional[TTSTask]]"
worker_threads: list[threading.Thread] = []
worker_streams: list[object] = []
worker_stream_voice_names: list[str] = []
default_voice_result = None
tts_config_singleton: Optional[TTSConfig] = None
decoder_play_q_original = None
decoder_play_q_silent_users = 0
decoder_play_q_lock = threading.Lock()


@dataclass(frozen=True)
class VoiceItem:
    name: str
    path: Path
    kind: str
    ref_text: Optional[str] = None


@dataclass
class TTSTask:
    text: str
    speed: float
    voice: VoiceItem
    enqueued_at: float
    result_queue: "queue.Queue[object]"


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    seed: float = Field(1.0, gt=0)
    voice: Optional[str] = None
    token: str = ""


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
    pcm = apply_output_gain(pcm)
    return (pcm * 32767.0).astype("<i2").tobytes()


def apply_output_gain(audio: np.ndarray) -> np.ndarray:
    pcm = np.asarray(audio, dtype=np.float32)
    if pcm.size == 0:
        return pcm

    if OUTPUT_GAIN != 1.0:
        pcm = pcm * OUTPUT_GAIN

    peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
    if peak > 0.98:
        pcm = pcm * (0.98 / peak)

    return np.clip(pcm, -1.0, 1.0)


def pcm_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    wav_io = BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_to_int16_bytes(audio))
    return wav_io.getvalue()


def log_timing(label: str, started_at: float) -> float:
    now = time.perf_counter()
    print(f"[web_tts] {label} elapsed_ms={(now - started_at) * 1000.0:.1f}", flush=True)
    return now


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


def load_voice_into_stream(stream, voice: VoiceItem) -> bool:
    if voice.name == DEFAULT_VOICE and default_voice_result is not None:
        return bool(stream.set_voice(copy.deepcopy(default_voice_result)))
    if voice.kind == "audio":
        if not voice.ref_text:
            raise RuntimeError(f"Missing reference text for audio voice: {voice.name}")
        return bool(stream.set_voice(str(voice.path), voice.ref_text))
    return bool(stream.set_voice(str(voice.path)))


def build_tts_config() -> TTSConfig:
    global tts_config_singleton

    desired_values = {
        "max_steps": 400,
        "temperature": 0.85,
        "sub_temperature": 0.85,
        "seed": 65,
        "sub_seed": 65,
        "streaming": False,
        "playback": False,
    }
    if tts_config_singleton is None:
        tts_config_singleton = TTSConfig(**desired_values)
        return tts_config_singleton

    changed = False
    for key, value in desired_values.items():
        if getattr(tts_config_singleton, key) != value:
            setattr(tts_config_singleton, key, value)
            changed = True
    if changed:
        print("[web_tts] tts config updated", flush=True)
    return tts_config_singleton


def create_bound_stream(voice: VoiceItem, purpose: str):
    if engine is None:
        raise RuntimeError("TTS engine is not initialized.")
    stream = engine.create_stream()
    if stream is None:
        raise RuntimeError(f"Unable to create stream for {purpose}.")
    voice_loaded = load_voice_into_stream(stream, voice)
    if not voice_loaded:
        raise RuntimeError(f"Failed to bind voice anchor for {purpose}: {voice.path}")
    return stream


def prepare_default_voice_result(voice: VoiceItem):
    if engine is None:
        raise RuntimeError("TTS engine is not initialized.")
    stream = engine.create_stream()
    if stream is None:
        raise RuntimeError("Unable to create default voice warmup stream.")
    try:
        if voice.kind == "audio":
            if not voice.ref_text:
                raise RuntimeError(f"Missing reference text for default voice: {voice.name}")
            result = stream.set_voice(str(voice.path), voice.ref_text)
        else:
            result = stream.set_voice(str(voice.path))
        if not result:
            raise RuntimeError(f"Failed to prepare default voice: {voice.path}")
        return result
    finally:
        try:
            stream.shutdown()
        except Exception:
            pass


def run_startup_warmup() -> None:
    warmup_voice = resolve_voice(DEFAULT_VOICE)
    warmup_stream = create_bound_stream(warmup_voice, "startup warmup")
    started_at = time.perf_counter()
    print(
        f"[web_tts] startup warmup start text={STARTUP_WARMUP_TEXT!r} voice={warmup_voice.name}",
        flush=True,
    )
    try:
        result = warmup_stream.clone(
            text=(STARTUP_WARMUP_TEXT.strip() or "hi"),
            language="Chinese",
            config=build_tts_config(),
        )
        if result is None or result.audio is None or len(result.audio) == 0:
            raise RuntimeError("Startup warmup returned no audio.")
        print(
            f"[web_tts] startup warmup done total_ms={(time.perf_counter() - started_at) * 1000.0:.1f}",
            flush=True,
        )
    finally:
        try:
            warmup_stream.shutdown()
        except Exception:
            pass


def process_tts_task(task: TTSTask, worker_index: int) -> None:
    total_started_at = time.perf_counter()
    wait_ms = (total_started_at - task.enqueued_at) * 1000.0
    print(f"[web_tts] task start worker={worker_index} voice={task.voice.name} wait_ms={wait_ms:.1f}", flush=True)

    stream_started_at = time.perf_counter()
    stream = worker_streams[worker_index]
    current_voice_name = worker_stream_voice_names[worker_index]
    if current_voice_name != task.voice.name:
        voice_loaded = load_voice_into_stream(stream, task.voice)
        if not voice_loaded:
            raise RuntimeError(f"Failed to switch worker {worker_index} voice to {task.voice.name}")
        worker_stream_voice_names[worker_index] = task.voice.name
        print(f"[web_tts] worker stream rebound index={worker_index} voice={task.voice.name}", flush=True)
    else:
        print(f"[web_tts] worker stream reuse index={worker_index} voice={task.voice.name}", flush=True)
    log_timing("request stream ready", stream_started_at)

    config = build_tts_config()
    sample_rate = max(1000, int(round(SAMPLE_RATE * task.speed)))
    clone_started_at = time.perf_counter()
    acquire_decoder_silence()
    try:
        result = stream.clone(text=task.text, language="Chinese", config=config)
        log_timing("clone finished", clone_started_at)
        if result is None or result.audio is None or len(result.audio) == 0:
            raise RuntimeError("Synthesis returned no audio.")
        wav_bytes = pcm_to_wav_bytes(np.asarray(result.audio, dtype=np.float32), sample_rate)
        task.result_queue.put(wav_bytes)
        print(
            f"[web_tts] task finished worker={worker_index} run_ms={(time.perf_counter() - total_started_at) * 1000.0:.1f} total_ms={(time.perf_counter() - task.enqueued_at) * 1000.0:.1f}",
            flush=True,
        )
    finally:
        release_decoder_silence()


def tts_worker(worker_index: int) -> None:
    print(f"[web_tts] worker ready index={worker_index}", flush=True)
    while True:
        try:
            task = request_queue.get()
        except Exception:
            continue

        if task is None:
            request_queue.task_done()
            print(f"[web_tts] worker exit index={worker_index}", flush=True)
            break

        try:
            process_tts_task(task, worker_index)
        except Exception as exc:
            task.result_queue.put(exc)
        finally:
            request_queue.task_done()


async def parse_flexible_request_payload(request: Request, route_name: str) -> dict[str, object]:
    content_type = request.headers.get("content-type", "")
    query_data = dict(request.query_params)
    raw_body = await request.body()

    parsed_form: dict[str, object] = {}
    parsed_json: Optional[object] = None
    fallback_form: dict[str, object] = {}

    if "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
        try:
            parsed_form = dict(await request.form())
        except Exception as exc:
            parsed_form = {"__form_parse_error__": str(exc)}

    if "application/json" in content_type:
        try:
            parsed_json = await request.json()
        except Exception as exc:
            parsed_json = {"__json_parse_error__": str(exc)}

    if not content_type and raw_body:
        raw_text = raw_body.decode("utf-8", errors="replace")
        if "=" in raw_text:
            parsed_qs = parse_qs(raw_text, keep_blank_values=True)
            fallback_form = {key: values[-1] if values else "" for key, values in parsed_qs.items()}

    body_preview = raw_body[:300].decode("utf-8", errors="replace")
    print(
        f"[web_tts] {route_name} request method={request.method} content_type={content_type!r} query={query_data} form={parsed_form} json={parsed_json} fallback_form={fallback_form} body_preview={body_preview!r}",
        flush=True,
    )

    payload: dict[str, object] = {}
    if query_data:
        payload.update(query_data)
    if isinstance(parsed_json, dict):
        payload.update(parsed_json)
    if parsed_form:
        payload.update(parsed_form)
    if fallback_form:
        payload.update(fallback_form)
    return payload


@asynccontextmanager
async def lifespan(_: FastAPI):
    global decoder_play_q_original, default_voice_result, engine, request_queue
    global tts_config_singleton, voice_map, worker_stream_voice_names, worker_streams, worker_threads

    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError(f"Model directory not found: {os.path.abspath(MODEL_DIR)}")

    voice_map = load_voice_map()
    if DEFAULT_VOICE not in voice_map:
        first_voice = sorted(voice_map)[0]
        raise RuntimeError(
            f"Default voice `{DEFAULT_VOICE}` not found. Try one of: {', '.join(sorted(voice_map))}. "
            f"For example, set QWEN_TTS_VOICE={first_voice}"
        )

    engine = TTSEngine(model_dir=MODEL_DIR, onnx_provider=ONNX_PROVIDER, chunk_size=CHUNK_SIZE)
    if not engine or not engine.ready:
        raise RuntimeError("TTS engine failed to initialize.")

    print(
        f"[web_tts] startup workers={MAX_CONCURRENT_STREAMS} startup_warmup={STARTUP_WARMUP} default_voice={DEFAULT_VOICE} output_gain={OUTPUT_GAIN} chunk_size={CHUNK_SIZE} ort_intra={os.environ.get('QWEN_TTS_ORT_INTRA_OP_THREADS')} ort_inter={os.environ.get('QWEN_TTS_ORT_INTER_OP_THREADS')}",
        flush=True,
    )
    decoder_play_q_original = engine.decoder.play_q
    default_voice_result = prepare_default_voice_result(resolve_voice(DEFAULT_VOICE))
    print(f"[web_tts] default voice ready name={DEFAULT_VOICE}", flush=True)

    if STARTUP_WARMUP:
        run_startup_warmup()

    request_queue = queue.Queue()
    worker_streams = []
    worker_stream_voice_names = []
    for worker_index in range(MAX_CONCURRENT_STREAMS):
        worker_stream = create_bound_stream(resolve_voice(DEFAULT_VOICE), f"worker {worker_index} startup")
        worker_streams.append(worker_stream)
        worker_stream_voice_names.append(DEFAULT_VOICE)
    print(
        f"[web_tts] worker bound streams ready count={len(worker_streams)} voice={DEFAULT_VOICE}",
        flush=True,
    )

    worker_threads = []
    for worker_index in range(MAX_CONCURRENT_STREAMS):
        worker = threading.Thread(target=tts_worker, args=(worker_index,), daemon=True)
        worker.start()
        worker_threads.append(worker)
    yield

    for _ in worker_threads:
        request_queue.put(None)
    for worker in worker_threads:
        worker.join()
    worker_threads = []

    for worker_stream in worker_streams:
        try:
            worker_stream.shutdown()
        except Exception:
            pass
    worker_streams = []
    worker_stream_voice_names = []

    if engine is not None:
        engine.shutdown()
        engine = None
    default_voice_result = None
    tts_config_singleton = None
    decoder_play_q_original = None


app = FastAPI(title="Qwen3-TTS Web TTS Service", lifespan=lifespan)


@app.post("/api/tts")
async def api_tts(request: Request):
    payload = await parse_flexible_request_payload(request, "/api/tts")

    text = str(payload.get("text", "")).strip()
    voice_name = payload.get("voice")
    voice_name = str(voice_name).strip() if voice_name is not None and str(voice_name).strip() else None
    token = str(payload.get("token", ""))
    try:
        seed = float(payload.get("seed", 1.0))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="seed must be a number")

    if API_STREAM_TOKEN and token != API_STREAM_TOKEN:
        raise HTTPException(status_code=404, detail="Not Found")
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")
    if seed <= 0:
        raise HTTPException(status_code=400, detail="seed must be greater than 0")

    request_started_at = time.perf_counter()
    try:
        voice = resolve_voice(voice_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    task = TTSTask(
        text=text,
        speed=seed,
        voice=voice,
        enqueued_at=time.perf_counter(),
        result_queue=queue.Queue(maxsize=1),
    )
    request_queue.put(task)
    print(
        f"[web_tts] enqueue voice={voice.name} text={text!r} queue_size={request_queue.qsize()} total_ms={(time.perf_counter() - request_started_at) * 1000.0:.1f}",
        flush=True,
    )

    result = await asyncio.to_thread(task.result_queue.get)
    if isinstance(result, Exception):
        raise HTTPException(status_code=500, detail=str(result))

    return Response(
        content=result,
        media_type="audio/wav",
        headers={
            "Content-Type": "audio/wav",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "web_tts:app",
        host="0.0.0.0",
        port=8210,
        reload=False,
        timeout_keep_alive=UVICORN_KEEP_ALIVE_SECONDS,
        timeout_graceful_shutdown=None,
    )
