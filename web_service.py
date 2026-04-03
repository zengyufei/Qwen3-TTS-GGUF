import base64
import copy
import io
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from qwen3_tts_gguf.inference import TTSEngine, TTSConfig

SAMPLE_RATE = 24000
MODEL_DIR = os.environ.get("QWEN_TTS_MODEL_DIR", "model-base-small")
VOICE_DIR = Path(os.environ.get("QWEN_TTS_VOICE_DIR", os.path.join("output", "elaborate")))
EXTRA_VOICE_DIR = Path(os.environ.get("QWEN_TTS_EXTRA_VOICE_DIR", os.path.join("output", "voices")))
DEFAULT_VOICE = os.environ.get("QWEN_TTS_VOICE", "Vivian")
ONNX_PROVIDER = os.environ.get("QWEN_TTS_ONNX_PROVIDER", "CUDA")
CACHE_DEFAULT_VOICE = os.environ.get("QWEN_TTS_CACHE_DEFAULT_VOICE", "1") == "1"

engine: Optional[TTSEngine] = None
tts_lock = threading.Lock()
default_voice_result = None


@dataclass(frozen=True)
class VoiceItem:
    name: str
    path: Path
    kind: str
    ref_text: Optional[str] = None


voice_map: dict[str, VoiceItem] = {}


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="要合成的文本")
    seed: float = Field(1.0, gt=0, description="语速倍率，1.0 为正常")
    voice: Optional[str] = Field(None, description="音色名称")


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


def speed_change(audio: np.ndarray, speed: float) -> np.ndarray:
    if speed == 1.0:
        return audio.astype(np.float32, copy=False)
    original = np.asarray(audio, dtype=np.float32)
    if original.size == 0:
        return original
    new_length = max(1, int(round(original.shape[0] / speed)))
    old_index = np.arange(original.shape[0], dtype=np.float32)
    new_index = np.linspace(0, original.shape[0] - 1, new_length, dtype=np.float32)
    return np.interp(new_index, old_index, original).astype(np.float32)


def wav_bytes(audio: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, format="WAV")
    return buffer.getvalue()


def log_timing(label: str, started_at: float) -> float:
    now = time.perf_counter()
    elapsed_ms = (now - started_at) * 1000.0
    print(f"[web_service] {label} elapsed_ms={elapsed_ms:.1f}", flush=True)
    return now


def synthesize(text: str, speed: float, voice_name: Optional[str]) -> tuple[bytes, str]:
    if engine is None:
        raise RuntimeError("TTS engine is not initialized.")

    voice = resolve_voice(voice_name)
    total_started_at = time.perf_counter()
    print(f"[web_service] synthesize start voice={voice.name} kind={voice.kind}", flush=True)

    with tts_lock:
        lock_started_at = time.perf_counter()
        log_timing("lock acquired", lock_started_at)
        stream = engine.create_stream()
        if stream is None:
            raise RuntimeError("Unable to create TTS stream.")
        stream_created_at = time.perf_counter()
        log_timing("stream created", stream_created_at)

        if CACHE_DEFAULT_VOICE and voice.name == DEFAULT_VOICE and default_voice_result is not None:
            print(f"[web_service] set_voice cache start name={voice.name}", flush=True)
            set_voice_started_at = time.perf_counter()
            voice_loaded = stream.set_voice(copy.deepcopy(default_voice_result))
            log_timing(f"set_voice cache done ok={bool(voice_loaded)}", set_voice_started_at)
        else:
            if voice.kind == "audio":
                if not voice.ref_text:
                    raise RuntimeError(f"Missing reference text for audio voice: {voice.name}")
                print(f"[web_service] set_voice audio start path={voice.path}", flush=True)
                set_voice_started_at = time.perf_counter()
                voice_loaded = stream.set_voice(str(voice.path), voice.ref_text)
                log_timing(f"set_voice audio done ok={bool(voice_loaded)}", set_voice_started_at)
            else:
                print(f"[web_service] set_voice json start path={voice.path}", flush=True)
                set_voice_started_at = time.perf_counter()
                voice_loaded = stream.set_voice(str(voice.path))
                log_timing(f"set_voice json done ok={bool(voice_loaded)}", set_voice_started_at)

        if not voice_loaded:
            raise RuntimeError(f"Failed to load voice anchor: {voice.path}")

        config = TTSConfig(
            max_steps=400,
            temperature=0.6,
            sub_temperature=0.6,
            seed=42,
            sub_seed=45,
            streaming=False,
        )
        print("[web_service] clone start", flush=True)
        clone_started_at = time.perf_counter()
        result = stream.clone(text=text, language="Chinese", config=config)
        log_timing("clone done", clone_started_at)
        if result is None or result.audio is None:
            raise RuntimeError("Synthesis returned no audio.")

        print("[web_service] speed_change start", flush=True)
        speed_started_at = time.perf_counter()
        adjusted = speed_change(result.audio, speed)
        log_timing("speed_change done", speed_started_at)
        print("[web_service] wav encode start", flush=True)
        wav_started_at = time.perf_counter()
        audio_bytes = wav_bytes(adjusted)
        log_timing("wav encode done", wav_started_at)
        print(
            f"[web_service] synthesize finished total_ms={(time.perf_counter() - total_started_at) * 1000.0:.1f}",
            flush=True,
        )
        return audio_bytes, voice.name


@asynccontextmanager
async def lifespan(_: FastAPI):
    global default_voice_result, engine, voice_map

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
        f"[web_service] startup cache_default_voice={CACHE_DEFAULT_VOICE}",
        flush=True,
    )

    default_voice_result = None
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
            f"[web_service] default voice cache ready name={DEFAULT_VOICE}",
            flush=True,
        )
    yield
    if engine is not None:
        engine.shutdown()
        engine = None
    default_voice_result = None


app = FastAPI(title="Qwen3-TTS Web Service", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    options = "\n".join(
        f'<option value="{name}" {"selected" if name == DEFAULT_VOICE else ""}>{name}</option>'
        for name in sorted(voice_map)
    )
    default_text = "你好，这里是 Qwen3 TTS 在线测试页面。"
    return f"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Qwen3-TTS Web Service</title>
  <style>
    :root {{ color-scheme: light; }}
    body {{ font-family: "Microsoft YaHei", sans-serif; max-width: 860px; margin: 40px auto; padding: 0 16px; }}
    textarea, input, button, select {{ width: 100%; box-sizing: border-box; font-size: 16px; }}
    textarea {{ min-height: 180px; padding: 12px; }}
    input, button, select {{ padding: 10px 12px; margin-top: 12px; }}
    button {{ cursor: pointer; }}
    .row {{ margin-top: 18px; }}
    audio {{ width: 100%; margin-top: 16px; }}
    .hint {{ color: #555; font-size: 14px; }}
    pre {{ background: #f6f8fa; padding: 12px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>Qwen3-TTS 在线测试</h1>
  <p class="hint">这个服务只提供网页和普通网页接口，不包含 /api/stream。</p>
  <div class="row">
    <label for="voice">音色</label>
    <select id="voice">{options}</select>
  </div>
  <div class="row">
    <label for="cacheDefaultVoice">默认音色内存缓存</label>
    <input id="cacheDefaultVoice" type="checkbox" {"checked" if CACHE_DEFAULT_VOICE else ""} disabled />
    <div class="hint">当前服务端配置：{"已启用" if CACHE_DEFAULT_VOICE else "未启用"}。如需修改，请重启服务并设置环境变量 <code>QWEN_TTS_CACHE_DEFAULT_VOICE</code>。</div>
  </div>
  <div class="row">
    <label for="text">文本</label>
    <textarea id="text">{default_text}</textarea>
  </div>
  <div class="row">
    <label for="seed">语速倍率（seed）</label>
    <input id="seed" type="number" min="0.5" max="3" step="0.1" value="1" />
  </div>
  <div class="row">
    <button id="submit">生成并播放</button>
  </div>
  <audio id="player" controls></audio>
  <div class="row">
    <strong>网页接口</strong>
    <pre>POST /api/generate
Content-Type: application/json

{{"text":"文字","seed":1,"voice":"{DEFAULT_VOICE}"}}</pre>
  </div>
  <script>
    const btn = document.getElementById("submit");
    const player = document.getElementById("player");
    btn.addEventListener("click", async () => {{
      btn.disabled = true;
      btn.textContent = "生成中...";
      try {{
        const payload = {{
          text: document.getElementById("text").value,
          seed: Number(document.getElementById("seed").value || 1),
          voice: document.getElementById("voice").value
        }};
        const res = await fetch("/api/generate", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(payload)
        }});
        if (!res.ok) {{
          const err = await res.json();
          throw new Error(err.detail || "请求失败");
        }}
        const data = await res.json();
        player.src = data.audio_data_url;
        player.play();
      }} catch (err) {{
        alert(err.message || String(err));
      }} finally {{
        btn.disabled = false;
        btn.textContent = "生成并播放";
      }}
    }});
  </script>
</body>
</html>
"""


@app.get("/health")
async def health():
    return {
        "ok": True,
        "model_dir": MODEL_DIR,
        "voice_dir": str(VOICE_DIR),
        "extra_voice_dir": str(EXTRA_VOICE_DIR),
        "default_voice": DEFAULT_VOICE,
        "voices": sorted(voice_map),
        "engine_ready": bool(engine and engine.ready),
    }


@app.get("/api/voices")
async def api_voices():
    return {
        "ok": True,
        "default_voice": DEFAULT_VOICE,
        "voices": [
            {
                "name": item.name,
                "path": str(item.path),
                "kind": item.kind,
                "has_ref_text": bool(item.ref_text),
            }
            for _, item in sorted(voice_map.items())
        ],
    }


@app.post("/api/generate")
async def api_generate(request: TTSRequest):
    total_started_at = time.perf_counter()
    try:
        print("[web_service] api_generate start", flush=True)
        synth_started_at = time.perf_counter()
        audio, voice_name = synthesize(request.text, request.seed, request.voice)
        log_timing(f"api_generate synthesize done bytes={len(audio)} voice={voice_name}", synth_started_at)
    except Exception as exc:
        print(
            f"[web_service] api_generate error {type(exc).__name__}: {exc} total_ms={(time.perf_counter() - total_started_at) * 1000.0:.1f}",
            flush=True,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    print("[web_service] base64 encode start", flush=True)
    base64_started_at = time.perf_counter()
    encoded = base64.b64encode(audio).decode("ascii")
    log_timing(f"base64 encode done chars={len(encoded)}", base64_started_at)
    response_started_at = time.perf_counter()
    payload = {
        "ok": True,
        "contentType": "audio/wav",
        "seed": request.seed,
        "voice": voice_name,
        "audio_base64": encoded,
        "audio_data_url": f"data:audio/wav;base64,{encoded}",
    }
    log_timing("json response build done", response_started_at)
    print(
        f"[web_service] api_generate finished total_ms={(time.perf_counter() - total_started_at) * 1000.0:.1f}",
        flush=True,
    )
    return JSONResponse(
        payload,
        headers={
            "X-Voice-Cache": "default" if CACHE_DEFAULT_VOICE and voice_name == DEFAULT_VOICE else "none",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_service:app", host="0.0.0.0", port=8210, reload=False)
