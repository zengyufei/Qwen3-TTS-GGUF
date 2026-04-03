import os
import time
from pathlib import Path

from qwen3_tts_gguf.inference import TTSEngine

MODEL_DIR = os.environ.get("QWEN_TTS_MODEL_DIR", "model-base-small")
VOICE_DIR = Path(os.environ.get("QWEN_TTS_EXTRA_VOICE_DIR", os.path.join("output", "voices")))
OUTPUT_DIR = Path(os.environ.get("QWEN_TTS_ELABORATE_DIR", os.path.join("output", "elaborate")))
ONNX_PROVIDER = os.environ.get("QWEN_TTS_ONNX_PROVIDER", "CUDA")
OVERWRITE = os.environ.get("QWEN_TTS_OVERWRITE_JSON", "0") == "1"


def iter_voice_files():
    for ext in ("*.mp3", "*.m4a", "*.wav", "*.flac", "*.opus"):
        for path in sorted(VOICE_DIR.glob(ext)):
            yield path


def main():
    if not VOICE_DIR.is_dir():
        raise FileNotFoundError(f"Voice directory not found: {VOICE_DIR.resolve()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[convert] model_dir={Path(MODEL_DIR).resolve()}")
    print(f"[convert] voice_dir={VOICE_DIR.resolve()}")
    print(f"[convert] output_dir={OUTPUT_DIR.resolve()}")
    print(f"[convert] overwrite={OVERWRITE}")

    engine = TTSEngine(model_dir=MODEL_DIR, onnx_provider=ONNX_PROVIDER)
    if not engine or not engine.ready:
        raise RuntimeError("TTS engine failed to initialize.")

    converted = 0
    skipped = 0
    failed = 0
    total_started_at = time.perf_counter()

    try:
        for audio_path in iter_voice_files():
            txt_path = audio_path.with_suffix(".txt")
            out_path = OUTPUT_DIR / f"{audio_path.stem}.json"

            if not txt_path.is_file():
                print(f"[convert] skip {audio_path.name}: missing {txt_path.name}")
                skipped += 1
                continue

            if out_path.exists() and not OVERWRITE:
                print(f"[convert] skip {audio_path.name}: {out_path.name} already exists")
                skipped += 1
                continue

            ref_text = txt_path.read_text(encoding="utf-8").strip()
            if not ref_text:
                print(f"[convert] skip {audio_path.name}: empty reference text")
                skipped += 1
                continue

            print(f"[convert] start {audio_path.name}")
            item_started_at = time.perf_counter()

            try:
                stream = engine.create_stream()
                if stream is None:
                    raise RuntimeError("Unable to create TTS stream.")

                set_voice_started_at = time.perf_counter()
                result = stream.set_voice(str(audio_path), ref_text)
                print(
                    f"[convert] set_voice done name={audio_path.stem} elapsed_ms={(time.perf_counter() - set_voice_started_at) * 1000.0:.1f}"
                )

                if not result:
                    raise RuntimeError("set_voice returned False")

                save_started_at = time.perf_counter()
                result.save_json(str(out_path))
                print(
                    f"[convert] save_json done name={audio_path.stem} elapsed_ms={(time.perf_counter() - save_started_at) * 1000.0:.1f}"
                )

                converted += 1
                print(
                    f"[convert] finished name={audio_path.stem} total_ms={(time.perf_counter() - item_started_at) * 1000.0:.1f}"
                )
            except Exception as exc:
                failed += 1
                print(f"[convert] failed {audio_path.name}: {type(exc).__name__}: {exc}")
    finally:
        engine.shutdown()

    print(
        f"[convert] summary converted={converted} skipped={skipped} failed={failed} total_ms={(time.perf_counter() - total_started_at) * 1000.0:.1f}"
    )


if __name__ == "__main__":
    main()
