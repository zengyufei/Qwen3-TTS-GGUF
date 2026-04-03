"""
audio.py - audio preprocessing helpers
Primary strategy:
1. Use soundfile for formats it handles well.
2. Use PyAV for MP4/M4A/AAC and other container-based inputs.
This avoids depending on an external ffmpeg executable.
"""

import math
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly as scipy_resample_poly

try:
    import av
except ImportError:  # pragma: no cover - optional dependency at import time
    av = None


def is_mp4_family_container(audio_path):
    """
    Detect ISO BMFF / MP4-family container by file header.
    Some files may use a .mp3 extension while actually being M4A/AAC containers.
    """
    try:
        with open(audio_path, "rb") as f:
            header = f.read(32)
    except OSError:
        return False
    return len(header) >= 12 and header[4:8] == b"ftyp"


def numpy_resample_poly(x, up, down, window_size=10):
    g = math.gcd(up, down)
    up //= g
    down //= g

    if up == down:
        return x.copy()

    max_rate = max(up, down)
    f_c = 1.0 / max_rate
    half_len = window_size * max_rate
    n_taps = 2 * half_len + 1

    t = np.arange(n_taps) - half_len
    h = np.sinc(f_c * t)

    beta = 5.0
    kaiser_win = np.i0(beta * np.sqrt(1 - (2 * t / (n_taps - 1)) ** 2)) / np.i0(beta)
    h = h * kaiser_win
    h = h * (up / np.sum(h))

    length_in = len(x)
    length_out = int(math.ceil(length_in * up / down))

    x_up = np.zeros(length_in * up + n_taps, dtype=np.float32)
    x_up[: length_in * up : up] = x

    y_full = np.convolve(x_up, h, mode="full")

    offset = (n_taps - 1) // 2
    y = y_full[offset : offset + length_in * up : down]

    return y[:length_out].astype(np.float32)


def resample_audio(audio, sr, target_sr):
    if sr == target_sr:
        return audio
    return scipy_resample_poly(audio, target_sr, sr).astype(np.float32, copy=False)


def load_audio_numpy(audio_path, sample_rate=24000, start_second=None, duration=None):
    info = sf.info(audio_path)
    sr = info.samplerate

    start_frame = int(start_second * sr) if start_second is not None else 0
    frames = int(duration * sr) if duration is not None else -1

    audio, sr = sf.read(audio_path, start=start_frame, frames=frames, dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != sample_rate:
        audio = resample_audio(audio, sr, sample_rate)

    return audio.astype(np.float32, copy=False)


def load_audio_pyav(audio_path, sample_rate=24000, start_second=None, duration=None):
    if av is None:
        raise RuntimeError(
            "PyAV is required to decode this audio file. Install it with `pip install av`."
        )

    audio_chunks = []
    source_rate = None

    container = av.open(str(audio_path))
    try:
        audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
        if audio_stream is None:
            raise RuntimeError(f"No audio stream found in file: {audio_path}")

        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)

        if start_second is not None:
            try:
                container.seek(int(start_second * av.time_base), any_frame=False, backward=True)
            except Exception:
                pass

        end_second = None if duration is None else (start_second or 0.0) + duration

        for frame in container.decode(audio_stream):
            source_rate = int(frame.sample_rate or audio_stream.rate or sample_rate)
            frame_start = float(frame.time or 0.0)
            frame_duration = frame.samples / float(source_rate)
            frame_end = frame_start + frame_duration

            if start_second is not None and frame_end <= start_second:
                continue
            if end_second is not None and frame_start >= end_second:
                break

            frame = resampler.resample(frame)
            if frame is None:
                continue
            if isinstance(frame, list):
                resampled_frames = frame
            else:
                resampled_frames = [frame]

            data_list = []
            for item in resampled_frames:
                data = item.to_ndarray()
                if data.ndim == 1:
                    mono = data.astype(np.float32, copy=False)
                else:
                    mono = data.astype(np.float32, copy=False).mean(axis=0)
                data_list.append(mono)

            if not data_list:
                continue

            mono = np.concatenate(data_list).astype(np.float32, copy=False)
            frame_rate = sample_rate
            frame_duration = mono.shape[0] / float(frame_rate)
            frame_end = frame_start + frame_duration

            if start_second is not None and frame_start < start_second:
                trim_start = int(round((start_second - frame_start) * frame_rate))
                mono = mono[max(0, trim_start) :]

            if end_second is not None and frame_end > end_second:
                keep = int(round((end_second - frame_start) * frame_rate))
                mono = mono[: max(0, keep)]

            if mono.size:
                audio_chunks.append(mono)
    finally:
        container.close()

    if not audio_chunks:
        return np.array([], dtype=np.float32)

    audio = np.concatenate(audio_chunks).astype(np.float32, copy=False)
    return audio.astype(np.float32, copy=False)


def load_audio(audio_path, sample_rate=24000, start_second=None, duration=None):
    """
    Main audio loader entry.
    - soundfile for stable native formats
    - PyAV for MP4/M4A/AAC and other container-based files
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    ext = Path(audio_path).suffix.lower()

    if is_mp4_family_container(audio_path):
        return load_audio_pyav(audio_path, sample_rate, start_second, duration)

    soundfile_formats = {".wav", ".flac", ".ogg", ".mp3"}
    if ext in soundfile_formats:
        return load_audio_numpy(audio_path, sample_rate, start_second, duration)

    return load_audio_pyav(audio_path, sample_rate, start_second, duration)
