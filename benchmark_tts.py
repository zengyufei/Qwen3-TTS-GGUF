import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Case:
    chunk_size: int
    workers: int
    ort_intra: int
    ort_inter: int
    result_poll_s: float
    proxy_poll_s: float


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def build_cases(args: argparse.Namespace) -> list[Case]:
    return [
        Case(
            chunk_size=combo[0],
            workers=1,
            ort_intra=combo[1],
            ort_inter=combo[2],
            result_poll_s=combo[3],
            proxy_poll_s=combo[4],
        )
        for combo in itertools.product(
            parse_int_list(args.chunk_sizes),
            parse_int_list(args.ort_intra_threads),
            parse_int_list(args.ort_inter_threads),
            parse_float_list(args.result_poll_seconds),
            parse_float_list(args.proxy_poll_seconds),
        )
    ]


def is_server_ready(host: str, port: int, timeout_s: float, proc: subprocess.Popen | None = None) -> bool:
    url = f"http://{host}:{port}/docs"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if 200 <= resp.status < 500:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def post_tts(host: str, port: int, token: str, text: str) -> tuple[float, float, int, float, float]:
    url = f"http://{host}:{port}/api/tts"
    body = json.dumps({"text": text, "token": token}).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            ttfb_ms = (time.perf_counter() - t0) * 1000.0
            total_bytes = 0
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                total_bytes += len(chunk)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body[:500]}") from e
    total_ms = (time.perf_counter() - t0) * 1000.0

    pcm_bytes = max(0, total_bytes - 44)
    audio_sec = pcm_bytes / 2.0 / 24000.0
    rtf = (total_ms / 1000.0) / audio_sec if audio_sec > 0 else float("nan")
    return ttfb_ms, total_ms, total_bytes, audio_sec, rtf


def spawn_server(
    python_exe: str,
    workdir: Path,
    case: Case,
    port: int,
) -> tuple[subprocess.Popen, Path, object]:
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = workdir / f"benchmark_server_{ts}_{int(time.time() * 1000) % 1000:03d}.log"
    log_fp = log_path.open("w", encoding="utf-8", errors="replace", buffering=1)

    env = os.environ.copy()
    zlib_candidates = [
        workdir / "zlib123dllx64" / "dll_x64",
        workdir / "zlib123dllx64",
    ]
    zlib_path = next((p for p in zlib_candidates if (p / "zlibwapi.dll").exists()), None)
    if zlib_path is not None:
        env["PATH"] = str(zlib_path) + os.pathsep + env.get("PATH", "")

    env.update(
        {
            "QWEN_TTS_PORT": str(port),
            "QWEN_TTS_ONNX_PROVIDER": "CUDA",
            "QWEN_TTS_STREAM_WORKERS": "1",
            "QWEN_TTS_CHUNK_SIZE": str(case.chunk_size),
            "QWEN_TTS_ORT_ALLOW_SPINNING": "1",
            "QWEN_TTS_ORT_INTRA_OP_THREADS": str(case.ort_intra),
            "QWEN_TTS_ORT_INTER_OP_THREADS": str(case.ort_inter),
            "QWEN_TTS_RESULT_POLL_INTERVAL_S": str(case.result_poll_s),
            "QWEN_TTS_PROXY_LISTENER_POLL_S": str(case.proxy_poll_s),
            "QWEN_TTS_ENGINE_READY_TIMEOUT": "none",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        }
    )

    proc = subprocess.Popen(
        [python_exe, "-u", "web.py"],
        cwd=str(workdir),
        env=env,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, log_path, log_fp


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def kill_listeners_on_port(port: int) -> None:
    try:
        out = subprocess.check_output(["netstat", "-ano"], text=True, encoding="utf-8", errors="replace")
    except Exception:
        return
    pids: set[int] = set()
    needle = f":{port}"
    for line in out.splitlines():
        s = line.strip()
        if "LISTENING" not in s or needle not in s:
            continue
        parts = s.split()
        if len(parts) >= 5:
            try:
                pids.add(int(parts[-1]))
            except ValueError:
                pass
    for pid in sorted(pids):
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False, capture_output=True)
            time.sleep(0.2)
        except Exception:
            pass


def write_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tail_text_file(path: Path, max_lines: int = 60) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        tail = "".join(lines[-max_lines:])
        return tail.strip()
    except Exception:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="One-click benchmark for web.py TTS service")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8210)
    parser.add_argument("--token", default="zengleqi")
    parser.add_argument("--text", default="benchmark request for tts latency and rtf")
    parser.add_argument("--chunk-sizes", default="24,32,48")
    parser.add_argument("--workers", default="1")
    parser.add_argument("--ort-intra-threads", default="0,4")
    parser.add_argument("--ort-inter-threads", default="0")
    parser.add_argument("--result-poll-seconds", default="0.005")
    parser.add_argument("--proxy-poll-seconds", default="0.02")
    parser.add_argument("--repeat-per-case", type=int, default=2)
    parser.add_argument("--warmup-per-case", action="store_true")
    args = parser.parse_args()

    workdir = Path.cwd()
    cases = build_cases(args)
    print(f"Total benchmark cases: {len(cases)}")

    results: list[dict] = []
    for idx, case in enumerate(cases, start=1):
        print(
            f"[{idx}/{len(cases)}] chunk={case.chunk_size} workers={case.workers} "
            f"intra={case.ort_intra} inter={case.ort_inter} "
            f"resultPoll={case.result_poll_s} proxyPoll={case.proxy_poll_s}"
        )

        proc = None
        log_path = None
        log_fp = None
        try:
            kill_listeners_on_port(args.port)
            proc, log_path, log_fp = spawn_server(args.python_exe, workdir, case, args.port)
            if not is_server_ready(args.host, args.port, timeout_s=120, proc=proc):
                raise RuntimeError(f"Server failed to become ready. log={log_path}")

            if args.warmup_per_case:
                post_tts(args.host, args.port, args.token, "hi")

            for rep in range(1, args.repeat_per_case + 1):
                ttfb_ms, total_ms, total_bytes, audio_sec, rtf = post_tts(
                    args.host, args.port, args.token, args.text
                )
                results.append(
                    {
                        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "case_index": idx,
                        "repeat": rep,
                        "chunk_size": case.chunk_size,
                        "workers": case.workers,
                        "ort_intra": case.ort_intra,
                        "ort_inter": case.ort_inter,
                        "result_poll_s": case.result_poll_s,
                        "proxy_poll_s": case.proxy_poll_s,
                        "ttfb_ms": round(ttfb_ms, 1),
                        "total_ms": round(total_ms, 1),
                        "audio_sec": round(audio_sec, 3),
                        "rtf": round(rtf, 3) if rtf == rtf else "",
                        "bytes": total_bytes,
                        "server_log": str(log_path),
                        "error": "",
                    }
                )
                print(
                    f"  run#{rep}: ttfb={ttfb_ms:.1f}ms total={total_ms:.1f}ms "
                    f"audio={audio_sec:.3f}s rtf={rtf:.3f}"
                )
        except Exception as e:
            log_tail = tail_text_file(log_path) if log_path else ""
            err_msg = str(e)
            if log_tail:
                print("  --- server log tail ---")
                print(log_tail)
            results.append(
                {
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "case_index": idx,
                    "repeat": -1,
                    "chunk_size": case.chunk_size,
                    "workers": case.workers,
                    "ort_intra": case.ort_intra,
                    "ort_inter": case.ort_inter,
                    "result_poll_s": case.result_poll_s,
                    "proxy_poll_s": case.proxy_poll_s,
                    "ttfb_ms": "",
                    "total_ms": "",
                    "audio_sec": "",
                    "rtf": "",
                    "bytes": "",
                    "server_log": str(log_path) if log_path else "",
                    "error": err_msg,
                }
            )
            print(f"  failed: {err_msg}")
        finally:
            if proc is not None:
                stop_server(proc)
            if log_fp is not None:
                try:
                    log_fp.close()
                except Exception:
                    pass
            kill_listeners_on_port(args.port)
            time.sleep(20.0)

    out_csv = workdir / f"benchmark_results_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    write_csv(results, out_csv)
    print(f"\nSaved: {out_csv}")

    ok_rows = [r for r in results if r["total_ms"] != ""]
    if ok_rows:
        best = sorted(ok_rows, key=lambda r: float(r["total_ms"]))[:10]
        print("\nTop 10 by total_ms:")
        for i, row in enumerate(best, start=1):
            print(
                f"{i:2d}. total={row['total_ms']}ms ttfb={row['ttfb_ms']}ms "
                f"rtf={row['rtf']} chunk={row['chunk_size']} workers={row['workers']} "
                f"intra={row['ort_intra']} inter={row['ort_inter']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
