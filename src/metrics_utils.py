from __future__ import annotations

import csv
import json
import math
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import torch


def _device_index_of(device: torch.device | str | int) -> Optional[int]:
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        if ":" in device:
            try:
                return int(device.split(":", 1)[1])
            except Exception:
                return None
        return 0 if device == "cuda" else None
    if isinstance(device, torch.device):
        return _device_index_of(str(device))
    return None


class PowerSampler:
    """NVML-based GPU power sampler with trapezoidal integration."""

    def __init__(self, device: torch.device | str | int = 0, poll_ms: int = 50):
        self.poll_s = max(1, int(poll_ms)) / 1000.0
        self.samples: List[tuple[float, float]] = []  # (t, watts)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._enabled = False
        self._handle = None
        self._t0 = 0.0

        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            idx = _device_index_of(device)
            if idx is None:
                idx = 0
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            self._pynvml = pynvml
            self._enabled = True
        except Exception:
            self._enabled = False
            self._pynvml = None
            try:
                print("[WARN] NVML unavailable; energy_J will be recorded as 0. Install pynvml and ensure GPU access.")
            except Exception:
                pass

    def start(self) -> float:
        self.samples.clear()
        self._stop.clear()
        self._t0 = time.perf_counter()
        if not self._enabled:
            return self._t0

        def _run():
            while not self._stop.is_set():
                try:
                    now = time.perf_counter()
                    mw = self._pynvml.nvmlDeviceGetPowerUsage(self._handle)  # type: ignore
                    self.samples.append((now, mw / 1000.0))  # W
                except Exception:
                    pass
                time.sleep(self.poll_s)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        return self._t0

    def stop_and_energy(self, t0: Optional[float] = None) -> float:
        if not self._enabled:
            return 0.0
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        s = list(self.samples)
        if not s:
            return 0.0
        # Prepend t0 with first power value if provided for better coverage
        if t0 is not None and s and s[0][0] > t0:
            s.insert(0, (t0, s[0][1]))
        # Trapezoidal integration over time, in Joules
        energy = 0.0
        for i in range(1, len(s)):
            t_prev, p_prev = s[i - 1]
            t_cur, p_cur = s[i]
            dt = max(0.0, t_cur - t_prev)
            energy += 0.5 * (p_prev + p_cur) * dt
        return energy


class Percentiles:
    @staticmethod
    def compute(values: Sequence[float], pctls: Sequence[int]) -> Dict[str, float]:
        if not values:
            return {f"p{p}": 0.0 for p in pctls}
        arr = sorted(values)
        out: Dict[str, float] = {}
        for p in pctls:
            if not arr:
                out[f"p{p}"] = 0.0
                continue
            k = (len(arr) - 1) * (p / 100.0)
            f = math.floor(k)
            c = min(len(arr) - 1, math.ceil(k))
            if f == c:
                out[f"p{p}"] = float(arr[int(k)])
            else:
                d0 = arr[f] * (c - k)
                d1 = arr[c] * (k - f)
                out[f"p{p}"] = float(d0 + d1)
        return out


@dataclass
class SampleRecord:
    sample_id: int
    stage: str
    latency_s: float
    peak_vram_gb: float
    energy_J: float
    is_warmup: bool


class MetricsSink:
    def __init__(self) -> None:
        self.rows: List[SampleRecord] = []

    def add(self, sample_id: int, stage: str, latency_s: float, peak_vram_gb: float, energy_J: float, is_warmup: bool) -> None:
        self.rows.append(SampleRecord(sample_id, stage, latency_s, peak_vram_gb, energy_J, is_warmup))

    def to_csv(self, out_path: str) -> None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["sample_id", "stage", "latency_s", "peak_vram_gb", "energy_J", "is_warmup"])
            for r in self.rows:
                w.writerow([r.sample_id, r.stage, f"{r.latency_s:.6f}", f"{r.peak_vram_gb:.6f}", f"{r.energy_J:.3f}", int(r.is_warmup)])

    def summarize(self, pctls: Sequence[int], warmup_mask_stage: Optional[str] = None) -> Dict[str, dict]:
        summary: Dict[str, dict] = {}
        stages = sorted(set(r.stage for r in self.rows))
        for st in stages:
            rows = [r for r in self.rows if r.stage == st]
            eff = [r for r in rows if not r.is_warmup]
            lat = [r.latency_s for r in eff]
            vram = [r.peak_vram_gb for r in eff]
            eng = [r.energy_J for r in eff]
            s = {
                "count_effective": len(eff),
                "latency": {
                    "mean": float(sum(lat) / len(lat)) if lat else 0.0,
                    "std": float((sum((x - (sum(lat)/len(lat)))**2 for x in lat) / len(lat)) ** 0.5) if len(lat) > 1 else 0.0,
                    **Percentiles.compute(lat, pctls),
                },
                "peak_vram_gb": {
                    "mean": float(sum(vram) / len(vram)) if vram else 0.0,
                    "max": float(max(vram)) if vram else 0.0,
                },
                "energy_J": {
                    "mean": float(sum(eng) / len(eng)) if eng else 0.0,
                    "sum": float(sum(eng)) if eng else 0.0,
                },
            }
            summary[st] = s
        return summary


@contextmanager
def stage_meter(stage: str, sink: MetricsSink, sample_id: int, device: torch.device | str | int = 0, power: Optional[PowerSampler] = None, is_warmup: bool = False):
    dev = device
    if torch.cuda.is_available():
        try:
            idx = _device_index_of(dev)
            if idx is not None:
                with torch.cuda.device(idx):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
            else:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    t0 = time.perf_counter()
    t_power0 = power.start() if power else None
    try:
        yield
    finally:
        if torch.cuda.is_available():
            try:
                idx = _device_index_of(dev)
                if idx is not None:
                    with torch.cuda.device(idx):
                        torch.cuda.synchronize()
                        peak_bytes = torch.cuda.max_memory_allocated()
                else:
                    torch.cuda.synchronize()
                    peak_bytes = torch.cuda.max_memory_allocated()
            except Exception:
                peak_bytes = 0
        else:
            peak_bytes = 0
        t1 = time.perf_counter()
        energy = power.stop_and_energy(t_power0) if power else 0.0
        sink.add(sample_id, stage, t1 - t0, peak_bytes / (1024**3), energy, is_warmup)


def env_info() -> Dict[str, str]:
    info = {
        "torch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda or "",
        "device_count": str(torch.cuda.device_count()),
    }
    if torch.cuda.is_available():
        try:
            idx = torch.cuda.current_device()
            info.update(
                {
                    "gpu_name": torch.cuda.get_device_name(idx),
                    "gpu_capability": str(torch.cuda.get_device_capability(idx)),
                    "driver": os.environ.get("NVIDIA_DRIVER_VERSION", ""),
                }
            )
        except Exception:
            pass
    return info
