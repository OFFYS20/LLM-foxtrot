"""Deterministic-ish simulated telemetry for demo mode.

Values wander smoothly (Ornstein–Uhlenbeck style) so charts look alive without
ever pretending to be measured: ``provenance`` is always ``"simulated"``.
"""

from __future__ import annotations

import math
import platform
import random
import time
from datetime import datetime, timezone

from app.schemas.hardware import DiskInfo, GPUInfo, HardwareSnapshot
from app.services.hardware.base import HardwareProvider

DEMO_GPUS = [
    {"name": "NVIDIA GeForce RTX 4090 (demo)", "vram": 24_564, "power_limit": 450.0},
    {"name": "NVIDIA RTX A6000 (demo)", "vram": 49_140, "power_limit": 300.0},
]


class SimulatedHardwareProvider(HardwareProvider):
    name = "simulated"
    provenance = "simulated"

    def __init__(self, gpu_count: int = 2, seed: int = 7) -> None:
        self._rng = random.Random(seed)
        self._t0 = time.time()
        self._gpu_count = max(0, min(gpu_count, len(DEMO_GPUS)))
        self._state = [
            {"util": 62.0, "mem": spec["vram"] * 0.55, "temp": 61.0, "power": 240.0}
            for spec in DEMO_GPUS[: self._gpu_count]
        ]
        self._cpu = 24.0
        self._ram = 18.0
        self._load_bias = 0.0

    def set_load_bias(self, bias: float) -> None:
        """Training/benchmark runs push utilisation up while they are active."""
        self._load_bias = max(0.0, min(1.0, bias))

    def _wander(self, value: float, target: float, volatility: float, lo: float, hi: float):
        pull = (target - value) * 0.18
        noise = self._rng.uniform(-volatility, volatility)
        return max(lo, min(hi, value + pull + noise))

    def snapshot(self) -> HardwareSnapshot:
        elapsed = time.time() - self._t0
        wave = (math.sin(elapsed / 9.0) + 1) / 2  # 0..1 slow breathing

        gpus: list[GPUInfo] = []
        for index, spec in enumerate(DEMO_GPUS[: self._gpu_count]):
            st = self._state[index]
            util_target = 30 + 55 * wave + 35 * self._load_bias - index * 8
            st["util"] = self._wander(st["util"], util_target, 3.5, 0.0, 100.0)
            mem_target = spec["vram"] * (0.35 + 0.45 * self._load_bias + 0.1 * wave)
            st["mem"] = self._wander(
                st["mem"], mem_target, spec["vram"] * 0.004, 512.0, spec["vram"] * 0.98
            )
            st["temp"] = self._wander(st["temp"], 45 + st["util"] * 0.38, 0.6, 30.0, 88.0)
            st["power"] = self._wander(
                st["power"],
                80 + spec["power_limit"] * (st["util"] / 130.0),
                6.0,
                40.0,
                spec["power_limit"],
            )
            gpus.append(
                GPUInfo(
                    index=index,
                    name=spec["name"],
                    utilization=round(st["util"], 1),
                    memory_used_mb=round(st["mem"], 1),
                    memory_total_mb=float(spec["vram"]),
                    temperature_c=round(st["temp"], 1),
                    power_draw_w=round(st["power"], 1),
                    power_limit_w=spec["power_limit"],
                    fan_speed_pct=round(min(100.0, 25 + st["temp"] * 0.7), 1),
                    clock_mhz=round(1600 + st["util"] * 9, 0),
                    compute_capability="8.9" if index == 0 else "8.6",
                    driver_version="550.107.02 (demo)",
                    processes=1 if self._load_bias > 0.1 else 0,
                )
            )

        self._cpu = self._wander(self._cpu, 15 + 45 * self._load_bias + 18 * wave, 4.0, 1.0, 100.0)
        self._ram = self._wander(self._ram, 12 + 22 * self._load_bias, 0.6, 2.0, 62.0)

        return HardwareSnapshot(
            ts=datetime.now(timezone.utc),
            provenance="simulated",
            compute_mode="cuda" if gpus else "cpu",
            gpu_available=bool(gpus),
            gpus=gpus,
            cpu_percent=round(self._cpu, 1),
            cpu_cores=32,
            cpu_model="AMD Ryzen 9 7950X (demo)",
            cpu_temperature_c=round(38 + self._cpu * 0.25, 1),
            ram_used_gb=round(self._ram, 1),
            ram_total_gb=64.0,
            swap_used_gb=round(1.2 + self._load_bias, 1),
            swap_total_gb=16.0,
            disks=[
                DiskInfo(mount="/", used_gb=412.0, total_gb=1024.0, percent=40.2),
                DiskInfo(mount="/mnt/models", used_gb=1830.0, total_gb=3840.0, percent=47.7),
            ],
            platform=f"{platform.system()} {platform.release()} (demo telemetry)",
            driver_version="550.107.02 (demo)",
            cuda_version="12.4 (demo)",
            note="Simulated hardware telemetry — no physical device is being read.",
        )
