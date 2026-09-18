"""Hardware detection and live telemetry.

Real readings only. When a counter cannot be read the field is ``None`` and the
UI renders "Unavailable" — the application never invents a number.
"""

from __future__ import annotations

import os
import platform
import shutil
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

from ai_studio.core import logging as log

try:  # optional
    import psutil
except Exception:  # noqa: BLE001 - pragma: no cover
    psutil = None  # type: ignore[assignment]

try:  # optional
    import pynvml
except Exception:  # noqa: BLE001 - pragma: no cover
    pynvml = None  # type: ignore[assignment]


@dataclass
class GPUInfo:
    index: int
    name: str
    total_memory_mb: float | None = None
    used_memory_mb: float | None = None
    free_memory_mb: float | None = None
    utilization: float | None = None
    temperature_c: float | None = None
    power_draw_w: float | None = None
    power_limit_w: float | None = None
    fan_percent: float | None = None
    compute_capability: str | None = None
    driver_version: str | None = None


@dataclass
class HardwareSnapshot:
    ts: float = field(default_factory=time.time)
    # --- compute -----------------------------------------------------------
    accelerator: str = "cpu"          # cuda | mps | cpu
    cuda_available: bool = False
    cuda_version: str | None = None
    torch_version: str | None = None
    gpus: list[GPUInfo] = field(default_factory=list)
    gpu_source: str | None = None     # nvml | torch | None
    # --- cpu / memory ------------------------------------------------------
    cpu_model: str | None = None
    cpu_cores_physical: int | None = None
    cpu_cores_logical: int | None = None
    cpu_percent: float | None = None
    ram_total_gb: float | None = None
    ram_used_gb: float | None = None
    ram_available_gb: float | None = None
    swap_total_gb: float | None = None
    swap_used_gb: float | None = None
    disk_total_gb: float | None = None
    disk_used_gb: float | None = None
    disk_free_gb: float | None = None
    platform: str = field(default_factory=lambda: f"{platform.system()} {platform.release()}")
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def primary_gpu(self) -> GPUInfo | None:
        return self.gpus[0] if self.gpus else None

    @property
    def total_vram_mb(self) -> float | None:
        values = [g.total_memory_mb for g in self.gpus if g.total_memory_mb]
        return sum(values) if values else None


class HardwareMonitor:
    """Samples the machine and keeps a short rolling history for the charts."""

    def __init__(self, history: int = 600) -> None:
        self._history: deque[HardwareSnapshot] = deque(maxlen=history)
        self._nvml_ready = False
        self._nvml_failed = False
        self._lock = threading.Lock()
        self._static: dict[str, Any] | None = None

    # ------------------------------------------------------------- helpers
    def _init_nvml(self) -> bool:
        if self._nvml_ready:
            return True
        if pynvml is None or self._nvml_failed:
            return False
        try:
            pynvml.nvmlInit()
            self._nvml_ready = True
        except Exception as exc:  # noqa: BLE001
            self._nvml_failed = True
            log.debug(f"NVML unavailable: {exc}", source="hardware", persist=False)
        return self._nvml_ready

    def _torch_info(self) -> dict[str, Any]:
        if self._static is not None:
            return self._static
        info: dict[str, Any] = {
            "torch_version": None,
            "cuda_available": False,
            "cuda_version": None,
            "accelerator": "cpu",
            "devices": [],
        }
        try:
            import torch

            info["torch_version"] = torch.__version__
            info["cuda_available"] = bool(torch.cuda.is_available())
            info["cuda_version"] = getattr(torch.version, "cuda", None)
            if info["cuda_available"]:
                info["accelerator"] = "rocm" if getattr(torch.version, "hip", None) else "cuda"
                for index in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(index)
                    info["devices"].append(
                        {
                            "index": index,
                            "name": props.name,
                            "total_memory_mb": props.total_memory / 1024**2,
                            "capability": f"{props.major}.{props.minor}",
                        }
                    )
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                info["accelerator"] = "mps"
                info["devices"].append({"index": 0, "name": "Apple Metal (MPS)"})
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001
            log.debug(f"torch probe failed: {exc}", source="hardware", persist=False)
        self._static = info
        return info

    def _read_gpus(self) -> tuple[list[GPUInfo], str | None]:
        # NVML gives live counters; torch only gives static properties.
        if self._init_nvml() and pynvml is not None:
            gpus: list[GPUInfo] = []
            try:
                driver = pynvml.nvmlSystemGetDriverVersion()
                driver = driver.decode() if isinstance(driver, bytes) else driver
                for index in range(pynvml.nvmlDeviceGetCount()):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                    name = pynvml.nvmlDeviceGetName(handle)
                    name = name.decode() if isinstance(name, bytes) else name
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = _try(lambda h=handle: pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
                    gpus.append(
                        GPUInfo(
                            index=index,
                            name=name,
                            total_memory_mb=memory.total / 1024**2,
                            used_memory_mb=memory.used / 1024**2,
                            free_memory_mb=memory.free / 1024**2,
                            utilization=float(util) if util is not None else None,
                            temperature_c=_try(lambda h=handle: float(pynvml.nvmlDeviceGetTemperature(h, 0))),
                            power_draw_w=_try(lambda h=handle: pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0),
                            power_limit_w=_try(
                                lambda h=handle: pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
                            ),
                            fan_percent=_try(lambda h=handle: float(pynvml.nvmlDeviceGetFanSpeed(h))),
                            driver_version=driver,
                        )
                    )
                if gpus:
                    return gpus, "nvml"
            except Exception as exc:  # noqa: BLE001
                log.debug(f"NVML read failed: {exc}", source="hardware", persist=False)

        torch_info = self._torch_info()
        if torch_info["devices"]:
            gpus = []
            for device in torch_info["devices"]:
                used = None
                try:
                    import torch

                    if torch_info["cuda_available"]:
                        used = torch.cuda.memory_reserved(device["index"]) / 1024**2
                except Exception:  # noqa: BLE001
                    used = None
                gpus.append(
                    GPUInfo(
                        index=device["index"],
                        name=device["name"],
                        total_memory_mb=device.get("total_memory_mb"),
                        used_memory_mb=used,
                        compute_capability=device.get("capability"),
                    )
                )
            return gpus, "torch"
        return [], None

    # -------------------------------------------------------------- public
    def snapshot(self) -> HardwareSnapshot:
        torch_info = self._torch_info()
        gpus, source = self._read_gpus()
        snapshot = HardwareSnapshot(
            accelerator=torch_info["accelerator"],
            cuda_available=torch_info["cuda_available"],
            cuda_version=torch_info["cuda_version"],
            torch_version=torch_info["torch_version"],
            gpus=gpus,
            gpu_source=source,
        )

        if psutil is not None:
            try:
                snapshot.cpu_percent = psutil.cpu_percent(interval=None)
                snapshot.cpu_cores_physical = psutil.cpu_count(logical=False)
                snapshot.cpu_cores_logical = psutil.cpu_count(logical=True)
                memory = psutil.virtual_memory()
                snapshot.ram_total_gb = memory.total / 1024**3
                snapshot.ram_used_gb = memory.used / 1024**3
                snapshot.ram_available_gb = memory.available / 1024**3
                swap = psutil.swap_memory()
                snapshot.swap_total_gb = swap.total / 1024**3
                snapshot.swap_used_gb = swap.used / 1024**3
            except Exception as exc:  # noqa: BLE001
                snapshot.notes.append(f"CPU/RAM telemetry unavailable: {exc}")
        else:
            snapshot.notes.append("psutil not installed — CPU and RAM readings unavailable.")

        snapshot.cpu_model = _cpu_model()

        try:
            usage = shutil.disk_usage(os.getcwd())
            snapshot.disk_total_gb = usage.total / 1024**3
            snapshot.disk_used_gb = usage.used / 1024**3
            snapshot.disk_free_gb = usage.free / 1024**3
        except Exception:  # noqa: BLE001
            pass

        if not snapshot.gpus:
            snapshot.notes.append(
                "No GPU detected — running in CPU mode. Small models train fine; "
                "anything above ~100M parameters will be very slow."
            )
        if torch_info["torch_version"] is None:
            snapshot.notes.append("PyTorch is not installed — training and inference are disabled.")

        with self._lock:
            self._history.append(snapshot)
        return snapshot

    def history(self, limit: int | None = None) -> list[HardwareSnapshot]:
        with self._lock:
            items = list(self._history)
        return items[-limit:] if limit else items

    def latest(self) -> HardwareSnapshot:
        with self._lock:
            if self._history:
                return self._history[-1]
        return self.snapshot()

    def describe(self) -> str:
        """One-line summary for the top bar."""
        snapshot = self.latest()
        gpu = snapshot.primary_gpu
        if gpu and gpu.total_memory_mb:
            return f"{gpu.name} · {gpu.total_memory_mb / 1024:.0f} GB VRAM"
        if gpu:
            return gpu.name
        cores = snapshot.cpu_cores_logical
        return f"CPU mode · {cores} threads" if cores else "CPU mode"


def _try(fn, default=None):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default


def _cpu_model() -> str | None:
    try:
        if platform.system() == "Linux":
            for line in open("/proc/cpuinfo", encoding="utf-8", errors="replace"):
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif platform.system() == "Darwin":
            import subprocess

            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or platform.machine() or None


monitor = HardwareMonitor()
