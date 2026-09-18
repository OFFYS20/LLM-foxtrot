"""Real telemetry via pynvml (GPU) + psutil (CPU/RAM/disk).

Both imports are optional: if pynvml is missing or no NVIDIA device is present
the provider still reports CPU/RAM/disk and marks ``compute_mode="cpu"``.
"""

from __future__ import annotations

import platform
from datetime import datetime, timezone

from app.core.logging import get_logger
from app.schemas.hardware import DiskInfo, GPUInfo, HardwareSnapshot
from app.services.hardware.base import HardwareProvider

logger = get_logger("foxtrot.hardware.nvml")

try:  # pragma: no cover - environment dependent
    import psutil
except Exception:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

try:  # pragma: no cover - environment dependent
    import pynvml
except Exception:  # pragma: no cover
    pynvml = None  # type: ignore[assignment]


def nvml_available() -> bool:
    if pynvml is None:
        return False
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        return count > 0
    except Exception:
        return False


def torch_compute_mode() -> str:
    """Detect the accelerator without importing torch unless it is installed."""
    try:  # pragma: no cover - environment dependent
        import torch

        if torch.cuda.is_available():
            return "rocm" if getattr(torch.version, "hip", None) else "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cuda" if nvml_available() else "cpu"


class NvmlHardwareProvider(HardwareProvider):
    name = "nvml"
    provenance = "measured"

    def __init__(self) -> None:
        self._nvml_ready = False
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._nvml_ready = True
            except Exception as exc:  # pragma: no cover
                logger.info("NVML unavailable: %s", exc)

    def _gpus(self) -> list[GPUInfo]:
        if not self._nvml_ready or pynvml is None:
            return []
        gpus: list[GPUInfo] = []
        try:
            driver = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver, bytes):
                driver = driver.decode()
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpus.append(
                    GPUInfo(
                        index=index,
                        name=name,
                        utilization=float(util.gpu),
                        memory_used_mb=mem.used / 1024**2,
                        memory_total_mb=mem.total / 1024**2,
                        temperature_c=_safe(lambda h=handle: pynvml.nvmlDeviceGetTemperature(h, 0)),
                        power_draw_w=_safe(
                            lambda h=handle: pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                        ),
                        power_limit_w=_safe(
                            lambda h=handle: pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
                        ),
                        fan_speed_pct=_safe(
                            lambda h=handle: float(pynvml.nvmlDeviceGetFanSpeed(h))
                        ),
                        clock_mhz=_safe(
                            lambda h=handle: float(pynvml.nvmlDeviceGetClockInfo(h, 0))
                        ),
                        driver_version=driver,
                        processes=len(
                            _safe(
                                lambda h=handle: pynvml.nvmlDeviceGetComputeRunningProcesses(h),
                                default=[],
                            )
                            or []
                        ),
                    )
                )
        except Exception as exc:  # pragma: no cover
            logger.warning("NVML sample failed: %s", exc)
        return gpus

    def snapshot(self) -> HardwareSnapshot:
        gpus = self._gpus()
        cpu_percent = psutil.cpu_percent(interval=None) if psutil else 0.0
        vm = psutil.virtual_memory() if psutil else None
        sm = psutil.swap_memory() if psutil else None
        disks: list[DiskInfo] = []
        if psutil:
            for part in psutil.disk_partitions(all=False)[:4]:
                try:
                    usage = psutil.disk_usage(part.mountpoint)
                except Exception:  # pragma: no cover
                    continue
                disks.append(
                    DiskInfo(
                        mount=part.mountpoint,
                        used_gb=usage.used / 1024**3,
                        total_gb=usage.total / 1024**3,
                        percent=usage.percent,
                    )
                )
        mode = torch_compute_mode()
        return HardwareSnapshot(
            ts=datetime.now(timezone.utc),
            provenance="measured",
            compute_mode=mode,  # type: ignore[arg-type]
            gpu_available=bool(gpus),
            gpus=gpus,
            cpu_percent=cpu_percent,
            cpu_cores=(psutil.cpu_count(logical=True) or 0) if psutil else 0,
            cpu_model=platform.processor() or platform.machine(),
            ram_used_gb=(vm.used / 1024**3) if vm else 0.0,
            ram_total_gb=(vm.total / 1024**3) if vm else 0.0,
            swap_used_gb=(sm.used / 1024**3) if sm else 0.0,
            swap_total_gb=(sm.total / 1024**3) if sm else 0.0,
            disks=disks,
            platform=f"{platform.system()} {platform.release()}",
            driver_version=gpus[0].driver_version if gpus else None,
            note=None if gpus else "No GPU detected — the platform is running in CPU mode.",
        )

    def close(self) -> None:  # pragma: no cover
        if self._nvml_ready and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def _safe(fn, default=None):  # pragma: no cover - thin helper
    try:
        return fn()
    except Exception:
        return default
