"""Pre-flight checks for local single-GPU training.

The doctor deliberately performs no CUDA allocation and writes only one small,
temporary file.  Its report is JSON serialisable so callers can show it in a
CLI, save it with a run manifest, or refuse to start training.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

GIB = 1024**3


class DoctorError(RuntimeError):
    """Raised when a required pre-flight check did not pass."""


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    """One independently actionable pre-flight check."""

    name: str
    ok: bool
    message: str
    required: bool = True
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DoctorReport:
    """Structured result returned by :func:`run_doctor`."""

    checks: tuple[DoctorCheck, ...]

    @property
    def ready(self) -> bool:
        return all(check.ok for check in self.checks if check.required)

    @property
    def failed(self) -> tuple[DoctorCheck, ...]:
        return tuple(c for c in self.checks if c.required and not c.ok)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "checks": [check.to_dict() for check in self.checks],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def require_ready(self) -> None:
        if not self.ready:
            failures = "; ".join(f"{c.name}: {c.message}" for c in self.failed)
            raise DoctorError(f"training pre-flight failed: {failures}")


def _memory_available_bytes() -> int | None:
    """Return available physical RAM without making psutil a hard dependency."""

    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.virtual_memory().available)
    except (ImportError, AttributeError, OSError):
        pass

    if os.name == "posix" and hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (ValueError, OSError):
            return None

    if os.name == "nt":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        try:
            windll = getattr(ctypes, "windll", None)
            if windll is not None and windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.available_physical)
        except (AttributeError, OSError):
            return None
    return None


def _nearest_existing_directory(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists() and candidate.parent != candidate:
        candidate = candidate.parent
    if not candidate.is_dir():
        raise NotADirectoryError(candidate)
    return candidate


def _benchmark_sequential_io(directory: Path, size_bytes: int) -> dict[str, float]:
    """Measure sequential I/O using a temporary file that is always removed."""

    payload = b"\0" * min(size_bytes, 1024 * 1024)
    remaining = size_bytes
    file_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
            file_name = stream.name
            started = time.perf_counter()
            while remaining:
                block = payload[: min(remaining, len(payload))]
                stream.write(block)
                remaining -= len(block)
            stream.flush()
            os.fsync(stream.fileno())
            write_seconds = max(time.perf_counter() - started, 1e-9)

        read_bytes = 0
        started = time.perf_counter()
        with open(file_name, "rb") as stream:
            while block := stream.read(1024 * 1024):
                read_bytes += len(block)
        read_seconds = max(time.perf_counter() - started, 1e-9)
        mib = size_bytes / 1024**2
        return {
            "size_mib": round(mib, 3),
            "write_mib_s": round(mib / write_seconds, 3),
            "read_mib_s": round((read_bytes / 1024**2) / read_seconds, 3),
        }
    finally:
        if file_name:
            with suppress(OSError):
                Path(file_name).unlink(missing_ok=True)


def _platform_check() -> DoctorCheck:
    system = platform.system()
    release = platform.release()
    version = platform.version()
    is_linux = system == "Linux"
    is_wsl = is_linux and (
        "microsoft" in release.lower()
        or "microsoft" in version.lower()
        or bool(os.environ.get("WSL_INTEROP"))
    )
    if is_wsl:
        message = "WSL2/Linux environment detected"
    elif is_linux:
        message = "native Linux environment detected (WSL2 is recommended on Windows)"
    else:
        message = f"{system} detected; run training inside WSL2 or Linux"
    return DoctorCheck(
        "platform",
        is_linux,
        message,
        details={"system": system, "release": release, "is_wsl": is_wsl},
    )


def _python_check() -> DoctorCheck:
    version = platform.python_version()
    supported = sys.version_info[:2] == (3, 12)
    return DoctorCheck(
        "python",
        supported,
        f"Python {version}"
        if supported
        else f"Python {version} is unsupported; install Python 3.12",
        details={"version": version, "required_series": "3.12"},
    )


def _torch_checks(min_free_vram_gb: float) -> Iterable[DoctorCheck]:
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:
        yield DoctorCheck("torch", False, f"PyTorch import failed: {error}")
        return

    yield DoctorCheck(
        "torch",
        True,
        f"PyTorch {torch.__version__}",
        details={"version": torch.__version__},
    )
    cuda_available = bool(torch.cuda.is_available())
    cuda_details: dict[str, Any] = {"available": cuda_available}
    cuda_message = "CUDA is not available"
    if cuda_available:
        try:
            cuda_details.update(
                name=torch.cuda.get_device_name(0),
                capability=list(torch.cuda.get_device_capability(0)),
            )
            cuda_message = f"CUDA device: {cuda_details['name']}"
        except (AssertionError, RuntimeError) as error:
            cuda_available = False
            cuda_message = f"CUDA query failed: {error}"
    yield DoctorCheck("cuda", cuda_available, cuda_message, details=cuda_details)

    bf16_ok = False
    if cuda_available:
        try:
            bf16_ok = bool(torch.cuda.is_bf16_supported())
        except (AttributeError, RuntimeError):
            capability = cuda_details.get("capability", [0, 0])
            bf16_ok = bool(capability and int(capability[0]) >= 8)
    yield DoctorCheck(
        "bf16",
        bf16_ok,
        "CUDA BF16 is supported" if bf16_ok else "CUDA BF16 is unavailable",
    )

    muon = getattr(torch.optim, "Muon", None)
    yield DoctorCheck(
        "muon",
        callable(muon),
        "torch.optim.Muon is available"
        if callable(muon)
        else "torch.optim.Muon is unavailable; install the pinned PyTorch version",
    )
    sdpa_ok = callable(getattr(functional, "scaled_dot_product_attention", None))
    yield DoctorCheck(
        "sdpa",
        sdpa_ok,
        "scaled-dot-product attention is available"
        if sdpa_ok
        else "torch.nn.functional.scaled_dot_product_attention is unavailable",
    )
    compile_ok = callable(getattr(torch, "compile", None))
    yield DoctorCheck(
        "torch_compile",
        compile_ok,
        "torch.compile is available" if compile_ok else "torch.compile is unavailable",
    )

    free_bytes = 0
    total_bytes = 0
    if cuda_available:
        try:
            free_bytes, total_bytes = (int(v) for v in torch.cuda.mem_get_info(0))
        except (AttributeError, RuntimeError, TypeError):
            try:
                props = torch.cuda.get_device_properties(0)
                total_bytes = int(props.total_memory)
                free_bytes = max(total_bytes - int(torch.cuda.memory_reserved(0)), 0)
            except (AttributeError, RuntimeError):
                pass
    vram_ok = cuda_available and free_bytes >= min_free_vram_gb * GIB
    yield DoctorCheck(
        "vram",
        vram_ok,
        (
            f"{free_bytes / GIB:.2f} GiB CUDA memory free"
            if cuda_available
            else "CUDA memory cannot be queried"
        ),
        details={
            "free_gib": round(free_bytes / GIB, 3),
            "total_gib": round(total_bytes / GIB, 3),
            "minimum_free_gib": min_free_vram_gb,
        },
    )


def run_doctor(
    data_dir: str | os.PathLike[str] | None = None,
    *,
    min_free_disk_gb: float = 50.0,
    min_free_ram_gb: float = 4.0,
    min_free_vram_gb: float = 0.7,
    io_test_bytes: int = 4 * 1024 * 1024,
) -> DoctorReport:
    """Run all required checks without allocating model or CUDA tensors."""

    if min(min_free_disk_gb, min_free_ram_gb, min_free_vram_gb) < 0:
        raise ValueError("minimum resource thresholds must be non-negative")
    if io_test_bytes <= 0:
        raise ValueError("io_test_bytes must be positive")

    target = Path(data_dir) if data_dir is not None else Path.cwd()
    checks: list[DoctorCheck] = [_platform_check(), _python_check()]
    checks.extend(_torch_checks(min_free_vram_gb))

    try:
        directory = _nearest_existing_directory(target)
        usage = shutil.disk_usage(directory)
        disk_ok = usage.free >= min_free_disk_gb * GIB
        checks.append(
            DoctorCheck(
                "disk",
                disk_ok,
                f"{usage.free / GIB:.2f} GiB disk space free at {directory}",
                details={
                    "path": str(directory),
                    "free_gib": round(usage.free / GIB, 3),
                    "total_gib": round(usage.total / GIB, 3),
                    "minimum_free_gib": min_free_disk_gb,
                },
            )
        )
        metrics = _benchmark_sequential_io(directory, io_test_bytes)
        checks.append(
            DoctorCheck(
                "data_io",
                True,
                "data directory is writable and sequential I/O succeeded",
                details={"path": str(directory), **metrics},
            )
        )
    except (OSError, ValueError) as error:
        checks.append(DoctorCheck("disk", False, f"disk query failed: {error}"))
        checks.append(DoctorCheck("data_io", False, f"data directory is not writable: {error}"))

    ram_bytes = _memory_available_bytes()
    ram_ok = ram_bytes is not None and ram_bytes >= min_free_ram_gb * GIB
    checks.append(
        DoctorCheck(
            "ram",
            ram_ok,
            (
                f"{ram_bytes / GIB:.2f} GiB RAM available"
                if ram_bytes is not None
                else "available RAM could not be determined"
            ),
            details={
                "available_gib": round(ram_bytes / GIB, 3) if ram_bytes is not None else None,
                "minimum_free_gib": min_free_ram_gb,
            },
        )
    )
    return DoctorReport(tuple(checks))


def require_training_ready(
    report: DoctorReport | None = None,
    **doctor_options: Any,
) -> DoctorReport:
    """Return a passing report or raise :class:`DoctorError`."""

    checked = report if report is not None else run_doctor(**doctor_options)
    checked.require_ready()
    return checked
