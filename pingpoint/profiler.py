import platform
import subprocess
import json
from typing import Optional

from pingpoint.models import Profile
from pingpoint.backend import detect_backend, Backend, OllamaBackend


def get_backend_models() -> list[str]:
    backend = detect_backend()
    if backend:
        return backend.list_models()
    return []


def is_backend_running() -> bool:
    return detect_backend() is not None


def get_backend_version() -> str:
    backend = detect_backend()
    if backend:
        return backend.version()
    return "not running"


def get_ollama_models() -> list[str]:
    try:
        return OllamaBackend().list_models()
    except Exception:
        return []


def is_ollama_running() -> bool:
    try:
        return OllamaBackend.probe()
    except Exception:
        return False


def get_ollama_version() -> str:
    try:
        return OllamaBackend().version()
    except Exception:
        return "unknown"


def get_gpu_info() -> tuple[list[str], Optional[float]]:
    gpus = []
    vram = None

    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10
            )
            for line in result.stdout.strip().split("\n")[1:]:
                line = line.strip()
                if line and "Name" not in line:
                    gpus.append(line)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    elif platform.system() == "Linux":
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        parts = line.split(",")
                        name = parts[0].strip()
                        gpus.append(name)
                        if len(parts) > 1:
                            vram_str = parts[1].strip().replace(" MiB", "")
                            try:
                                vram_gb = int(vram_str) / 1024
                                if vram is None:
                                    vram = vram_gb
                            except ValueError:
                                pass
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    elif platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10
            )
            for line in result.stdout.split("\n"):
                if "Chip" in line or "Processor" in line:
                    gpus.append(line.split(":")[-1].strip())
            try:
                result = subprocess.run(
                    ["sysctl", "hw.memsize"],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5
                )
                mem_bytes = int(result.stdout.strip().split()[-1])
                vram = mem_bytes / (1024 ** 3)
            except (ValueError, subprocess.TimeoutExpired):
                pass
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return gpus, vram


def get_cpu_info() -> tuple[str, int]:
    system = platform.system()
    cores = 0
    cpu_name = platform.processor() or "Unknown"

    if system == "Windows":
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10
            )
            lines = [l.strip() for l in result.stdout.split("\n") if l.strip()]
            if len(lines) > 1:
                cpu_name = lines[1]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "NumberOfCores"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10
            )
            lines = [l.strip() for l in result.stdout.split("\n") if l.strip()]
            if len(lines) > 1:
                cores = int(lines[1])
        except (ValueError, FileNotFoundError, subprocess.TimeoutExpired):
            cores = 0
    else:
        try:
            result = subprocess.run(
                ["nproc" if system == "Linux" else "sysctl", "-n",
                 "hw.ncpu" if system == "Darwin" else ""],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5
            )
            if result.stdout.strip():
                cores = int(result.stdout.strip())
        except (ValueError, FileNotFoundError, subprocess.TimeoutExpired):
            pass

    if cores == 0:
        cores = platform.machine().count("64") + 2

    return cpu_name, cores


def get_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass

    system = platform.system()
    if system == "Linux":
        try:
            result = subprocess.run(
                ["free", "-b"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5
            )
            for line in result.stdout.split("\n"):
                if line.startswith("Mem:"):
                    parts = line.split()
                    return int(parts[1]) / (1024 ** 3)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    elif system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "hw.memsize"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5
            )
            mem_bytes = int(result.stdout.strip().split()[-1])
            return mem_bytes / (1024 ** 3)
        except (ValueError, subprocess.TimeoutExpired):
            pass

    return 8.0


def profile() -> Profile:
    gpus, vram = get_gpu_info()
    cpu_name, cpu_cores = get_cpu_info()
    ram = get_ram_gb()

    backend = detect_backend()
    backend_models = backend.list_models() if backend else []
    backend_running = backend is not None

    score = 0.0
    if gpus and gpus[0] != "Unknown":
        score += 40.0
        if vram and vram >= 16:
            score += 20.0
        elif vram and vram >= 8:
            score += 10.0
    score += min(cpu_cores * 2.0, 20.0)
    score += min(ram / 2.0, 10.0)
    score += min(len(backend_models) * 5.0, 10.0)

    return Profile(
        platform=platform.system(),
        cpu=cpu_name,
        cpu_cores=cpu_cores,
        ram_gb=round(ram, 1),
        gpu=gpus if gpus else ["Unknown"],
        vram_gb=round(vram, 1) if vram else None,
        ollama_models=backend_models,
        ollama_running=backend_running,
        score=round(score, 1),
    )
