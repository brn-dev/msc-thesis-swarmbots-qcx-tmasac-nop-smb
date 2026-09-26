from __future__ import annotations

import ctypes
import functools
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


try:
    import torch
except ImportError:
    torch = None


@functools.lru_cache(maxsize=1)
def collect_machine_specs() -> dict[str, Any]:
    return {
        "machine_name": socket.gethostname(),
        "fqdn": socket.getfqdn(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "cpu": _collect_cpu_specs(),
        "memory": _collect_memory_specs(),
        "gpus": _collect_gpu_specs(),
        "torch": _collect_torch_specs(),
    }


def _collect_cpu_specs() -> dict[str, Any]:
    system = platform.system()
    specs: dict[str, Any] = {
        "name": platform.processor() or None,
        "logical_cores": os.cpu_count(),
    }

    if system == "Windows":
        windows_cpu = _collect_windows_cpu_specs()
        if windows_cpu:
            specs.update(windows_cpu)
    elif system == "Linux":
        specs.update(_collect_linux_cpu_specs())
    elif system == "Darwin":
        specs.update(_collect_darwin_cpu_specs())

    return _drop_none_values(specs)


def _collect_windows_cpu_specs() -> dict[str, Any]:
    output = _run_command(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Processor | "
            "Select-Object Name,Manufacturer,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | "
            "ConvertTo-Json -Compress",
        ]
    )
    data = _parse_json_object_or_list(output)
    if not data:
        return {}

    cpu = data[0]
    return _drop_none_values(
        {
            "name": _clean_string(cpu.get("Name")),
            "manufacturer": _clean_string(cpu.get("Manufacturer")),
            "physical_cores": _to_int(cpu.get("NumberOfCores")),
            "logical_cores": _to_int(cpu.get("NumberOfLogicalProcessors")),
            "max_clock_mhz": _to_int(cpu.get("MaxClockSpeed")),
        }
    )


def _collect_linux_cpu_specs() -> dict[str, Any]:
    cpuinfo = _read_text(Path("/proc/cpuinfo"))
    specs: dict[str, Any] = {}
    if cpuinfo:
        for line in cpuinfo.splitlines():
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if key == "model name" and "name" not in specs:
                specs["name"] = value

    max_freq_khz = _to_int(_read_text(Path("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")))
    if max_freq_khz is not None:
        specs["max_clock_mhz"] = int(round(max_freq_khz / 1000))

    return _drop_none_values(specs)


def _collect_darwin_cpu_specs() -> dict[str, Any]:
    name = _run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
    frequency_hz = _to_int(_run_command(["sysctl", "-n", "hw.cpufrequency_max"]))
    specs: dict[str, Any] = {"name": _clean_string(name)}
    if frequency_hz is not None:
        specs["max_clock_mhz"] = int(round(frequency_hz / 1_000_000))
    return _drop_none_values(specs)


def _collect_memory_specs() -> dict[str, Any]:
    system = platform.system()
    specs: dict[str, Any] = {}

    total_bytes = _get_total_memory_bytes(system)
    if total_bytes is not None:
        specs["total_bytes"] = total_bytes
        specs["total_gib"] = round(total_bytes / 1024 ** 3, 2)

    if system == "Windows":
        modules = _collect_windows_memory_modules()
        if modules:
            specs["modules"] = modules
            configured_speeds = [
                module["configured_clock_speed_mhz"]
                for module in modules
                if module.get("configured_clock_speed_mhz") is not None
            ]
            rated_speeds = [module["speed_mhz"] for module in modules if module.get("speed_mhz") is not None]
            if configured_speeds:
                specs["configured_clock_speed_mhz"] = max(configured_speeds)
            if rated_speeds:
                specs["rated_speed_mhz"] = max(rated_speeds)

    return specs


def _get_total_memory_bytes(system: str) -> int | None:
    if system == "Windows":
        return _get_windows_total_memory_bytes()
    if system == "Linux":
        return _get_linux_total_memory_bytes()
    if system == "Darwin":
        return _to_int(_run_command(["sysctl", "-n", "hw.memsize"]))
    return None


def _get_windows_total_memory_bytes() -> int | None:
    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullTotalPhys)


def _get_linux_total_memory_bytes() -> int | None:
    meminfo = _read_text(Path("/proc/meminfo"))
    if not meminfo:
        return None
    for line in meminfo.splitlines():
        key, _, value = line.partition(":")
        if key == "MemTotal":
            parts = value.strip().split()
            if len(parts) >= 2 and parts[1].lower() == "kb":
                kilobytes = _to_int(parts[0])
                return None if kilobytes is None else kilobytes * 1024
    return None


def _collect_windows_memory_modules() -> list[dict[str, Any]]:
    output = _run_command(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_PhysicalMemory | "
            "Select-Object Manufacturer,PartNumber,Capacity,Speed,ConfiguredClockSpeed | "
            "ConvertTo-Json -Compress",
        ]
    )
    data = _parse_json_object_or_list(output)
    modules: list[dict[str, Any]] = []
    for module in data:
        capacity_bytes = _to_int(module.get("Capacity"))
        entry = _drop_none_values(
            {
                "manufacturer": _clean_string(module.get("Manufacturer")),
                "part_number": _clean_string(module.get("PartNumber")),
                "capacity_bytes": capacity_bytes,
                "capacity_gib": None if capacity_bytes is None else round(capacity_bytes / 1024 ** 3, 2),
                "speed_mhz": _to_int(module.get("Speed")),
                "configured_clock_speed_mhz": _to_int(module.get("ConfiguredClockSpeed")),
            }
        )
        if entry:
            modules.append(entry)
    return modules


def _collect_gpu_specs() -> list[dict[str, Any]]:
    gpus = _collect_torch_gpu_specs()
    nvidia_gpus = _collect_nvidia_smi_gpu_specs()
    if nvidia_gpus:
        gpus = _merge_gpu_specs(gpus, nvidia_gpus)

    if not gpus and platform.system() == "Windows":
        gpus = _collect_windows_video_controller_specs()

    return gpus


def _collect_torch_gpu_specs() -> list[dict[str, Any]]:
    if torch is None or not torch.cuda.is_available():
        return []

    gpus: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        gpus.append(
            _drop_none_values(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "total_memory_gib": round(int(props.total_memory) / 1024 ** 3, 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multiprocessor_count": int(props.multi_processor_count),
                }
            )
        )
    return gpus


def _collect_nvidia_smi_gpu_specs() -> list[dict[str, Any]]:
    output = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version,clocks.max.memory,clocks.max.graphics",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return []

    gpus: list[dict[str, Any]] = []
    for line in output.splitlines():
        fields = [_clean_string(field) for field in line.split(",")]
        if len(fields) != 6:
            continue
        index, name, memory_total_mib, driver_version, memory_clock_mhz, graphics_clock_mhz = fields
        total_memory_mib = _to_int(memory_total_mib)
        gpus.append(
            _drop_none_values(
                {
                    "index": _to_int(index),
                    "name": name,
                    "total_memory_mib": total_memory_mib,
                    "total_memory_bytes": None if total_memory_mib is None else total_memory_mib * 1024 ** 2,
                    "total_memory_gib": None if total_memory_mib is None else round(total_memory_mib / 1024, 2),
                    "driver_version": driver_version,
                    "max_memory_clock_mhz": _to_int(memory_clock_mhz),
                    "max_graphics_clock_mhz": _to_int(graphics_clock_mhz),
                }
            )
        )
    return gpus


def _collect_windows_video_controller_specs() -> list[dict[str, Any]]:
    output = _run_command(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name,AdapterRAM,DriverVersion | "
            "ConvertTo-Json -Compress",
        ]
    )
    data = _parse_json_object_or_list(output)
    gpus: list[dict[str, Any]] = []
    for index, gpu in enumerate(data):
        adapter_ram = _to_int(gpu.get("AdapterRAM"))
        entry = _drop_none_values(
            {
                "index": index,
                "name": _clean_string(gpu.get("Name")),
                "total_memory_bytes": adapter_ram,
                "total_memory_gib": None if adapter_ram is None else round(adapter_ram / 1024 ** 3, 2),
                "driver_version": _clean_string(gpu.get("DriverVersion")),
            }
        )
        if entry:
            gpus.append(entry)
    return gpus


def _merge_gpu_specs(primary: list[dict[str, Any]], secondary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged_by_index = {gpu.get("index"): dict(gpu) for gpu in primary if gpu.get("index") is not None}
    unindexed = [dict(gpu) for gpu in primary if gpu.get("index") is None]

    for gpu in secondary:
        index = gpu.get("index")
        if index is None or index not in merged_by_index:
            unindexed.append(dict(gpu))
            continue
        merged_by_index[index].update({key: value for key, value in gpu.items() if value is not None})

    indexed = [merged_by_index[index] for index in sorted(merged_by_index)]
    return indexed + unindexed


def _collect_torch_specs() -> dict[str, Any]:
    if torch is None:
        return {"available": False}
    return _drop_none_values(
        {
            "available": True,
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }
    )


def _run_command(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _parse_json_object_or_list(text: str | None) -> list[dict[str, Any]]:
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _clean_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(str(value).strip()))
    except ValueError:
        return None


def _drop_none_values(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}
