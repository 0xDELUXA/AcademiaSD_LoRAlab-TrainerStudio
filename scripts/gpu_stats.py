# -*- coding: utf-8 -*-
"""
gpu_stats.py — GPU elegida para entrenar y lectura de la VRAM para el panel de los entrenadores.
GPU chosen for training and VRAM reading for the trainers' panel.

Elección de GPU: el lanzador guarda el índice de nvidia-smi en settings/ui.json ("gpu") y, al abrir
un entrenador, pone CUDA_DEVICE_ORDER=PCI_BUS_ID y CUDA_VISIBLE_DEVICES=<índice>. Los servidores y
los scripts heredan esas variables, así que siguen usando "cuda:0", que pasa a ser esa tarjeta.

VRAM, por orden: nvidia-smi (de la GPU elegida) -> all-smi (AMD en Windows: VRAM de toda la GPU con
los contadores de Windows, https://github.com/lablup/all-smi, Lablup, Apache-2.0) -> ficheros del
driver amdgpu en Linux -> torch como último recurso. torch.cuda.mem_get_info() en el servidor web
solo ve su propio proceso en ROCm para Windows y además le quita VRAM al entrenamiento.
"""
import json
import os
import shutil
import subprocess
import time
from glob import glob
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
UI_SETTINGS = BASE_DIR / "settings" / "ui.json"
ALL_SMI = [BASE_DIR / "tools" / "all-smi" / "all-smi.exe", BASE_DIR / "tools" / "all-smi" / "all-smi"]
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
_cache = {"time": 0.0, "value": None}


def _run(cmd, timeout=5):
    return subprocess.check_output(cmd, text=True, errors="ignore", timeout=timeout,
                                   creationflags=_NO_WINDOW, stderr=subprocess.DEVNULL)


# --------------------------------------------------------------------------------------------
# Elección de GPU / GPU choice
# --------------------------------------------------------------------------------------------

def list_nvidia():
    """[{"index": "0", "name": ..., "total_gb": ...}] en el orden de nvidia-smi (bus PCI); [] sin NVIDIA."""
    try:
        out = _run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,nounits,noheader"])
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            gpus.append({"index": parts[0], "name": parts[1], "total_gb": round(float(parts[2] or 0) / 1024, 1)})
    return gpus


def saved_gpu():
    try:
        value = json.loads(UI_SETTINGS.read_text(encoding="utf-8")).get("gpu", "")
    except (OSError, ValueError):
        return ""
    return str(value) if str(value).isdigit() else ""


def save_gpu(index):
    try:
        cfg = json.loads(UI_SETTINGS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cfg = {}
    cfg["gpu"] = str(index) if str(index).isdigit() else ""
    UI_SETTINGS.parent.mkdir(exist_ok=True)
    UI_SETTINGS.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def apply_saved_gpu(env):
    """Fija en env la GPU guardada, si existe y hay más de una. Si no, env no cambia (también respeta
    un CUDA_VISIBLE_DEVICES puesto a mano). Una tarjeta que ya no existe se ignora: se usa la primera."""
    chosen = saved_gpu()
    gpus = list_nvidia()
    if chosen and len(gpus) > 1 and any(g["index"] == chosen for g in gpus):
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = chosen


def _selected_index():
    """Índice de nvidia-smi de la GPU que usa este proceso, si se eligió una (o se puso a mano)."""
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        return None
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    return value if value.isdigit() else None


# --------------------------------------------------------------------------------------------
# VRAM para el panel / VRAM for the panel
# --------------------------------------------------------------------------------------------

def _info(name, used_b, total_b, temp=0):
    return {"gpu_name": name, "total_gb": round(total_b / 1024 ** 3, 2), "used_gb": round(used_b / 1024 ** 3, 2),
            "percent": round(used_b / total_b * 100, 1) if total_b > 0 else 0.0, "temp_c": int(temp or 0)}


def _from_nvidia_smi():
    cmd = ["nvidia-smi", "--query-gpu=memory.total,memory.used,name,temperature.gpu", "--format=csv,nounits,noheader"]
    index = _selected_index()
    if index is not None:
        cmd += ["-i", index]
    parts = [p.strip() for p in _run(cmd).strip().splitlines()[0].split(",")]
    mib = 1024 ** 2
    return _info(parts[2], float(parts[1]) * mib, float(parts[0]) * mib, float(parts[3]))


def _from_all_smi():
    exe = next((str(p) for p in ALL_SMI if p.is_file()), None) or shutil.which("all-smi")
    if not exe:
        return None
    data = json.loads(_run([exe, "snapshot", "--format", "json", "--pretty=false", "--include", "gpu"], timeout=10))
    gpus = [g for g in data.get("gpus") or [] if g.get("device_type", "GPU") == "GPU" and g.get("total_memory")]
    if not gpus:
        return None
    # Con gráfica integrada y dedicada, la dedicada es la de más VRAM.
    g = max(gpus, key=lambda x: x["total_memory"])
    temp = g.get("temperature") or 0
    return _info(g.get("name", "GPU"), g.get("used_memory", 0), g["total_memory"], temp if 0 < temp < 200 else 0)


def _from_amdgpu_sysfs():
    cards = []
    for total_f in glob("/sys/class/drm/card*/device/mem_info_vram_total"):
        dev = Path(total_f).parent
        try:
            cards.append((int(Path(total_f).read_text()), int((dev / "mem_info_vram_used").read_text()), dev))
        except (OSError, ValueError):
            continue
    if not cards:
        return None
    total, used, dev = max(cards, key=lambda c: c[0])
    temp = 0
    for f in glob(str(dev / "hwmon" / "hwmon*" / "temp1_input")):
        try:
            temp = int(Path(f).read_text()) / 1000
            break
        except (OSError, ValueError):
            pass
    try:
        name = (dev / "product_name").read_text().strip() or "AMD GPU"
    except OSError:
        name = "AMD GPU"
    return _info(name, used, total, temp)


def _from_torch():
    import torch
    if not torch.cuda.is_available():
        return None
    free_b, total_b = torch.cuda.mem_get_info(0)
    return _info(torch.cuda.get_device_name(0), total_b - free_b, total_b)


def vram_info():
    """{"gpu_name", "total_gb", "used_gb", "percent", "temp_c"} de la GPU de entrenamiento. Se guarda
    2 s: la GUI pregunta cada pocos segundos y all-smi es un programa aparte."""
    now = time.time()
    if _cache["value"] is not None and now - _cache["time"] < 2:
        return _cache["value"]
    value = None
    for reader in (_from_nvidia_smi, _from_all_smi, _from_amdgpu_sysfs, _from_torch):
        try:
            value = reader()
        except Exception:
            value = None
        if value:
            break
    _cache.update(time=now, value=value or {"gpu_name": "N/A", "total_gb": 0.0, "used_gb": 0.0,
                                            "percent": 0.0, "temp_c": 0})
    return _cache["value"]
