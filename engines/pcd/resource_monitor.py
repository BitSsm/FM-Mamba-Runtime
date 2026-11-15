"""Process CPU/RAM plus whole-device NVML telemetry, sampled independently."""
import csv
import threading
import time
from pathlib import Path

import psutil
import pynvml as nvml


class ResourceMonitor:
    def __init__(self, path, interval=0.2, device_index=0, include_children=False):
        self.path = Path(path)
        self.interval = interval
        self.process = psutil.Process()
        self.process.cpu_percent(None)
        self.include_children = include_children
        self.children = {}
        self.stop_event = threading.Event()
        self.wake_event = threading.Event()
        self.lock = threading.Lock()
        self.phase, self.step = "setup", 0
        self.start = time.perf_counter()
        self.phase_start = self.start
        self.rows, self.segments, self.errors = [], [], set()
        nvml.nvmlInit()
        self.handle = nvml.nvmlDeviceGetHandleByIndex(device_index)
        self.uuid = nvml.nvmlDeviceGetUUID(self.handle)
        self.thread = threading.Thread(target=self._run, name="experiment-telemetry", daemon=True)

    def set_phase(self, phase, step=0):
        with self.lock:
            now = time.perf_counter()
            changed = phase != self.phase
            if changed:
                self.segments.append((self.phase, self.phase_start-self.start, now-self.start))
                self.phase_start = now
            self.phase, self.step = phase, step
        if changed:
            # Short phases can fall entirely between periodic samples.
            self.wake_event.set()

    def safe(self, name, fn):
        try:
            return fn()
        except nvml.NVMLError as exc:
            self.errors.add(f"{name}: {exc}")
            return None

    def sample(self):
        with self.lock:
            phase, step = self.phase, self.step
        util = self.safe("utilization", lambda: nvml.nvmlDeviceGetUtilizationRates(self.handle))
        memory = self.safe("memory", lambda: nvml.nvmlDeviceGetMemoryInfo(self.handle))
        result = {"unix_s": time.time(), "elapsed_s": time.perf_counter()-self.start, "phase": phase, "step": step,
            "process_cpu_percent": self.process.cpu_percent(None),
            "process_rss_mib": self.process.memory_info().rss / 2**20,
            "system_ram_used_mib": psutil.virtual_memory().used / 2**20,
            "gpu_util_percent": util.gpu if util else None,
            "gpu_memory_controller_percent": util.memory if util else None,
            "device_memory_used_mib": memory.used / 2**20 if memory else None,
            "device_power_w": self.safe("power", lambda: nvml.nvmlDeviceGetPowerUsage(self.handle)/1000),
            "gpu_temperature_c": self.safe("temperature", lambda: nvml.nvmlDeviceGetTemperature(self.handle, nvml.NVML_TEMPERATURE_GPU)),
            "gpu_graphics_clock_mhz": self.safe("graphics_clock", lambda: nvml.nvmlDeviceGetClockInfo(self.handle, nvml.NVML_CLOCK_GRAPHICS)),
            "gpu_memory_clock_mhz": self.safe("memory_clock", lambda: nvml.nvmlDeviceGetClockInfo(self.handle, nvml.NVML_CLOCK_MEM))}
        if self.include_children:
            cpu, rss, alive = 0., 0., {}
            for process in self.process.children(recursive=True):
                try:
                    cached = self.children.get(process.pid, process)
                    cpu += cached.cpu_percent(None)
                    rss += cached.memory_info().rss / 2**20
                    alive[process.pid] = cached
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self.children = alive
            result.update(process_tree_cpu_percent=result["process_cpu_percent"]+cpu,
                          process_tree_rss_mib=result["process_rss_mib"]+rss,
                          child_process_count=len(alive))
        return result

    def _run(self):
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = None
            while not self.stop_event.is_set():
                try:
                    row = self.sample()
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    stream.flush()
                    self.rows.append(row)
                except Exception as exc:
                    self.errors.add(repr(exc))
                self.wake_event.wait(self.interval)
                self.wake_event.clear()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.set_phase("finished")
        self.stop_event.set()
        self.wake_event.set()
        self.thread.join(timeout=5)
        nvml.nvmlShutdown()

    def summary(self, phase="train"):
        segments = [(a, b) for name, a, b in self.segments if name == phase]
        selected = [r for r in self.rows if r["phase"] == phase]
        result = {"phase": phase, "phase_wall_s": sum(b-a for a, b in segments), "samples": len(selected),
            "nvml_uuid": self.uuid, "sampling_interval_s": self.interval, "errors": sorted(self.errors),
            "sampling_policy": "periodic plus immediate wake on phase changes",
            "power_scope": "whole GPU, includes desktop/other processes; linearly interpolated sampled estimate",
            "process_cpu_percent_scope": "100% equals one logical CPU; can exceed 100%"}
        keys = ["device_power_w", "gpu_util_percent", "gpu_temperature_c", "process_rss_mib", "device_memory_used_mib",
                "process_cpu_percent", "gpu_graphics_clock_mhz"]
        if self.include_children:
            keys += ["process_tree_cpu_percent", "process_tree_rss_mib", "child_process_count"]
            result["process_tree_rss_scope"] = "sum of trainer and loader child RSS; shared pages may be counted more than once"
        for key in keys:
            area = covered = 0.
            for row0, row1 in zip(self.rows, self.rows[1:]):
                t0, t1 = row0["elapsed_s"], row1["elapsed_s"]
                v0, v1 = row0[key], row1[key]
                if v0 is None or v1 is None or t1 <= t0:
                    continue
                for start, end in segments:
                    lo, hi = max(t0, start), min(t1, end)
                    if hi > lo:
                        left = v0+(v1-v0)*(lo-t0)/(t1-t0)
                        right = v0+(v1-v0)*(hi-t0)/(t1-t0)
                        area += (left+right)*0.5*(hi-lo)
                        covered += hi-lo
            values = [r[key] for r in selected if r[key] is not None]
            result[key+"_mean"] = area/covered if covered else None
            result[key+"_max"] = max(values) if values else None
            if key == "device_power_w":
                result["estimated_device_energy_wh"] = area/3600 if covered else None
                result["energy_covered_s"] = covered
        return result
