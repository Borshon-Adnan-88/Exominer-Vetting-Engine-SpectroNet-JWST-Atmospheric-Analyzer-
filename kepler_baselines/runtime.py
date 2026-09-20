"""Process RAM measurement without another dependency."""
import os
import threading


def process_memory():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counters = Counters(); counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)
    import resource
    import sys
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    if os.path.exists("/proc/self/statm"):
        with open("/proc/self/statm") as stream:
            rss = int(stream.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    else:
        rss = peak
    return rss, peak


class MemoryMonitor:
    def __init__(self):
        self.phase = "input_validation"
        self.peaks = {}
        self.error = None
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stopped.is_set():
            try:
                rss, _ = process_memory()
                self.peaks[self.phase] = max(rss, self.peaks.get(self.phase, 0))
            except Exception as exc:
                self.error = type(exc).__name__
                return
            self.stopped.wait(.05)

    def start(self):
        process_memory()  # Fail visibly if measurement is unavailable.
        self.thread.start()

    def stop(self):
        self.stopped.set(); self.thread.join()
        if self.error:
            raise RuntimeError(f"RAM measurement failed: {self.error}")
        return {"process_peak_rss_bytes": process_memory()[1], "sampled_phase_peak_rss_bytes": self.peaks,
                "sampling_interval_seconds": .05,
                "scope": "Process working set/RSS including interpreter and feature matrix; family peaks are sampled, not incremental allocations."}
