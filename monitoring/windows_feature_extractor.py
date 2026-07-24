#!/usr/bin/env python3
"""
EarlyStrike - Windows Real Feature Extractor
Pulls ACTUAL system data from Windows for ML model inference.

Features collected (must match your 15-feature training schema):
  0  cpu_usage          - system-wide CPU %
  1  entropy            - estimated from write-heavy processes (heuristic)
  2  path_length        - avg path length of recently written files
  3  file_deletion      - 1 if high file deletion rate detected
  4  powershell         - 1 if powershell/cmd process is running with high CPU
  5  suspicious_write   - 1 if bulk write operations detected
  6  high_cpu           - 1 if cpu > 80%
  7  high_entropy       - 1 if estimated entropy > threshold
  8  encrypted_file     - 1 if known encrypted extensions found in recent writes
  9  system_user        - 1 if suspicious process runs as SYSTEM
 10  unknown_process    - 1 if unknown/unsigned process detected
 11  permission_change  - 1 if ACL/attribute changes detected (heuristic)
 12  temp_directory     - 1 if writes from TEMP/AppData
 13  moderate_cpu       - 1 if 40 < cpu <= 80
 14  moderate_entropy   - 1 if estimated entropy between 4 and 7
"""

import os
import time
import math
import threading
import psutil
import numpy as np
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #

RANSOMWARE_EXTENSIONS = {
    '.encrypted', '.locked', '.crypted', '.crypto', '.rsa', '.enc',
    '.crypt', '.locky', '.zepto', '.odin', '.aesir', '.zzzzz',
    '.thor', '.micro', '.cerber', '.cerber2', '.cerber3',
}

SUSPICIOUS_PROCESS_NAMES = {
    'powershell', 'cmd', 'wscript', 'cscript', 'mshta',
    'regsvr32', 'rundll32', 'certutil', 'bitsadmin',
}

KNOWN_SAFE_PROCESSES = {
    'explorer', 'svchost', 'lsass', 'winlogon', 'csrss',
    'services', 'spoolsv', 'taskhostw', 'dwm', 'sihost',
    'RuntimeBroker', 'SearchIndexer', 'WmiPrvSE',
}

TEMP_PATH_KEYWORDS = ['\\temp\\', '\\tmp\\', '\\appdata\\local\\temp',
                      '\\appdata\\roaming\\', '%temp%']

# Directories to watch for file events (common ransomware targets)
WATCH_DIRS = [
    os.path.expanduser('~/Documents'),
    os.path.expanduser('~/Desktop'),
    os.path.expanduser('~/Pictures'),
    os.path.expanduser('~/Downloads'),
]


# --------------------------------------------------------------------------- #
#  File System Event Watcher (uses watchdog if available, fallback otherwise)
# --------------------------------------------------------------------------- #

class FileEventTracker:
    """
    Tracks file system events in high-value directories.
    Uses watchdog library if installed, otherwise polls via os.stat.
    """

    def __init__(self, watch_dirs=None, history_seconds=30):
        self.watch_dirs = [d for d in (watch_dirs or WATCH_DIRS) if os.path.exists(d)]
        self.history_seconds = history_seconds

        # Rolling event log: list of dicts with timestamp, path, event_type
        self._events = deque()
        self._lock = threading.Lock()
        self._observer = None
        self._running = False
        self._poll_thread = None

        self._start()

    # ------------------------------------------------------------------
    def _start(self):
        """Try watchdog first, fall back to polling."""
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            tracker = self  # capture for inner class

            class _Handler(FileSystemEventHandler):
                def on_any_event(self, event):
                    if not event.is_directory:
                        tracker._record(event.src_path, event.event_type)

            self._observer = Observer()
            for d in self.watch_dirs:
                self._observer.schedule(_Handler(), d, recursive=True)
            self._observer.start()
            self._running = True

        except ImportError:
            # Watchdog not installed — use lightweight polling fallback
            self._start_polling()

    def _start_polling(self):
        """Snapshot mtime of files every 2 s; emit 'modified' on change."""
        self._snapshot = {}
        self._running = True

        def _poll():
            while self._running:
                for d in self.watch_dirs:
                    try:
                        for root, _, files in os.walk(d):
                            for f in files:
                                fp = os.path.join(root, f)
                                try:
                                    mtime = os.path.getmtime(fp)
                                    prev = self._snapshot.get(fp)
                                    if prev is None:
                                        self._snapshot[fp] = mtime
                                    elif mtime != prev:
                                        self._record(fp, 'modified')
                                        self._snapshot[fp] = mtime
                                except OSError:
                                    pass
                    except PermissionError:
                        pass
                time.sleep(2)

        self._poll_thread = threading.Thread(target=_poll, daemon=True)
        self._poll_thread.start()

    # ------------------------------------------------------------------
    def _record(self, path: str, event_type: str):
        now = time.time()
        with self._lock:
            self._events.append({'ts': now, 'path': path, 'type': event_type})

    def _prune(self):
        cutoff = time.time() - self.history_seconds
        with self._lock:
            while self._events and self._events[0]['ts'] < cutoff:
                self._events.popleft()

    def recent_events(self):
        self._prune()
        with self._lock:
            return list(self._events)

    def stop(self):
        self._running = False
        if self._observer:
            self._observer.stop()
            self._observer.join()


# --------------------------------------------------------------------------- #
#  Entropy Estimator
# --------------------------------------------------------------------------- #

def estimate_write_entropy(processes) -> float:
    """
    Heuristic entropy score based on process write I/O patterns.
    Real disk entropy requires reading file bytes — impractical in real time.
    We approximate: processes with high write bytes AND high CPU get higher scores.
    Returns a value roughly in [1.0, 9.0].
    """
    total_write = 0.0
    high_write_cpu = 0.0

    for p in processes:
        try:
            io = p.io_counters()
            cpu = p.cpu_percent(interval=None)
            w = io.write_bytes / (1024 * 1024)  # MB
            total_write += w
            if w > 10 and cpu > 20:
                high_write_cpu += cpu
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            pass

    # Normalise to [1, 9]
    base = 2.0
    if total_write > 500:
        base = 7.5
    elif total_write > 200:
        base = 6.0
    elif total_write > 50:
        base = 4.5
    elif total_write > 10:
        base = 3.0

    jitter = min(high_write_cpu / 200.0, 1.5)
    return round(min(base + jitter, 9.0), 3)


# --------------------------------------------------------------------------- #
#  Main Feature Extractor
# --------------------------------------------------------------------------- #

class WindowsFeatureExtractor:
    """
    Extracts the 15 real-time features from Windows that match the training schema.
    Call .extract() to get a numpy array of shape (15,).
    """

    def __init__(self):
        self.file_tracker = FileEventTracker()
        # Snapshot for delta-based disk I/O
        self._prev_disk = psutil.disk_io_counters()
        self._prev_net  = psutil.net_io_counters()
        self._prev_ts   = time.time()
        # Process list cache (refresh every 5 s)
        self._proc_cache = []
        self._proc_cache_ts = 0.0

    # ------------------------------------------------------------------
    def _get_processes(self):
        now = time.time()
        if now - self._proc_cache_ts > 5:
            procs = []
            for p in psutil.process_iter(
                ['pid', 'name', 'cpu_percent', 'memory_percent',
                 'username', 'status', 'cmdline']
            ):
                try:
                    procs.append(p)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self._proc_cache = procs
            self._proc_cache_ts = now
        return self._proc_cache

    # ------------------------------------------------------------------
    def _delta_disk(self):
        """Returns (read_MB/s, write_MB/s) since last call."""
        now = time.time()
        curr = psutil.disk_io_counters()
        dt = max(now - self._prev_ts, 0.1)
        r = (curr.read_bytes  - self._prev_disk.read_bytes)  / (1024*1024*dt)
        w = (curr.write_bytes - self._prev_disk.write_bytes) / (1024*1024*dt)
        self._prev_disk = curr
        self._prev_ts   = now
        return max(r, 0), max(w, 0)

    # ------------------------------------------------------------------
    def extract(self) -> np.ndarray:
        """Return feature vector of shape (15,) matching training schema."""

        processes  = self._get_processes()
        cpu_usage  = psutil.cpu_percent(interval=0.2)
        mem        = psutil.virtual_memory()
        _, write_mb = self._delta_disk()
        events     = self.file_tracker.recent_events()

        # ---- Feature 0: cpu_usage ----------------------------------------
        f_cpu_usage = cpu_usage

        # ---- Feature 1: entropy (heuristic) --------------------------------
        f_entropy = estimate_write_entropy(processes)

        # ---- Feature 2: path_length -----------------------------------------
        if events:
            f_path_length = float(np.mean([len(e['path']) for e in events]))
        else:
            f_path_length = 20.0

        # ---- Feature 3: file_deletion ----------------------------------------
        del_count = sum(1 for e in events if e['type'] in ('deleted', 'moved'))
        f_file_deletion = 1 if del_count > 5 else 0

        # ---- Feature 4: powershell -------------------------------------------
        ps_high = any(
            p.info['name'] and
            any(s in p.info['name'].lower() for s in SUSPICIOUS_PROCESS_NAMES) and
            p.info['cpu_percent'] > 10
            for p in processes
        )
        f_powershell = 1 if ps_high else 0

        # ---- Feature 5: suspicious_write ------------------------------------
        # High write rate OR many modified events in short window
        mod_count = sum(1 for e in events if e['type'] in ('modified', 'created'))
        f_suspicious_write = 1 if (write_mb > 20 or mod_count > 15) else 0

        # ---- Feature 6: high_cpu --------------------------------------------
        f_high_cpu = 1 if cpu_usage > 80 else 0

        # ---- Feature 7: high_entropy ----------------------------------------
        f_high_entropy = 1 if f_entropy > 7.0 else 0

        # ---- Feature 8: encrypted_file --------------------------------------
        enc_hits = sum(
            1 for e in events
            if any(e['path'].lower().endswith(ext) for ext in RANSOMWARE_EXTENSIONS)
        )
        f_encrypted_file = 1 if enc_hits > 0 else 0

        # ---- Feature 9: system_user -----------------------------------------
        system_high = any(
            p.info.get('username', '') and
            'system' in str(p.info.get('username', '')).lower() and
            p.info['cpu_percent'] > 30
            for p in processes
        )
        f_system_user = 1 if system_high else 0

        # ---- Feature 10: unknown_process ------------------------------------
        unknown = [
            p for p in processes
            if p.info['name'] and
            p.info['name'].lower().replace('.exe', '') not in KNOWN_SAFE_PROCESSES and
            p.info['cpu_percent'] > 25
        ]
        f_unknown_process = 1 if len(unknown) > 3 else 0

        # ---- Feature 11: permission_change ----------------------------------
        # Heuristic: many 'moved' or 'created' events in system dirs
        sys_events = sum(
            1 for e in events
            if 'system32' in e['path'].lower() or 'windows' in e['path'].lower()
        )
        f_permission_change = 1 if sys_events > 2 else 0

        # ---- Feature 12: temp_directory ------------------------------------
        temp_events = sum(
            1 for e in events
            if any(kw in e['path'].lower() for kw in TEMP_PATH_KEYWORDS)
        )
        f_temp_directory = 1 if temp_events > 3 else 0

        # ---- Feature 13: moderate_cpu --------------------------------------
        f_moderate_cpu = 1 if 40 < cpu_usage <= 80 else 0

        # ---- Feature 14: moderate_entropy ----------------------------------
        f_moderate_entropy = 1 if 4.0 <= f_entropy <= 7.0 else 0

        features = np.array([
            f_cpu_usage,        # 0
            f_entropy,          # 1
            f_path_length,      # 2
            f_file_deletion,    # 3
            f_powershell,       # 4
            f_suspicious_write, # 5
            f_high_cpu,         # 6
            f_high_entropy,     # 7
            f_encrypted_file,   # 8
            f_system_user,      # 9
            f_unknown_process,  # 10
            f_permission_change,# 11
            f_temp_directory,   # 12
            f_moderate_cpu,     # 13
            f_moderate_entropy, # 14
        ], dtype=np.float32)

        return features

    # ------------------------------------------------------------------
    def describe(self, features: np.ndarray) -> dict:
        """Return human-readable dict for logging."""
        names = [
            'cpu_usage', 'entropy', 'path_length', 'file_deletion',
            'powershell', 'suspicious_write', 'high_cpu', 'high_entropy',
            'encrypted_file', 'system_user', 'unknown_process',
            'permission_change', 'temp_directory', 'moderate_cpu', 'moderate_entropy'
        ]
        return dict(zip(names, features.tolist()))

    def stop(self):
        self.file_tracker.stop()


# --------------------------------------------------------------------------- #
#  Quick self-test
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    print('EarlyStrike Windows Feature Extractor — self-test')
    extractor = WindowsFeatureExtractor()
    time.sleep(3)  # Let file watcher warm up

    for i in range(5):
        feats = extractor.extract()
        desc  = extractor.describe(feats)
        print(f'\n[Sample {i+1}]')
        for k, v in desc.items():
            print(f'  {k:<22} = {v}')
        time.sleep(2)

    extractor.stop()
    print('\nDone.')
