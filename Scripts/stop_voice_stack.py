from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time


def pids_on_ports(ports: set[int]) -> set[int]:
    result = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, check=False)
    pids: set[int] = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0].upper() != "TCP":
            continue
        local_address = parts[1]
        try:
            port = int(local_address.rsplit(":", 1)[1])
        except (IndexError, ValueError):
            continue
        if port in ports:
            try:
                pids.add(int(parts[-1]))
            except ValueError:
                pass
    pids.discard(0)
    pids.discard(os.getpid())
    return pids


def main() -> int:
    parser = argparse.ArgumentParser(description="Stop stale Robit realtime voice processes by port.")
    parser.add_argument("--ports", nargs="*", type=int, default=[7861, 8081])
    args = parser.parse_args()

    pids = pids_on_ports(set(args.ports))
    if not pids:
        print("[voice-cleanup] no stale voice listeners found")
        return 0

    print(f"[voice-cleanup] stopping stale voice listener pids: {', '.join(str(pid) for pid in sorted(pids))}")
    for pid in sorted(pids):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    time.sleep(1.0)
    for pid in sorted(pids_on_ports(set(args.ports))):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
