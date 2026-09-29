from __future__ import annotations

import argparse
import os
import subprocess


DEFAULT_PORTS = (8080, 8081, 7861, 8091, 8093)


def pids_on_ports(ports: set[int]) -> set[int]:
    result = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, check=False)
    pids: set[int] = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0].upper() != "TCP":
            continue
        try:
            port = int(parts[1].rsplit(":", 1)[1])
            pid = int(parts[-1])
        except (IndexError, ValueError):
            continue
        if port in ports and pid != os.getpid():
            pids.add(pid)
    return pids


def main() -> int:
    parser = argparse.ArgumentParser(description="Stop stale Robit stack processes by listener port.")
    parser.add_argument("--ports", nargs="*", type=int, default=DEFAULT_PORTS)
    args = parser.parse_args()

    pids = sorted(pids_on_ports(set(args.ports)))
    if not pids:
        print("[stop] no Robit listener processes found")
        return 0

    print(f"[stop] stopping Robit process trees: {', '.join(str(pid) for pid in pids)}")
    for pid in pids:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
