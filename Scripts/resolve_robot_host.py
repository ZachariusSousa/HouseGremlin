from __future__ import annotations

import argparse
import socket
import sys
import threading
from urllib.parse import urlparse, urlunparse

from zeroconf import IPVersion, ServiceBrowser, ServiceInfo, ServiceListener, Zeroconf


def with_host(url: str, host: str) -> str:
    parsed = urlparse(url)
    netloc = host if parsed.port is None else f"{host}:{parsed.port}"
    return urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)).rstrip("/")


def system_resolve(host: str) -> str | None:
    try:
        addresses = socket.getaddrinfo(host, None, socket.AF_INET, socket.SOCK_STREAM)
    except socket.gaierror:
        return None
    return addresses[0][4][0] if addresses else None


class RobotListener(ServiceListener):
    def __init__(self, target_host: str):
        self.target_host = target_host.rstrip(".").casefold()
        self.address: str | None = None
        self.found = threading.Event()

    def add_service(self, zeroconf: Zeroconf, service_type: str, name: str) -> None:
        self._consider(zeroconf.get_service_info(service_type, name, timeout=1500))

    def update_service(self, zeroconf: Zeroconf, service_type: str, name: str) -> None:
        self._consider(zeroconf.get_service_info(service_type, name, timeout=1500))

    def remove_service(self, zeroconf: Zeroconf, service_type: str, name: str) -> None:
        return

    def _consider(self, info: ServiceInfo | None) -> None:
        if info is None or info.server.rstrip(".").casefold() != self.target_host:
            return
        addresses = info.parsed_addresses(IPVersion.V4Only)
        if addresses:
            self.address = addresses[0]
            self.found.set()


def mdns_resolve(host: str, timeout_seconds: float) -> str | None:
    listener = RobotListener(host)
    zeroconf = Zeroconf(ip_version=IPVersion.V4Only)
    browser = ServiceBrowser(zeroconf, "_http._tcp.local.", listener)
    try:
        listener.found.wait(timeout_seconds)
        return listener.address
    finally:
        browser.cancel()
        zeroconf.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve Robit's .local name using DNS or mDNS service discovery.")
    parser.add_argument("robot_url")
    parser.add_argument("--timeout", type=float, default=8.0)
    args = parser.parse_args()
    parsed = urlparse(args.robot_url)
    host = parsed.hostname
    if not host:
        parser.error("robot_url must include a hostname")
    if not host.casefold().endswith(".local"):
        print(args.robot_url.rstrip("/"))
        return 0

    address = system_resolve(host) or mdns_resolve(host, args.timeout)
    if address is None:
        print(
            f"[resolve-robot][error] Could not discover {host}. Confirm Robit is powered on and on this network.",
            file=sys.stderr,
        )
        return 1
    print(with_host(args.robot_url, address))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
