from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAGE_PATH = ROOT / "web_control" / "index.html"
HARNESS_PATH = Path(__file__).with_name("web_control_behavior.cjs")


@dataclass
class Element:
    tag: str
    attrs: dict[str, str]
    parent: "Element | None"
    children: list["Element"] = field(default_factory=list)
    text_parts: list[str] = field(default_factory=list)

    @property
    def classes(self) -> set[str]:
        return set(self.attrs.get("class", "").split())

    @property
    def text(self) -> str:
        return " ".join("".join(self.text_parts).split())

    def descendants(self):
        for child in self.children:
            yield child
            yield from child.descendants()


class TreeParser(HTMLParser):
    VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Element("document", {}, None)
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        element = Element(tag, dict(attrs), self.stack[-1])
        self.stack[-1].children.append(element)
        if tag not in self.VOID_TAGS:
            self.stack.append(element)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID_TAGS:
            self.stack.pop()

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data):
        for element in self.stack:
            element.text_parts.append(data)


def parsed_page() -> tuple[str, Element]:
    page = PAGE_PATH.read_text(encoding="utf-8")
    parser = TreeParser()
    parser.feed(page)
    return page, parser.root


def find_all(root: Element, predicate):
    return [element for element in root.descendants() if predicate(element)]


def find_id(root: Element, element_id: str) -> Element:
    matches = find_all(root, lambda element: element.attrs.get("id") == element_id)
    assert len(matches) == 1, f"expected exactly one #{element_id}, found {len(matches)}"
    return matches[0]


def is_descendant(element: Element, ancestor: Element) -> bool:
    current = element.parent
    while current is not None:
        if current is ancestor:
            return True
        current = current.parent
    return False


def test_dashboard_has_six_modes_and_overview_owns_the_only_camera_and_decorations():
    _page, root = parsed_page()
    panels = find_all(root, lambda element: element.tag == "main" and "mode-panel" in element.classes)
    assert [panel.attrs.get("data-panel") for panel in panels] == [
        "overview",
        "telemetry",
        "manual",
        "voice",
        "memory",
        "diagnostic",
    ]
    f2 = find_all(root, lambda element: element.tag == "button" and element.attrs.get("data-key") == "F2")
    assert len(f2) == 1
    assert f2[0].text == "TELEMETRY"

    overview = next(panel for panel in panels if panel.attrs["data-panel"] == "overview")
    camera = find_id(root, "cameraStream")
    assert is_descendant(camera, overview)
    assert len(find_all(root, lambda element: element.attrs.get("id") == "cameraStream")) == 1
    for decoration_class in ("concept-stamp", "registration", "patch-cables"):
        decorations = find_all(root, lambda element: decoration_class in element.classes)
        assert len(decorations) == 1
        assert is_descendant(decorations[0], overview)


def test_dashboard_removes_seeded_status_events_and_static_network_artifacts():
    page, root = parsed_page()
    banned = (
        "192.168.04.01",
        "localhost:8080",
        "08 ms / secure",
        "14:08:22",
        "cycle 245",
    )
    for value in banned:
        assert value not in page.lower()
    assert not re.search(r">\s*73\s*<|width\s*:\s*73%", page, re.IGNORECASE)

    transcript = find_id(root, "transcript")
    assert not find_all(transcript, lambda element: "msg" in element.classes)
    events = find_id(root, "events")
    assert not find_all(events, lambda element: element.tag == "p")

    network_chart = find_id(root, "networkChart")
    assert "HOST NETWORK THROUGHPUT" in network_chart.text
    assert "BYTES / SECOND" in network_chart.text
    assert "RX" in network_chart.text and "TX" in network_chart.text
    assert not find_all(network_chart, lambda element: element.tag == "polyline" and bool(element.attrs.get("points")))
    assert find_id(root, "networkRxPath").tag == "path"
    assert find_id(root, "networkTxPath").tag == "path"

    rtt_chart = find_id(root, "robotRttChart")
    assert "ESP CONTROL ROUND-TRIP" in rtt_chart.text
    assert "MILLISECONDS" in rtt_chart.text
    assert find_id(root, "robotRttPath").tag == "path"


def test_telemetry_panel_has_separate_pc_llm_and_esp_cards_with_explicit_initial_states():
    _page, root = parsed_page()
    telemetry = next(
        element
        for element in root.descendants()
        if element.tag == "main" and element.attrs.get("data-panel") == "telemetry"
    )
    assert "PC / LLM TELEMETRY" in telemetry.text
    assert "ESP TELEMETRY" in telemetry.text
    for element_id in (
        "hostStatus",
        "hostCpu",
        "hostMemory",
        "hostProcess",
        "hostLoopLag",
        "gpuStatus",
        "gpuVramValue",
        "llmStatus",
        "llmProbe",
        "llmInference",
        "robotStatus",
        "robotSampleAge",
        "robotMovement",
        "robotHeadActual",
        "robotHeadTarget",
        "robotEyes",
        "robotFirmware",
        "robotNetwork",
        "robotMemory",
        "robotWatchdog",
        "robotControl",
        "trackingTelemetryStatus",
        "telemetryFaults",
    ):
        value = find_id(root, element_id).text.upper()
        assert any(state in value for state in ("LOADING", "UNKNOWN", "N/A", "OFFLINE")), element_id


def test_inline_javascript_is_syntactically_valid(tmp_path):
    page = PAGE_PATH.read_text(encoding="utf-8")
    scripts = re.findall(r"<script(?:\s[^>]*)?>(.*?)</script>", page, re.DOTALL | re.IGNORECASE)
    assert scripts
    script_path = tmp_path / "web-control-inline.js"
    script_path.write_text("\n".join(scripts), encoding="utf-8")

    command = subprocess.list2cmdline(["node", "--check", str(script_path)])
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        shell=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_web_control_runtime_contracts_in_dependency_free_dom_harness():
    command = subprocess.list2cmdline(["node", str(HARNESS_PATH), str(PAGE_PATH)])
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        shell=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "web control behavior: ok" in result.stdout
