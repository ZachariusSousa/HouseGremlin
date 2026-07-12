from __future__ import annotations

from pathlib import Path


FIELD = '''    responses_api_request_timeout_s: float = field(
        default=180.0,
        metadata={"help": "Read timeout in seconds for OpenAI-compatible Responses API requests."},
    )
'''


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    target = (
        root
        / "pc_brain"
        / ".venv"
        / "Lib"
        / "site-packages"
        / "speech_to_speech"
        / "arguments_classes"
        / "responses_api_language_model_arguments.py"
    )
    if not target.exists():
        print(f"[patch-s2s][warn] not found: {target}")
        return 0

    text = target.read_text(encoding="utf-8")
    if "responses_api_request_timeout_s" in text:
        print("[patch-s2s] responses_api_request_timeout_s already available")
        return 0

    marker = '''    responses_api_disable_thinking: bool = field(
        default=True,
        metadata={
            "help": "Disable provider-side thinking/reasoning when supported by the OpenAI-compatible backend. "
            "For Together Qwen3.5 models this sends chat_template_kwargs.enable_thinking=false."
        },
    )
'''
    if marker not in text:
        print("[patch-s2s][error] could not find Responses API timeout insertion point")
        return 1

    target.write_text(text.replace(marker, marker + FIELD), encoding="utf-8")
    print("[patch-s2s] added responses_api_request_timeout_s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
