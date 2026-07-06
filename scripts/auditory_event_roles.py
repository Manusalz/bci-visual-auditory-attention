"""Auditory event taxonomy used by the reproducible pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioBlock:
    module: str
    instruction: str
    block: int | None = None


@dataclass(frozen=True)
class AudioEvent:
    module: str
    side: str
    subtype: str
    event_class: str


def parse_block_start(marker: str) -> AudioBlock | None:
    parts = marker.split("/")
    if len(parts) < 3 or parts[1] != "block_start":
        return None
    module = parts[0]
    if module not in {"audio2", "audio4"}:
        return None
    instruction = parts[2]
    block = None
    if len(parts) >= 4:
        try:
            block = int(parts[3].removeprefix("b"))
        except ValueError:
            block = None
    return AudioBlock(module=module, instruction=instruction, block=block)


def parse_audio_event(marker: str) -> AudioEvent | None:
    parts = marker.split("/")
    if len(parts) < 4 or parts[1] != "event":
        return None
    module = parts[0]
    if module not in {"audio2", "audio4"}:
        return None
    side = parts[2]
    subtype = parts[3]
    return AudioEvent(module=module, side=side, subtype=subtype, event_class=f"{side}/{subtype}")


def attended_event_class(module: str, instruction: str) -> str:
    if module == "audio2":
        if instruction not in {"attL", "attR"}:
            raise ValueError(f"Unknown audio2 instruction: {instruction}")
        side = "L" if instruction == "attL" else "R"
        return f"{side}/tgt"
    if module == "audio4":
        side, tone = instruction.split("_", 1)
        side_code = {"left": "L", "right": "R"}[side]
        if tone not in {"low", "high"}:
            raise ValueError(f"Unknown audio4 tone: {tone}")
        return f"{side_code}/tgt_{tone}"
    raise ValueError(f"Unknown module: {module}")


def classify_audio_event(event: AudioEvent, block: AudioBlock) -> str:
    """Return standard, target_atendido, or target_ignorado.

    `standard` is only the common/non-target sound. It is not a catch-all
    category for every non-attended event.
    """

    if event.module != block.module:
        raise ValueError("Event and block belong to different modules.")
    if event.subtype == "std":
        return "standard"
    intended = attended_event_class(block.module, block.instruction)
    if event.event_class == intended:
        return "target_atendido"
    return "target_ignorado"


def condition_label(role: str, responded: bool | None = None) -> str:
    if role == "target_atendido":
        if responded is True:
            return "target_atendido_hit"
        if responded is False:
            return "target_atendido_miss"
    return role
