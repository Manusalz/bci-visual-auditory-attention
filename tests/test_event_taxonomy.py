from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.auditory_event_roles import (  # noqa: E402
    AudioBlock,
    AudioEvent,
    attended_event_class,
    classify_audio_event,
    condition_label,
)


def main() -> int:
    block = AudioBlock(module="audio2", instruction="attL")
    assert attended_event_class("audio2", "attL") == "L/tgt"
    assert classify_audio_event(AudioEvent("audio2", "L", "std", "L/std"), block) == "standard"
    assert classify_audio_event(AudioEvent("audio2", "R", "std", "R/std"), block) == "standard"
    assert classify_audio_event(AudioEvent("audio2", "L", "tgt", "L/tgt"), block) == "target_atendido"
    assert classify_audio_event(AudioEvent("audio2", "R", "tgt", "R/tgt"), block) == "target_ignorado"
    assert condition_label("target_atendido", True) == "target_atendido_hit"
    assert condition_label("target_atendido", False) == "target_atendido_miss"

    block4 = AudioBlock(module="audio4", instruction="left_low")
    assert attended_event_class("audio4", "left_low") == "L/tgt_low"
    assert classify_audio_event(AudioEvent("audio4", "L", "tgt_low", "L/tgt_low"), block4) == "target_atendido"
    assert classify_audio_event(AudioEvent("audio4", "L", "tgt_high", "L/tgt_high"), block4) == "target_ignorado"
    assert classify_audio_event(AudioEvent("audio4", "R", "tgt_low", "R/tgt_low"), block4) == "target_ignorado"
    print("event taxonomy ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
