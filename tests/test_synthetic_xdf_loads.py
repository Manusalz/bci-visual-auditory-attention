from pathlib import Path

import pyxdf


def main() -> int:
    path = Path("data/synthetic/P01_synthetic_high_performance.xdf")
    assert path.exists(), f"missing synthetic XDF: {path}"
    streams, _header = pyxdf.load_xdf(str(path))
    names = {stream["info"]["name"][0]: stream for stream in streams}
    assert "openvibeSignal" in names
    assert "BCI_Markers" in names

    eeg = names["openvibeSignal"]
    markers = names["BCI_Markers"]
    assert int(eeg["info"]["channel_count"][0]) == 8
    assert float(eeg["info"]["nominal_srate"][0]) == 250.0
    assert len(eeg["time_stamps"]) > 1000
    assert len(markers["time_stamps"]) > 20

    marker_values = [row[0] for row in markers["time_series"]]
    assert any(value.startswith("visual/cue/") for value in marker_values)
    assert any(value == "visual/response/space" for value in marker_values)
    assert any(value.startswith("audio2/event/") for value in marker_values)
    assert any(value == "audio2/response/space" for value in marker_values)
    print("synthetic XDF loads ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
