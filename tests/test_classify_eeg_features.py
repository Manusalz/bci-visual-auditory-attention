from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.classify_eeg_features import run_classifier  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(123)
    n = 40
    y = np.r_[np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)]
    frame = pd.DataFrame(
        {
            "condition": np.where(y == 1, "target", "standard"),
            "f1": y + rng.normal(0, 0.3, size=n),
            "f2": rng.normal(0, 1, size=n),
        }
    )
    result = run_classifier(
        frame,
        feature_cols=["f1", "f2"],
        label_col="condition",
        positive_label="target",
        negative_label="standard",
        permutations=10,
        bootstrap=20,
        seed=7,
    )
    assert result["n_positive"] == 20
    assert result["n_negative"] == 20
    assert 0.0 <= result["balanced_accuracy"] <= 1.0
    assert 0.0 <= result["p_permutation"] <= 1.0
    print("classifier smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
