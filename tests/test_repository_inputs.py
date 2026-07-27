import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config_utils import load_config


def test_example_config_paths_exist():
    cfg = load_config(str(ROOT / "examples" / "parameters_example.txt"))
    assert Path(cfg["floris"]["input_file"]).exists()
    assert Path(cfg["case"]["wind_rose_file"]).exists()
    assert Path(cfg["case"]["ti_file"]).exists()


def test_example_arrays_are_compatible():
    for idx in (1, 2, 3):
        wind = np.load(ROOT / "examples" / "wind_conditions" / f"windRose_{idx}.npy")
        ti = np.load(ROOT / "examples" / "ti_matrices" / f"tiMatrix_{idx}.npy")
        assert wind.shape == ti.shape
        assert np.all(np.isfinite(wind))
        assert np.all(wind >= 0)
        assert np.isclose(wind.sum(), 1.0, atol=1e-6)
