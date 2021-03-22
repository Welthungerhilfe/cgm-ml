import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parents[1]))

from utils import (COLUMN_NAME_GOODBAD,  # noqa :E402;
                   calculate_percentage_confusion_matrix,
                   draw_uncertainty_goodbad_plot, get_dataset_path)


def test_draw_uncertainty_goodbad_plot():
    uncertainties = [1.010987, 1.073083, 1.312352, 3.515901, 1.602865]
    goodbad = [1.0, 0.0, 1.0, 0.0, 0.0]

    df = pd.DataFrame(list(zip(uncertainties, goodbad)), columns=['uncertainties', COLUMN_NAME_GOODBAD])
    with tempfile.NamedTemporaryFile() as tmp:
        draw_uncertainty_goodbad_plot(df, tmp.name)


def test_calculate_percentage_confusion_matrix():
    data = np.array([[2, 4, 0],
                     [2, 2, 1],
                     [1, 1, 2]])
    T, FP, FN = calculate_percentage_confusion_matrix(data)
    assert T == 40
    assert FP == 33.33
    assert FN == 26.67


def test_get_dataset_path():
    result = get_dataset_path(Path("/tmp/data/"), "test")
    assert result == '/tmp/data/test'
