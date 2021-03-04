import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parents[1]))

from utils import COLUMN_NAME_GOODBAD  # noqa: E402
from utils import (calculate_percentage_confusion_matrix,
                   draw_uncertainty_goodbad_plot)


def test_draw_uncertainty_goodbad_plot():
    uncertainties = [1.010987, 1.073083, 1.312352, 3.515901, 1.602865]
    goodbad = [1.0, 0.0, 1.0, 0.0, 0.0]

    df = pd.DataFrame(list(zip(uncertainties, goodbad)), columns=['uncertainties', COLUMN_NAME_GOODBAD])
    with tempfile.NamedTemporaryFile() as tmp:
        draw_uncertainty_goodbad_plot(df, tmp.name)

def test_calculate_percentage_confusion_matrix():
    T,FP,FN=calculate_percentage_confusion_matrix(data)
    assert T==
    assert FP==
    assert FN==


