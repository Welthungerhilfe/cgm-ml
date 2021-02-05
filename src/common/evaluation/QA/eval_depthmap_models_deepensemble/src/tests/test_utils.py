from pathlib import Path
import sys
import tempfile

import pandas as pd

sys.path.append(str(Path(__file__).parents[1]))

from utils import draw_uncertainty_goodbad_plot, COLUMN_NAME_GOODBAD  # noqa: E402


def test_draw_uncertainty_goodbad_plot():
    uncertainties = [1.010987, 1.073083, 1.312352, 3.515901, 1.602865]
    goodbad = [1.0, 0.0, 1.0, 0.0, 0.0]

    df = pd.DataFrame(list(zip(uncertainties, goodbad)), columns=['uncertainties', COLUMN_NAME_GOODBAD])
    with tempfile.NamedTemporaryFile() as tmp:
        draw_uncertainty_goodbad_plot(df, tmp.name)
