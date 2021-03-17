from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[2]))  # common

from evaluation.uncertainty_utils import change_dropout_strength  # noqa: E402
from model_utils.utils import create_base_cnn  # noqa: E402


def test_get_prediction_uncertainty():
    input_shape = (240, 180, 1)
    model = create_base_cnn(input_shape, dropout=True)
    new_model = change_dropout_strength(model, 2)
    assert model.layers[3].rate == 0.05
    assert new_model.layers[3].rate == 0.05 * 2
