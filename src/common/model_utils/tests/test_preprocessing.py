from pathlib import Path
import pytest
import sys

sys.path.append(str(Path(__file__).parents[2]))  # common/ dir

from model_utils.preprocessing import sample_systematic_from_artifacts, sample_windows_from_artifacts, REGEX_PICKLE  # noqa: E402


def test_sample_windows_from_artifacts_multiple_results():
    artifacts = ['001.p', '002.p', '003.p', '004.p', '005.p', '006.p']
    actual = list(sample_windows_from_artifacts(artifacts, 5))
    expected = [
        ['001.p', '002.p', '003.p', '004.p', '005.p'],
        ['002.p', '003.p', '004.p', '005.p', '006.p'],
    ]
    assert actual == expected


def test_sample_windows_from_artifacts_one_result():
    artifacts = ['001.p', '002.p', '003.p', '004.p', '005.p']
    actual = list(sample_windows_from_artifacts(artifacts, 5))
    expected = [
        ['001.p', '002.p', '003.p', '004.p', '005.p'],
    ]
    assert actual == expected


def test_sample_windows_from_artifacts_no_result():
    artifacts = ['001.p', '002.p', '003.p', '004.p']
    actual = list(sample_windows_from_artifacts(artifacts, 5))
    assert actual == []


def test_systematic_sample_from_many_artifacts():
    artifacts = list(range(20, 0, -1))
    n_artifacts = 5
    selected_artifacts = sample_systematic_from_artifacts(artifacts, n_artifacts)
    assert selected_artifacts == [18, 14, 10, 6, 2]
    assert len(selected_artifacts) == n_artifacts


def test_systematic_sample_from_few_artifacts():
    artifacts = ['0', '1', '2', '3', '4', '5', '6']
    n_artifacts = 5
    selected_artifacts = sample_systematic_from_artifacts(artifacts, n_artifacts)
    assert selected_artifacts[0] == '0'
    assert selected_artifacts[4] == '4'
    assert len(selected_artifacts) == n_artifacts


def test_systematic_sample_from_artifacts_too_few():
    artifacts = list(range(3, 0, -1))
    n_artifacts = 5
    with pytest.raises(Exception):
        sample_systematic_from_artifacts(artifacts, n_artifacts)


def test_regex_pickle():
    fname = "pc_1583462470-16tvfmb1d0_1591122155216_100_000.p"

    match_result = REGEX_PICKLE.search(fname)
    assert match_result.group("qrcode") == "1583462470-16tvfmb1d0"
    assert match_result.group("unixepoch") == "1591122155216"
    assert match_result.group("code") == "100"
    assert match_result.group("idx") == "000"
