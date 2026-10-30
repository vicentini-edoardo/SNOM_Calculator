from io import BytesIO

import io
from pathlib import Path

import numpy as np
import pytest

from snom_calculator.io import combined_results_csv, map_experiment_columns, parse_experiment, read_experiment_columns, result_csv, read_numeric_table
from snom_calculator.models import SpectrumResult


@pytest.mark.parametrize("unwrap", [True, False])
def test_experimental_phase_rotation_preserves_complex_signal(unwrap):
    from snom_calculator.io import adjust_experimental_phase

    frequency = np.array([900., 901., 904.])
    phase = np.array([3., -3., -2.5])
    original = phase.copy()
    offset, tilt = 7., 4.
    adjusted = adjust_experimental_phase(frequency, phase, offset, tilt, unwrap=unwrap)
    expected = np.unwrap(phase) + offset + tilt * (frequency - frequency[0])
    np.testing.assert_allclose(np.exp(1j * adjusted), np.exp(1j * phase) * np.exp(1j * (offset + tilt * (frequency - frequency[0]))))
    if unwrap:
        np.testing.assert_allclose(adjusted, expected)
    else:
        assert np.all(np.abs(adjusted) <= np.pi)
    np.testing.assert_array_equal(phase, original)
    np.testing.assert_allclose(adjust_experimental_phase(frequency, phase, 0., 0., unwrap=False), phase)


def test_experiment_phase_is_optional():
    values = parse_experiment(
        io.BytesIO(b"frequency_cm1,amplitude\n900,1.1\n901,1.2\n")
    )
    assert values["phase_rad"] is None
    np.testing.assert_allclose(values["amplitude"], [1.1, 1.2])


def test_result_csv_contains_complex_signal_and_metadata():
    result = SpectrumResult.from_contrast(
        frequency_cm1=np.array([900.0, 901.0]),
        sample_harmonic=np.array([1 + 2j, 2 + 3j]),
        reference_harmonic=np.ones(2, dtype=complex),
        contrast=np.array([1 + 2j, 2 + 3j]),
        permittivities={"sample": np.array([4 + 0.1j, 5 + 0.2j])},
        metadata={"model": "bulk", "harmonic": 3},
    )
    text = result_csv(result)
    assert "contrast_real,contrast_imag,amplitude,phase_rad" in text
    assert "# model: bulk" in text
    assert "sample_epsilon_real" in text


def test_experiment_accepts_whitespace_txt():
    values = parse_experiment(
        io.BytesIO(b"frequency_cm1 amplitude phase_rad\n900 1.1 0.2\n901 1.2 0.3\n")
    )
    np.testing.assert_allclose(values["phase_rad"], [0.2, 0.3])


def test_experiment_column_mapping_for_named_and_headerless_text():
    headers, rows = read_experiment_columns(io.BytesIO(b"signal frequency phase\n2 900 0.2\n3 901 0.3\n"))
    assert headers == ["signal", "frequency", "phase"]
    values = map_experiment_columns(rows, 1, "Amplitude + phase", (0, 2))
    np.testing.assert_allclose(values["frequency_cm1"], [900, 901])
    np.testing.assert_allclose(values["phase_rad"], [0.2, 0.3])

    headers, rows = read_experiment_columns(io.BytesIO(b"900,3,4\n901,5,12\n"))
    assert headers == ["Column 1", "Column 2", "Column 3"]
    values = map_experiment_columns(rows, 0, "Real + imaginary", (1, 2))
    np.testing.assert_allclose(values["amplitude"], [5, 13])
    np.testing.assert_allclose(values["phase_rad"], np.angle([3 + 4j, 5 + 12j]))

    # A non-finite first data row stays data (and is rejected), not a header.
    headers, rows = read_experiment_columns(io.BytesIO(b"900,nan,4\n901,5,12\n"))
    assert headers == ["Column 1", "Column 2", "Column 3"] and len(rows) == 2
    with pytest.raises(ValueError, match="finite"):
        map_experiment_columns(rows, 0, "Real + imaginary", (1, 2))


def test_experiment_complex_column_and_invalid_selection():
    _, rows = read_experiment_columns(io.BytesIO(b"frequency,signal\n900,3+4i\n901,5+12i\n"))
    values = map_experiment_columns(rows, 0, "Complex", (1,))
    np.testing.assert_allclose(values["amplitude"], [5, 13])
    with np.testing.assert_raises_regex(ValueError, "different column"):
        map_experiment_columns(rows, 0, "Complex", (0,))


def test_fake_experiment_file_loads_real_and_imaginary_columns():
    with (Path(__file__).parent / "fixtures" / "fake_experiment.txt").open("rb") as file:
        headers, rows = read_experiment_columns(file)
    assert headers == ["frequency_cm1", "real", "imag"]
    values = map_experiment_columns(rows, 0, "Real + imaginary", (1, 2))
    np.testing.assert_allclose(values["frequency_cm1"], [900, 901, 902, 903, 904])
    np.testing.assert_allclose(values["amplitude"], [5, 13, 17, 25, 29])
    np.testing.assert_allclose(values["phase_rad"], np.angle([3 + 4j, 5 + 12j, 8 + 15j, 7 + 24j, 20 + 21j]))


@pytest.mark.parametrize("data", [b"900,1,nan\n901,2,0\n", b"900,-1,0\n901,2,0\n"])
def test_experiment_rejects_invalid_signals(data):
    with pytest.raises(ValueError):
        parse_experiment(BytesIO(b"frequency_cm1,amplitude,phase_rad\n" + data))


def test_experiment_skips_metadata_comments():
    data = parse_experiment(BytesIO(b"# exported data\nfrequency_cm1,amplitude\n900,1\n901,2\n"))
    np.testing.assert_array_equal(data["amplitude"], [1, 2])


@pytest.mark.parametrize("reader", [read_experiment_columns, read_numeric_table])
def test_table_readers_ignore_leading_metadata(reader):
    result = reader(BytesIO(
        b"\xef\xbb\xbf# calculation\n% tip settings\n// illumination\n"
        b"Tip radius: 25 nm\nmodel = bulk\n\n"
        b"frequency_cm1,amplitude\n900,1\n901,2\n"
    ))
    if reader is read_experiment_columns:
        assert result == (["frequency_cm1", "amplitude"], [["900", "1"], ["901", "2"]])
    else:
        np.testing.assert_array_equal(result["amplitude"], [1, 2])


def test_combined_export_metadata_roundtrips_and_keeps_harmonic_checks():
    results = {
        n: SpectrumResult.from_contrast(
            frequency_cm1=np.array([900., 901.]),
            sample_harmonic=np.ones(2), reference_harmonic=np.ones(2),
            contrast=np.ones(2) * n, permittivities={},
            metadata={"model": "bulk", "harmonic": n, "note": "first\nsecond",
                      "M_check_max_error": n / 100},
        ) for n in (2, 3)
    }
    text = combined_results_csv(results)
    assert text.startswith("# model: bulk\n")
    assert "# harmonics: [2, 3]" in text
    assert "# second\n" in text
    assert '"3": {"model": "bulk", "harmonic": 3' in text
    headers, rows = read_experiment_columns(BytesIO(text.encode()))
    assert headers == ["wn", "O2A", "O2P", "O3A", "O3P"]
    np.testing.assert_allclose(map_experiment_columns(rows, 0, "Amplitude + phase", (3, 4))["amplitude"], [3, 3])
