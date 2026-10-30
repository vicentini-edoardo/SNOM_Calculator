import pytest

from snom_calculator._preset_defaults import PRESETS, preset_nm


def test_report_table_rows():
    # final-report.html, "Resulting M/N by minimum distance"
    assert preset_nm("Strong", "isotropic", True, 3) == (60, 80)
    assert preset_nm("Very strong", "uniaxial", False, 2) == (240, 80)


def test_no_default_below_fitted_range_or_off_ladder():
    assert preset_nm("Strong", "uniaxial", False, .5) is None
    assert preset_nm("Very strong", "uniaxial", True, 1) is None


@pytest.mark.parametrize("preset", PRESETS)
def test_larger_gap_never_needs_more(preset):
    for kind in ("isotropic", "uniaxial"):
        picks = [preset_nm(preset, kind, False, g) for g in (1, 2, 5, 10)]
        picks = [p for p in picks if p]
        assert picks == sorted(picks, reverse=True)


def _window(name, start, end):
    import numpy as np

    from snom_calculator.materials import catalog_material
    material = catalog_material(name, np.arange(start - 50., end + 51.))
    return material


@pytest.mark.parametrize("name, window, expected", [
    ("aSiO2", (1062, 1269), "Weak"), ("calcite", (1385, 1570), "Weak"), ("hBN", (1336, 1638), "Weak"),
    ("MgO", (400, 757), "Strong"), ("SiC6H", (769, 991), "Strong"),
    ("SiC3C", (777, 992), "Strong"), ("SiC4H", (763, 989), "Very strong"),
    ("SiC4H", (1300, 1500), "Weak"),  # far from the reststrahlen band
])
def test_automatic_preset_from_beta(name, window, expected):
    from snom_calculator._preset_defaults import automatic_preset
    assert automatic_preset(_window(name, *window)) == expected


def test_stack_takes_largest_m_and_n_at_each_distance():
    from snom_calculator._preset_defaults import stack_nm
    top = ("Weak", "isotropic", True, 1.)
    buried = ("Very strong", "isotropic", False, 11.)  # substrate under a 10 nm film
    assert stack_nm([top]) == preset_nm("Weak", "isotropic", True, 1.)
    expected = [preset_nm(*entry) for entry in (top, buried)]
    assert stack_nm([top, buried]) == (max(e[0] for e in expected), max(e[1] for e in expected))
    assert stack_nm([("Very strong", "uniaxial", False, .5)]) is None
    assert stack_nm([]) is None


def test_stack_k_uses_the_stack_table_and_falls_back_to_fim():
    from snom_calculator._preset_defaults import stack_k
    assert stack_k([("Weak", 50.)]) == 4  # single film: film table
    assert stack_k([("Strong", 50.)]) == 16 and stack_k([("Very strong", 50.)]) is None
    assert stack_k([("Weak", 50.), ("Weak", 7.)]) == 12  # thinnest film
    assert stack_k([("Weak", 50.), ("Strong", 100.)]) is None  # strongest preset: Strong stacks use FIM
    assert stack_k([("Weak", 1.), ("Weak", 100.)]) is None  # thinner than the study


def test_beta_uses_the_passive_lossless_uniaxial_branch():
    import numpy as np

    from snom_calculator._preset_defaults import max_beta
    from snom_calculator.materials import MaterialSpectrum

    material = MaterialSpectrum(np.array([-9]), np.array([-4]), "negative axes")
    assert max_beta(material) == pytest.approx(1.4)
