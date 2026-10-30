"""Setup interchange and validation, independent of the Streamlit UI."""

import json

import numpy as np
import pytest

from snom_calculator.setup import _apply_setup, _decode_setup, _default_setup, _encode_setup


def test_setup_export_excludes_pending_m_refinement():
    state = {**_default_setup(), "m_refinement_request": {"error": .2},
             "m_refinement_next": {"error": .1}}
    encoded = json.loads(_encode_setup(state))
    assert "m_refinement_request" not in encoded
    assert "m_refinement_next" not in encoded


def test_setup_json_round_trips_catalog_and_constant_materials():
    state = _default_setup()
    state["finite_layer_ids"] = [1]
    state["next_layer_id"] = 2
    state["layer_1_source"] = "Constant"
    state["layer_1_real"] = 4.0
    state["layer_1_imag"] = 0.2
    state["layer_1_thickness"] = 25.0
    restored = {}

    _apply_setup(restored, _decode_setup(_encode_setup(state).encode()))

    assert restored["finite_layer_ids"] == [1]
    assert restored["layer_1_real"] == 4.0
    assert restored["layer_1_imag"] == 0.2
    assert restored["layer_1_thickness"] == 25.0


def test_setup_json_round_trips_custom_library_material_selection():
    state = _default_setup()
    state["substrate_source"] = "Custom library"
    state["substrate_custom"] = "My substrate"
    state["reference_source"] = "Custom library"
    state["reference_custom"] = "My reference"
    restored = {}

    _apply_setup(restored, _decode_setup(_encode_setup(state).encode()))

    assert restored["substrate_source"] == "Custom library"
    assert restored["substrate_custom"] == "My substrate"
    assert restored["reference_source"] == "Custom library"
    assert restored["reference_custom"] == "My reference"


def test_invalid_setup_does_not_partially_mutate_state():
    state = {"sentinel": "unchanged"}
    bad = json.dumps({"frequency": {"start": 1200, "end": 600, "points": 1}, "setup_version": 2}).encode()

    with pytest.raises(ValueError, match="setup"):
        _apply_setup(state, _decode_setup(bad))

    assert state == {"sentinel": "unchanged"}


def test_setup_json_round_trips_drude_terms():
    state = _default_setup()
    state["finite_layer_ids"] = [1]
    state["next_layer_id"] = 2
    state.update({"layer_1_source": "Catalog", "layer_1_catalog": "SiC4H", "layer_1_thickness": 20.0,
                  "layer_1_drude_enabled": True, "layer_1_drude_wp_cm1": 1500.0, "layer_1_drude_gamma_cm1": 40.0,
                  "substrate_drude_enabled": True, "substrate_drude_wp_cm1": 900.0})
    restored = {}

    _apply_setup(restored, _decode_setup(_encode_setup(state).encode()))

    assert restored["layer_1_drude_enabled"] is True
    assert restored["layer_1_drude_wp_cm1"] == 1500.0
    assert restored["layer_1_drude_gamma_cm1"] == 40.0
    assert restored["substrate_drude_wp_cm1"] == 900.0


def test_setup_saved_before_the_m_n_rename_is_rejected():
    # Its "bulk_custom_N" held tapping points, which are M now.
    with pytest.raises(ValueError, match="before the M"):
        _decode_setup(json.dumps({"bulk_custom_N": 80, "bulk_custom_m": 40}).encode())
    assert _decode_setup(json.dumps({"setup_version": 2}).encode()) == {"setup_version": 2}


def test_setup_rejects_negative_drude_frequency():
    with pytest.raises(ValueError, match="setup"):
        _apply_setup({}, {"substrate_drude_wp_cm1": -1.0})


def test_invalid_unlisted_layer_thickness_is_rejected():
    with pytest.raises(ValueError, match="setup"):
        _apply_setup({}, {"finite_layer_ids": [1], "layer_2_thickness": -1.0})


def test_setup_rejects_a_declared_layer_without_its_fields():
    with pytest.raises(ValueError, match="setup"):
        _apply_setup({}, {"finite_layer_ids": [1]})


def test_setup_rejects_empty_custom_library_layer_name():
    with pytest.raises(ValueError, match="setup"):
        _apply_setup(
            {},
            {
                "finite_layer_ids": [1],
                "next_layer_id": 2,
                "layer_1_source": "Custom library",
                "layer_1_custom": "",
                "layer_1_thickness": 20.0,
            },
        )


def test_setup_rejects_a_next_layer_id_that_is_already_in_use():
    with pytest.raises(ValueError, match="setup"):
        _apply_setup(
            {},
            {
                "finite_layer_ids": [1],
                "next_layer_id": 1,
                "layer_1_source": "Catalog",
                "layer_1_catalog": "SiC4H",
                "layer_1_thickness": 20.0,
            },
        )


def test_setup_restore_keeps_custom_precision_and_reference_catalog():
    restored = {}

    _apply_setup(
        restored,
        {
            "preset": "Custom",
            "reference_source": "Catalog",
            "reference_catalog": "SiC4H",
            "bulk_custom_M": 42,
            "bulk_custom_N": 43,
            "bulk_custom_quad": 44,
            "layer_custom_M": 45,
            "layer_custom_N": 46,
            "layer_custom_quad": 47,
            "layer_custom_K": 12,
        },
    )

    assert (restored["precision_mode"], restored["preset"]) == ("Custom", "Weak")  # saved before Automatic
    assert restored["reference_catalog"] == "SiC4H"
    assert [restored[key] for key in (
        "bulk_custom_M", "bulk_custom_N", "layer_custom_M", "layer_custom_N", "layer_custom_K",
    )] == [42, 43, 45, 46, 12]
    assert "bulk_custom_quad" not in restored
    assert "layer_custom_quad" not in restored


@pytest.mark.parametrize("enabled", [True, False])
def test_setup_restores_fim_instead_of_retaining_the_current_choice(enabled):
    saved = _default_setup()
    saved["force_fim"] = enabled
    restored = {"force_fim": not enabled}
    _apply_setup(restored, _decode_setup(_encode_setup(saved).encode()))
    assert restored["force_fim"] is enabled


def test_older_setup_resets_fim_to_automatic():
    restored = {"force_fim": True}
    _apply_setup(restored, {})
    assert restored["force_fim"] is False


def test_setup_export_omits_obsolete_quadrature_settings():
    state = {**_default_setup(), "bulk_custom_quad": 16, "layer_custom_quad": 44}
    saved = json.loads(_encode_setup(state))
    assert "bulk_custom_quad" not in saved
    assert "layer_custom_quad" not in saved


def test_setup_migrates_old_default_precision_without_retaining_quadrature():
    state = {}
    _apply_setup(state, {"bulk_custom_M": 30, "bulk_custom_N": 8, "bulk_custom_quad": 16})
    assert (state["bulk_custom_M"], state["bulk_custom_N"]) == (80, 80)
    assert "bulk_custom_quad" not in state


def test_setup_restore_discards_serialized_material_uploads():
    state = {"layer_1_upload": object()}

    _apply_setup(
        state,
        {
            "finite_layer_ids": [1],
            "next_layer_id": 2,
            "layer_1_source": "Upload CSV/TXT",
            "layer_1_thickness": 20.0,
            "layer_1_upload": "saved-material.csv",
        },
    )

    assert "layer_1_upload" not in state


def test_setup_export_omits_dialog_state():
    state = {**_default_setup(), "editor": "tip", "editor_snapshot": {"tip_R": 25.0},
             "editor_restore": {"tip_R": 25.0}, "editor_cleanup": True,
             "custom_material_builder_name": "unfinished", "substrate_delete_custom_confirm": "x"}

    saved = json.loads(_encode_setup(state))

    assert "editor" not in saved
    assert "editor_snapshot" not in saved
    assert "editor_restore" not in saved
    assert "editor_cleanup" not in saved
    assert "custom_material_builder_name" not in saved
    assert "substrate_delete_custom_confirm" not in saved


def test_advanced_controls_are_saved_and_reject_invalid_ranges():
    state = _default_setup()
    state.update({"average_gaps_enabled": True, "average_gap_min": 2.0,
                  "average_gap_max": 6.0, "average_gap_count": 3,
                  "resolution_enabled": True, "resolution_fwhm": 4.0})
    restored = {}

    _apply_setup(restored, _decode_setup(_encode_setup(state).encode()))

    assert all(restored[key] == state[key] for key in (
        "average_gaps_enabled", "average_gap_min", "average_gap_max",
        "average_gap_count", "resolution_enabled", "resolution_fwhm",
    ))
    with pytest.raises(ValueError, match="setup"):
        _apply_setup({}, {**state, "average_gap_max": 1.0})


@pytest.mark.parametrize("setup", [
    {"workers": 5}, {"far_angle": np.nan}, {"far_angle": 90}, {"far_enabled": "false"},
    {"preset": "unknown"}, {"tip_eps_real": "bad"}, {"substrate_real": "bad"},
    {"bulk_custom_M": "bad"}, {"check_n": "false"}, {"force_fim": "false"}, {"frequency_start": -5},
])
def test_invalid_setup_values_do_not_mutate_session(setup):
    from snom_calculator.setup import _apply_setup

    state = {"sentinel": "unchanged"}
    with pytest.raises(ValueError, match="setup"):
        _apply_setup(state, setup)
    assert state == {"sentinel": "unchanged"}


def test_independent_gap_ranges_round_trip_and_legacy_migration():
    state = _default_setup()
    state.update(tip_Hs=3.0, tip_Hs_max=7.0, average_gap_count=3,
                 tip_Hr=1.0, tip_Hr_max=5.0, tip_reference_gap_count=2)
    restored = {}
    _apply_setup(restored, _decode_setup(_encode_setup(state).encode()))
    for key in ("tip_Hs", "tip_Hs_max", "average_gap_count", "tip_Hr", "tip_Hr_max", "tip_reference_gap_count"):
        assert restored[key] == state[key]
    for changes in ({"tip_Hr_max": 0.5}, {"tip_reference_gap_count": 26}):
        with pytest.raises(ValueError, match="setup"):
            _apply_setup({}, {**state, **changes})
    _apply_setup(restored, {"tip_Hs": 8.0, "tip_Hr": 9.0})
    assert restored["tip_Hs_max"] == 8.0
    assert restored["tip_Hr_max"] == 9.0
    _apply_setup(restored, {"average_gaps_enabled": True, "average_gap_min": 3.0, "average_gap_max": 7.0})
    assert (restored["tip_Hs"], restored["tip_Hs_max"]) == (3.0, 7.0)


@pytest.mark.parametrize("saved, expected", [
    ({}, ("Automatic", "Weak")),
    ({"preset": "Very strong"}, ("Presets", "Very strong")),  # saved before Automatic
    ({"precision_mode": "Presets", "preset": "Strong"}, ("Presets", "Strong")),
])
def test_setup_precision_mode(saved, expected):
    restored = {}
    _apply_setup(restored, saved)
    assert (restored["precision_mode"], restored["preset"]) == expected


def test_setup_rejects_unknown_precision_mode():
    with pytest.raises(ValueError):
        _apply_setup({}, {"precision_mode": "Unknown"})


def test_zero_length_tip_is_an_invalid_spheroid():
    from snom_calculator.bulk import calculate_bulk_harmonics
    from snom_calculator.materials import constant_material
    from snom_calculator.models import NumericalSettings, TipParameters

    frequency = np.array([900.0, 950.0])
    sample = constant_material(4 + 0.1j, frequency, "sample")
    with pytest.raises(ValueError, match="L must be greater than 2R"):
        calculate_bulk_harmonics(frequency, sample=sample, reference=None, tip=TipParameters(length_nm=0.0),
                                 numerical=NumericalSettings(M=8, N=8, quadQ=16), harmonics=(2,))
    with pytest.raises(ValueError, match="Invalid setup"):
        _apply_setup({}, {"tip_L": 0.0})


def test_setup_ignores_removed_feature_keys():
    removed = {"field_enabled": True, "field_window_nm": 0,
               "tip_source": "Catalog", "tip_catalog": "Au", "tip_drude_enabled": True,
               "layer_1_tilt_deg": 20., "substrate_twist_deg": 5., "biaxial_charge_rank": 3}
    assert not removed.keys() & _default_setup().keys()
    state = {}
    _apply_setup(state, removed)
    assert not removed.keys() & state.keys()
