from streamlit.testing.v1 import AppTest

import snom_calculator._ui_save as save_module


def test_save_as_writes_selected_file_and_cancel_keeps_it(tmp_path, monkeypatch):
    monkeypatch.setenv("SNOM_NATIVE_SAVE_DIALOG", "1")
    target = tmp_path / "chosen name.json"
    choices = iter([target, None])
    monkeypatch.setattr(save_module, "_choose_save_path", lambda name: next(choices))
    app = AppTest.from_string('''
import streamlit as st
from snom_calculator._ui_save import save_button
save_button(st, "Save setup", data=b'{"setup": true}', file_name="setup.json", mime="application/json")
''').run()
    app.button[0].click().run()
    assert not app.exception
    assert target.read_bytes() == b'{"setup": true}'
    target.write_bytes(b"unchanged")
    app.button[0].click().run()
    assert not app.exception
    assert target.read_bytes() == b"unchanged"


def test_save_failure_is_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("SNOM_NATIVE_SAVE_DIALOG", "1")
    monkeypatch.setattr(save_module, "_choose_save_path", lambda name: tmp_path / "missing" / name)
    app = AppTest.from_string('''
import streamlit as st
from snom_calculator._ui_save import save_button
save_button(st, "Save setup", data=b'{}', file_name="setup.json", mime="application/json")
''').run()
    app.button[0].click().run()
    assert not app.exception
    assert "Could not save file" in app.error[0].value


def test_browser_save_shows_only_original_button(monkeypatch):
    monkeypatch.delenv("SNOM_NATIVE_SAVE_DIALOG", raising=False)
    app = AppTest.from_string('''
import streamlit as st
from snom_calculator._ui_save import save_button
save_button(st, "Save setup", data=b'{}', file_name="setup.json", mime="application/json")
''').run()
    assert not app.exception
    assert not app.text_input
    assert app.get("download_button")[0].proto.label == "Save setup"
