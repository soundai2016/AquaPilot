import json

import pytest

from aquapilot.utils import save_json


def test_nonfinite_write_leaves_existing_document_intact(tmp_path):
    destination = tmp_path / "metrics.json"
    save_json(destination, {"run": "complete", "value": 2.0})
    original = destination.read_bytes()
    with pytest.raises(ValueError):
        save_json(destination, {"value": float("nan")})
    assert destination.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_json_preserves_undefined_value_as_null(tmp_path):
    destination = tmp_path / "metrics.json"
    save_json(destination, {"rank": None, "undefined_count": 4})
    assert json.loads(destination.read_text()) == {"rank": None, "undefined_count": 4}
