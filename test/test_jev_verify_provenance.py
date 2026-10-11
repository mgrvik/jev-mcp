import json
from types import SimpleNamespace

import pytest

import veda_jev_mcp as jev


def _result(provider, model):
    def fake_provider_answers(pack, state):
        answers = {
            key: SimpleNamespace(value="supports", confidence=0.99, probabilities={"supports": 0.99})
            for key in pack.questions
        }
        return SimpleNamespace(provider=provider, model_resolved=model, answers=answers)
    return fake_provider_answers


def _payload(response):
    assert response["isError"] is True
    return json.loads(response["content"][0]["text"])


@pytest.mark.parametrize(("provider", "model"), [
    ("mock", "mock-decision-provider"),
    ("typesafe-jev", "agent-jev-0.6b"),
    ("typesafe-jev", "laya-421m-onnx"),
    ("typesafe-jev", "offline-heuristic"),
    ("", ""),
])
def test_jev_verify_returns_typed_provenance_error_without_mock_claims(monkeypatch, provider, model):
    monkeypatch.setattr(jev, "_provider_answers", _result(provider, model))

    error = _payload(jev.jev_verify({"claims": ["the exact claim"], "evidence": []}))

    assert error["error"].startswith("CLOUD_JUDGE_PROVENANCE_UNVERIFIED")
    assert error["verified_claims"] == 0
    assert error["offline_fallback"] is True
    assert "summary" not in error
    assert "results" not in error


def test_jev_verify_preserves_claim_results_for_canonical_cloud_model(monkeypatch):
    monkeypatch.setattr(jev, "_provider_answers", _result("typesafe-jev", "jev-latest"))
    monkeypatch.setattr(jev, "service", lambda: SimpleNamespace(
        provider=SimpleNamespace(base_url="https://api.typesafe.ai")))

    response = jev.jev_verify({"claims": ["the exact claim"], "evidence": []})

    assert response["isError"] is False
    result = json.loads(response["content"][0]["text"])
    assert result["engine"] == "typesafe-jev"
    assert result["model"] == "jev-latest"
    assert result["endpoint"] == "https://api.typesafe.ai/v1/systemone"
    assert result["summary"]["verified"] == 1


def test_jev_verify_rejects_local_endpoint_even_with_cloud_labels(monkeypatch):
    monkeypatch.setattr(jev, "_provider_answers", _result("typesafe-jev", "jev-latest"))
    monkeypatch.setattr(jev, "service", lambda: SimpleNamespace(
        provider=SimpleNamespace(base_url="http://127.0.0.1:5205?token=must-not-leak")))

    error = _payload(jev.jev_verify({"claims": ["the exact claim"], "evidence": []}))

    assert error["error"].startswith("CLOUD_JUDGE_PROVENANCE_UNVERIFIED")
    assert error["endpoint"] == "http://127.0.0.1:5205/v1/systemone"
    assert "token" not in str(error)
    assert "summary" not in error


def test_jev_verify_returns_actionable_provider_unavailable(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise TimeoutError("private prompt details omitted")
    monkeypatch.setattr(jev, "_provider_answers", unavailable)

    error = _payload(jev.jev_verify({"claims": ["the exact claim"], "evidence": []}))

    assert error["error"] == "PROVIDER_UNAVAILABLE: verification provider call failed"
    assert error["error_type"] == "TimeoutError"
    assert error["verified_claims"] == 0
    assert "private prompt details" not in str(error)
