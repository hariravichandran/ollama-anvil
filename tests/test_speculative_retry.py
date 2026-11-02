"""Unit tests for the shared Ollama POST retry helper in anvil.llm.speculative.

These exercise the retry policy only, with ``requests.post`` and ``time.sleep``
replaced, so they run without a live Ollama server.
"""

from __future__ import annotations

from unittest import mock

import pytest
import requests

from anvil.llm import speculative
from anvil.llm.speculative import _post_with_retry


class _OkResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"ok": True}


@pytest.fixture
def no_sleep():
    with mock.patch.object(speculative.time, "sleep") as sleep:
        yield sleep


def test_returns_json_on_first_success(no_sleep):
    with mock.patch.object(speculative.requests, "post", return_value=_OkResponse()) as post:
        assert _post_with_retry("http://x/api", {"a": 1}, 5) == {"ok": True}
    assert post.call_count == 1
    no_sleep.assert_not_called()


def test_retries_connection_errors_then_succeeds(no_sleep):
    side_effect = [requests.ConnectionError("down"), requests.Timeout("slow"), _OkResponse()]
    with mock.patch.object(speculative.requests, "post", side_effect=side_effect) as post:
        assert _post_with_retry("http://x/api", {}, 5) == {"ok": True}
    assert post.call_count == 3
    # Backoff grows linearly: 2s after attempt 1, 4s after attempt 2.
    assert [c.args[0] for c in no_sleep.call_args_list] == [2, 4]


def test_raises_after_max_attempts(no_sleep):
    with mock.patch.object(
        speculative.requests, "post", side_effect=requests.ConnectionError("down")
    ) as post, pytest.raises(requests.ConnectionError):
        _post_with_retry("http://x/api", {}, 5)
    assert post.call_count == speculative._MAX_ATTEMPTS
    assert no_sleep.call_count == speculative._MAX_ATTEMPTS - 1


def test_passes_url_payload_and_timeout(no_sleep):
    with mock.patch.object(speculative.requests, "post", return_value=_OkResponse()) as post:
        _post_with_retry("http://x/v1/chat/completions", {"model": "m"}, 42)
    post.assert_called_once_with("http://x/v1/chat/completions", json={"model": "m"}, timeout=42)
