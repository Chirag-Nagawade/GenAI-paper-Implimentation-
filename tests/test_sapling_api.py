
"""Unit tests for the Sapling API integration.

The tests use unittest and unittest.mock to avoid real network
traffic or dependence on Streamlit secrets.
"""

import json
import unittest
from unittest import mock

import requests

from ai_detect.sapling_api import detect_with_sapling


def _make_response(status=200, json_data=None, raise_for=None):
    """Build a mock requests.Response object."""
    mock_resp = mock.Mock()
    mock_resp.status_code = status
    mock_resp.ok = 200 <= status < 300

    if raise_for:
        mock_resp.json.side_effect = raise_for
    else:
        mock_resp.json.return_value = json_data or {}

    return mock_resp


class TestSaplingAPI(unittest.TestCase):

    def setUp(self):
        # Pretend the API key is present by default.
        self.patcher_key = mock.patch(
            "ai_detect.sapling_api._get_api_key",
            return_value="FAKE_KEY"
        )
        self.mock_key = self.patcher_key.start()

        # Prevent real HTTP requests.
        self.patcher_post = mock.patch(
            "ai_detect.sapling_api.requests.post"
        )
        self.mock_post = self.patcher_post.start()

    def tearDown(self):
        self.patcher_key.stop()
        self.patcher_post.stop()

    # 1. Successful response with all fields
    def test_successful_response(self):
        payload = {
            "score": 0.78,
            "verdict": "ai",
            "ai_fraction": 0.65,
            "low_confidence": False,
            "low_confidence_reasons": None,
            "sentence_scores": [0.1, 0.9],
            "version": "1.2.3",
        }

        self.mock_post.return_value = _make_response(
            json_data=payload
        )

        res = detect_with_sapling("some meaningful text")

        self.assertIn("score", res)
        self.assertEqual(res["score"], 0.78)
        self.assertEqual(res["detector_version"], "1.2.3")
        self.assertNotIn("error", res)

    # 2. Missing API key
    def test_missing_api_key(self):
        self.mock_key.return_value = None

        res = detect_with_sapling("text")

        self.assertIn("error", res)
        self.assertIn("key", res["error"].lower())
        self.mock_post.assert_not_called()

    # 3. Empty input
    def test_empty_input(self):
        res = detect_with_sapling("")

        self.assertIn("error", res)
        self.assertIn("empty", res["error"].lower())
        self.mock_post.assert_not_called()

    # 4. HTTP 401
    def test_http_401(self):
        self.mock_post.return_value = _make_response(status=401)

        res = detect_with_sapling("text")

        self.assertIn("error", res)
        self.assertIn("authentication", res["error"].lower())

    # 5. HTTP 429
    def test_http_429(self):
        self.mock_post.return_value = _make_response(status=429)

        res = detect_with_sapling("text")

        self.assertIn("error", res)
        self.assertIn("rate limit", res["error"].lower())

    # 6. Other HTTP error (e.g., 500)
    def test_other_http_error(self):
        self.mock_post.return_value = _make_response(status=500)

        res = detect_with_sapling("text")

        self.assertIn("error", res)
        self.assertIn("HTTP 500", res["error"])

    # 7. Network timeout
    def test_network_timeout(self):
        self.mock_post.side_effect = requests.exceptions.Timeout(
            "timeout"
        )

        res = detect_with_sapling("text")

        self.assertIn("error", res)
        self.assertIn("network", res["error"].lower())

    # 8. Invalid JSON response
    def test_invalid_json(self):
        self.mock_post.return_value = _make_response(
            raise_for=json.JSONDecodeError("msg", "doc", 0)
        )

        res = detect_with_sapling("text")

        self.assertIn("error", res)
        self.assertIn("json", res["error"].lower())

    # 9. Missing required score field
    def test_missing_score(self):
        payload = {"verdict": "ai"}

        self.mock_post.return_value = _make_response(
            json_data=payload
        )

        res = detect_with_sapling("text")

        self.assertIn("error", res)
        self.assertIn("missing required field", res["error"].lower())

    # 10. Invalid score type or out-of-range score
    def test_invalid_score(self):
        payload = {"score": 1.5, "verdict": "ai"}

        self.mock_post.return_value = _make_response(
            json_data=payload
        )

        res = detect_with_sapling("text")

        self.assertIn("error", res)
        self.assertIn("score", res["error"].lower())

    # 11. Low-confidence response
    def test_low_confidence_response(self):
        payload = {
            "score": 0.45,
            "verdict": "review",
            "low_confidence": True,
            "low_confidence_reasons": "Too short",
        }

        self.mock_post.return_value = _make_response(
            json_data=payload
        )

        res = detect_with_sapling("short text")

        self.assertFalse(res.get("error"))
        self.assertTrue(res.get("low_confidence"))
        self.assertEqual(
            res.get("low_confidence_reasons"),
            "Too short"
        )

    # 12. Short non-empty text is still sent
    def test_short_nonempty_text_sent(self):
        payload = {"score": 0.5, "verdict": "human"}

        self.mock_post.return_value = _make_response(
            json_data=payload
        )

        short = "abc"
        res = detect_with_sapling(short)

        self.mock_post.assert_called_once()
        self.assertIn("score", res)
        self.assertEqual(res["verdict"], "human")


if __name__ == "__main__":
    unittest.main()
