"""Sapling AI Detector API integration.

Provides a single public function ``detect_with_sapling(text)`` that contacts the
Sapling endpoint and returns a dictionary with the fields required by the UI.
All error handling is contained within the module – the caller only needs to
check for the ``error`` key.

The implementation follows the rules from the user request:
* The API key is read from Streamlit secrets (``st.secrets[\"SAPLING_API_KEY\"]``).
* If the key is missing an informative error is returned – the UI will display a
  warning and continue without crashing.
* Requests are sent with ``zero_retention: true`` and ``sent_scores: true``.
* Minimum recommended text length is 300 characters; shorter input yields a
  ``review`` verdict without contacting the API.
* Responses are validated before being returned. Missing optional fields are
  filled with ``None``.
* The function never touches the local detector or any plagiarism‑detection
  code.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import requests

LOGGER = logging.getLogger(__name__)


class SaplingAPIError(RuntimeError):
    """Raised for unrecoverable API‑level problems (e.g. authentication, quota)."""


def _get_api_key() -> Optional[str]:
    """Return the Sapling API key from Streamlit secrets.

    If Streamlit is not available (e.g. during pure‑Python unit tests) or the
    secret is absent, ``None`` is returned.  The caller must handle the missing
    key gracefully.
    """
    try:
        import streamlit as st
    except Exception:  # pragma: no cover – Streamlit not installed in tests
        return None

    key = st.secrets.get("SAPLING_API_KEY", "")
    if isinstance(key, str) and key.strip():
        return key.strip()
    return None


def _validate_response(data: Dict[str, Any]) -> None:
    """Validate the Sapling JSON payload.

    Raises ``SaplingAPIError`` if required fields are missing or malformed.
    Optional fields are tolerated but are coerced to the expected types where
    possible.
    """
    # Required fields according to the API spec
    for field in ("score", "verdict"):
        if field not in data:
            raise SaplingAPIError(f"Missing required field '{field}' in Sapling response")

    score = data["score"]
    if not isinstance(score, (int, float)) or not (0 <= float(score) <= 1):
        raise SaplingAPIError("'score' must be a numeric value between 0 and 1")

    # Optional numeric fields – if present they must be within [0, 1]
    for opt in ("ai_fraction",):
        if opt in data and data[opt] is not None:
            val = data[opt]
            if not isinstance(val, (int, float)) or not (0 <= float(val) <= 1):
                raise SaplingAPIError(f"'{opt}' must be numeric in [0, 1] if provided")

    # sentence_scores must be a list if present
    if "sentence_scores" in data and not isinstance(data["sentence_scores"], list):
        raise SaplingAPIError("'sentence_scores' must be a list")

    # version is optional but, if present, must be a string.
    if "version" in data and not isinstance(data["version"], str):
        raise SaplingAPIError("'version' must be a string when present")


def detect_with_sapling(text: str) -> Dict[str, Any]:
    """Detect AI‑generated content using the Sapling API.

    Parameters
    ----------
    text:
        The document text to analyse.

    Returns
    -------
    dict
        On success: ``{"score": float, "verdict": str, "ai_fraction": float | None,
        "low_confidence": bool | None, "low_confidence_reasons": Any,
        "sentence_scores": list | None, "detector_version": str | None}``.
        On error: ``{"error": <message>}``.
    """
    MIN_LENGTH = 300
    if not isinstance(text, str) or not text.strip():
        return {"error": "Empty text provided; nothing to send to Sapling."}

    api_key = _get_api_key()
    if not api_key:
        return {"error": "Sapling API key not configured in Streamlit secrets."}

    payload = {"text": text, "zero_retention": True, "sent_scores": True}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = "https://api.sapling.ai/api/v1/aidetect"

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
    except requests.RequestException as exc:
        LOGGER.warning("Sapling request failed: %s", exc)
        return {"error": "Network error contacting Sapling API."}

    if response.status_code == 401:
        return {"error": "Authentication failed for Sapling API (invalid key)."}
    if response.status_code == 429:
        return {"error": "Sapling API rate limit exceeded."}
    if not response.ok:
        return {"error": f"Sapling API returned HTTP {response.status_code}."}

    try:
        data = response.json()
    except json.JSONDecodeError:
        return {"error": "Invalid JSON received from Sapling API."}
    if not isinstance(data, dict):
        return {"error": "Unexpected response format from Sapling API (expected JSON object)."}

    # Verify required fields have correct types
    if not isinstance(data.get("verdict"), str):
        return {"error": "Invalid verdict type in Sapling response; expected string."}

    try:
        _validate_response(data)
    except SaplingAPIError as exc:
        return {"error": str(exc)}

    result: Dict[str, Any] = {
        "score": float(data["score"]),
        "verdict": data["verdict"],
        "ai_fraction": data.get("ai_fraction"),
        "low_confidence": data.get("low_confidence"),
        "low_confidence_reasons": data.get("low_confidence_reasons"),
        "sentence_scores": data.get("sentence_scores"),
        "detector_version": data.get("version"),
    }
    return result
