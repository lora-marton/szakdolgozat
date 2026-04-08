"""
Session data services.

Handles result persistence, session discovery, result loading,
and SSE event generation. All methods are stateless — state
(like the SSE queue) is passed in by the caller.
"""

import json
import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


class SessionService:
    """Data-access and persistence logic for dance comparison sessions."""

    @staticmethod
    def find_latest_session() -> str | None:
        """Find the most recent session directory inside data/.

        Session directories are named YYYYMMDD_HHMMSS, so reverse-sorting
        by name gives newest first.

        Returns:
            The session directory name, or None if no sessions exist.
        """
        if not os.path.isdir(DATA_DIR):
            return None
        sessions = sorted(
            [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))],
            reverse=True,
        )
        return sessions[0] if sessions else None

    @staticmethod
    def load_results(session_id: str) -> dict | None:
        """Load the results JSON for a given session.

        Args:
            session_id: Session directory name (e.g. '20260306_134500').

        Returns:
            Parsed dict from results.json, or None if the file does not exist.
        """
        results_path = os.path.join(DATA_DIR, session_id, "results.json")
        if not os.path.isfile(results_path):
            return None
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_results(session_id: str, results: dict) -> None:
        """Persist comparison results to disk as JSON.

        Walks the results dict and converts numpy arrays to lists
        for JSON serialization. Handles one level of nesting (dicts
        containing arrays).

        Args:
            session_id: Session directory name (e.g. '20260306_134500').
            results: Dict returned by VideoProcessor.process_videos.
        """
        session_dir = os.path.join(DATA_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)

        serializable = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
            elif isinstance(value, dict):
                serializable[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in value.items()}
            else:
                serializable[key] = value

        results_path = os.path.join(session_dir, "results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
