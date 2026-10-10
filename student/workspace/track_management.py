"""Track initialization, scoring, and deletion helpers.

Part H supplies lidar-driven existence decisions (docs/HUONG_DAN_KY_THUAT.md §2).
Use tracking parameters for the score window, thresholds, and covariance limit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fusion_lab.workspace_support import get_tracking_params


def init_track_state_from_meas(meas: Any) -> dict[str, Any]:
    """Initialize track state, covariance, lifecycle state, and score from a measurement.

    Args:
        meas: Lidar measurement with ``z``, ``R``, ``sensor``.

    Returns:
        Dict with keys ``x``, ``P``, ``state``, ``score`` (matrices as ``np.matrix``).
    """
    params = get_tracking_params()
    z = np.asarray(meas.z, dtype=float).reshape(3, 1)
    T = np.asarray(meas.sensor.sens_to_veh, dtype=float)
    R_sens_to_veh = T[:3, :3]
    t_sens_to_veh = T[:3, 3:4]

    pos_veh = R_sens_to_veh @ z + t_sens_to_veh
    vel_veh = np.zeros((3, 1), dtype=float)
    x = np.vstack([pos_veh, vel_veh])

    P_pos = R_sens_to_veh @ np.asarray(meas.R, dtype=float) @ R_sens_to_veh.T
    P = np.zeros((6, 6), dtype=float)
    P[:3, :3] = P_pos
    P[3, 3] = params.sigma_p44**2
    P[4, 4] = params.sigma_p55**2
    P[5, 5] = params.sigma_p66**2

    score = 1.0 / params.window
    state = "initialized"

    return {
        "x": np.asmatrix(x),
        "P": np.asmatrix(P),
        "state": state,
        "score": score,
    }


def update_track_score(track: dict[str, Any], associated: bool) -> dict[str, Any]:
    """Update existence once per lidar frame; camera passes never call this helper.

    A hit adds 1/window, capped at one; an in-FOV miss subtracts 1/window.
    Confirm above confirmed_threshold, and preserve confirmed state after misses.

    Args:
        track: Dict-like track with ``score``, ``state``.
        associated: True for a lidar hit; False for a lidar miss within the lidar FOV.

    Returns:
        Updated track dict.
    """
    params = get_tracking_params()
    score = float(track["score"])
    state = str(track["state"])

    if associated:
        score = min(1.0, score + 1.0 / params.window)
        if state != "confirmed":
            state = "confirmed" if score > params.confirmed_threshold else "tentative"
    else:
        score = score - 1.0 / params.window

    track["score"] = score
    track["state"] = state
    return track


def should_delete_track(track: dict[str, Any]) -> bool:
    """Return whether a lidar lifecycle pass should remove this track.

    Delete if either horizontal variance exceeds max_P, or if a confirmed
    track has score < delete_threshold, or an unconfirmed track has score <= 0.
    Camera passes never trigger deletion.

    Args:
        track: Dict with ``score``, ``state``, ``P``.

    Returns:
        True if track should be removed.
    """
    params = get_tracking_params()
    P = np.asarray(track.get("P", np.eye(6)), dtype=float)
    pxx = float(P[0, 0])
    pyy = float(P[1, 1])

    if pxx > params.max_P or pyy > params.max_P:
        return True

    state = track.get("state")
    score = float(track.get("score", 0.0))

    if state == "confirmed":
        return bool(score < params.delete_threshold)
    else:
        return bool(score <= 0.0)
