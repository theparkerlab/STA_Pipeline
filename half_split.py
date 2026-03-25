"""
Time-based pseudo-half splitting for EBC half-check and bootstrap half-null thresholds.

Interleaved: even-indexed time blocks [0, B), [2B, 3B), ... -> pseudo-half A (maps to "half 1");
odd blocks -> pseudo-half B ("half 2"). Contiguous: first floor(n/2) rows -> A, rest -> B.
"""
from __future__ import annotations

import warnings

import numpy as np


def interleaved_half_a_mask_times(times_s: np.ndarray, block_sec: float) -> np.ndarray:
    """True where time falls in even-indexed blocks of length block_sec."""
    if block_sec <= 0:
        raise ValueError("interleave_block_sec must be positive")
    t = np.asarray(times_s, dtype=float)
    block_idx = np.floor(t / block_sec).astype(np.int64)
    return (block_idx % 2) == 0


def contiguous_half_a_mask_nrows(n_rows: int) -> np.ndarray:
    """First half of row indices (iloc order), same as legacy split."""
    if n_rows <= 0:
        return np.zeros(0, dtype=bool)
    half_len = n_rows // 2
    m = np.zeros(n_rows, dtype=bool)
    m[:half_len] = True
    return m


def dlc_interleaved_masks(
    times_s: np.ndarray,
    n_frames: int,
    half_split_mode: str,
    interleave_block_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Boolean masks length n_frames aligned to dlc_df iloc order.
    Returns (mask_half_a, mask_half_b) for pseudo-half 1 and 2.
    """
    if half_split_mode not in ("contiguous", "interleaved"):
        raise ValueError("half_split_mode must be 'contiguous' or 'interleaved'")

    if half_split_mode == "contiguous":
        m_a = contiguous_half_a_mask_nrows(n_frames)
    else:
        m_a = interleaved_half_a_mask_times(times_s, interleave_block_sec)
        m_b = ~m_a
        if not np.any(m_a) or not np.any(m_b):
            warnings.warn(
                "Interleaved split yielded an empty pseudo-half on DLC timeline; "
                "falling back to contiguous split.",
                UserWarning,
                stacklevel=2,
            )
            m_a = contiguous_half_a_mask_nrows(n_frames)

    m_b = ~m_a
    return m_a, m_b


def bootstrap_sample_times_for_rows(model_data_df, model_t: np.ndarray, model_dt: float) -> np.ndarray:
    """Sample time at each model grid row (matches filter_and_interpolate: model_t + model_dt/2)."""
    idx = model_data_df.index.to_numpy()
    if not np.issubdtype(idx.dtype, np.integer) and not np.issubdtype(idx.dtype, np.unsignedinteger):
        idx = idx.astype(np.int64, copy=False)
    return model_t[idx] + model_dt / 2.0


def bootstrap_row_half_a_mask(
    model_data_df,
    model_t: np.ndarray,
    model_dt: float,
    half_split_mode: str,
    interleave_block_sec: float,
) -> np.ndarray:
    """
    Boolean mask length len(model_data_df) in iloc order: True -> pseudo-half 1 (A).
    """
    n = len(model_data_df)
    if half_split_mode == "contiguous":
        return contiguous_half_a_mask_nrows(n)

    sample_times = bootstrap_sample_times_for_rows(model_data_df, model_t, model_dt)
    m_a = interleaved_half_a_mask_times(sample_times, interleave_block_sec)
    if not np.any(m_a) or not np.any(~m_a):
        warnings.warn(
            "Interleaved split yielded an empty pseudo-half on model timeline; "
            "falling back to contiguous split.",
            UserWarning,
            stacklevel=2,
        )
        return contiguous_half_a_mask_nrows(n)
    return m_a


def index_in_half_a_map(model_data_df, row_half_a_mask: np.ndarray) -> dict:
    """Map model_data_df index label -> True if pseudo-half A."""
    labels = model_data_df.index.to_numpy()
    return {labels[i]: bool(row_half_a_mask[i]) for i in range(len(model_data_df))}
