"""
csdm.py — Cold Start Detection Module (CSDM)

Splits any dataset into two in-memory subsets using the cold_start_flag column:
    - cold_start_flag == 1  → cold-start path  (MPMN+VML)
    - cold_start_flag == 0  → non-cold-start path (GATEFuse)

Also provides merge_results() to recombine both outputs by original index
after inference.

Note: classify_cold_start() (rule-based flagging for deployment) is paused.
      This file is used purely for routing during training and testing.
"""

import pandas as pd

COLD_START_FLAG_COL = "cold_start_flag"


def split_by_cold_start(df: pd.DataFrame) -> tuple:
    """
    Splits df into two in-memory subsets using the cold_start_flag column.

    Returns:
        (cold_df, non_cold_df) — both retain their original index.

    Raises:
        ValueError if cold_start_flag column is missing.
    """
    if COLD_START_FLAG_COL not in df.columns:
        raise ValueError(
            f"Column '{COLD_START_FLAG_COL}' not found in DataFrame. "
            "Ensure your dataset includes this flag before calling split_by_cold_start()."
        )

    cold_df     = df[df[COLD_START_FLAG_COL] == 1].copy()
    non_cold_df = df[df[COLD_START_FLAG_COL] == 0].copy()

    total = len(df)
    print(f"[CSDM] Total observations  : {total}")
    print(f"[CSDM] Cold-start          : {len(cold_df):>6} ({100 * len(cold_df) / total:.1f}%)")
    print(f"[CSDM] Non-cold-start      : {len(non_cold_df):>6} ({100 * len(non_cold_df) / total:.1f}%)")

    return cold_df, non_cold_df


def merge_results(
    cold_results: pd.DataFrame,
    non_cold_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Concatenates predictions from both paths and restores original row order.

    Both DataFrames must retain the original index from the input dataset
    (cold_start_test.py and non_cold_start_test.py both preserve index).
    """
    if cold_results.empty and non_cold_results.empty:
        raise ValueError("Both cold and non-cold result DataFrames are empty.")
    if cold_results.empty:
        return non_cold_results.sort_index()
    if non_cold_results.empty:
        return cold_results.sort_index()

    merged = pd.concat([cold_results, non_cold_results]).sort_index()
    return merged