"""
Order Book Features Module

Features derived from order book depth data.
Captures liquidity, market microstructure, and depth imbalances.

Data Structure:
- timestamp: Snapshot time
- percentage: Price level as % from mid (-5 to +5)
- depth: Quantity at this level
- notional: Dollar value at this level

Bid side: negative percentages
Ask side: positive percentages

All features use only past data to avoid look-ahead bias.
"""

from typing import List, Optional, Tuple
import polars as pl
import numpy as np


# ============================================================================
# DEPTH IMBALANCE FEATURES
# ============================================================================

def pivot_book_depth(df: pl.DataFrame) -> pl.DataFrame:
    """
    Pivot book depth data from long to wide format.
    
    Input format (long):
    - time, percentage, depth, notional
    
    Output format (wide):
    - time, bid_depth_5, bid_depth_4, ..., ask_depth_1, ..., ask_depth_5
    - And corresponding notional columns
    
    Args:
        df: Book depth DataFrame in long format
        
    Returns:
        Pivoted DataFrame with one row per timestamp
    """
    # Separate bids and asks
    bids = df.filter(pl.col("percentage") < 0)
    asks = df.filter(pl.col("percentage") > 0)
    
    # Pivot bids
    bids_pivot = bids.pivot(
        on="percentage",
        index="time",
        values=["depth", "notional"]
    )
    
    # Rename bid columns
    for col in bids_pivot.columns:
        if col != "time":
            if "depth" in col:
                pct = col.split("_")[-1]
                new_name = f"bid_depth_{abs(int(float(pct)))}"
            elif "notional" in col:
                pct = col.split("_")[-1]
                new_name = f"bid_notional_{abs(int(float(pct)))}"
            else:
                new_name = col
            bids_pivot = bids_pivot.rename({col: new_name})
    
    # Pivot asks
    asks_pivot = asks.pivot(
        on="percentage",
        index="time",
        values=["depth", "notional"]
    )
    
    # Rename ask columns
    for col in asks_pivot.columns:
        if col != "time":
            if "depth" in col:
                pct = col.split("_")[-1]
                new_name = f"ask_depth_{int(float(pct))}"
            elif "notional" in col:
                pct = col.split("_")[-1]
                new_name = f"ask_notional_{int(float(pct))}"
            else:
                new_name = col
            asks_pivot = asks_pivot.rename({col: new_name})
    
    # Join bids and asks
    result = bids_pivot.join(asks_pivot, on="time", how="outer")
    
    return result.sort("time")


def add_depth_imbalance(
    df: pl.DataFrame,
    levels: List[int] = [1, 2, 3, 5]
) -> pl.DataFrame:
    """
    Calculate depth imbalance at various price levels.
    
    Depth Imbalance = (Bid Depth - Ask Depth) / (Bid Depth + Ask Depth)
    
    Interpretation:
    - Positive: More bids than asks (buying pressure)
    - Negative: More asks than bids (selling pressure)
    - Close to ±1: Highly imbalanced (potential price move)
    
    Args:
        df: DataFrame with pivoted book depth data
        levels: Price levels to calculate imbalance for
        
    Returns:
        DataFrame with imbalance features
    """
    result = df.clone()
    
    for level in levels:
        bid_col = f"bid_depth_{level}"
        ask_col = f"ask_depth_{level}"
        
        if bid_col in result.columns and ask_col in result.columns:
            # Depth imbalance
            result = result.with_columns([
                ((pl.col(bid_col) - pl.col(ask_col)) 
                 / (pl.col(bid_col) + pl.col(ask_col) + 1e-10))
                .alias(f"depth_imbalance_{level}")
            ])
            
            # Notional imbalance
            bid_notional = f"bid_notional_{level}"
            ask_notional = f"ask_notional_{level}"
            
            if bid_notional in result.columns and ask_notional in result.columns:
                result = result.with_columns([
                    ((pl.col(bid_notional) - pl.col(ask_notional))
                     / (pl.col(bid_notional) + pl.col(ask_notional) + 1e-10))
                    .alias(f"notional_imbalance_{level}")
                ])
    
    # Cumulative depth imbalance (across all levels)
    available_levels = [l for l in levels if f"bid_depth_{l}" in result.columns]
    
    if len(available_levels) > 0:
        total_bid = sum([pl.col(f"bid_depth_{l}") for l in available_levels])
        total_ask = sum([pl.col(f"ask_depth_{l}") for l in available_levels])
        
        result = result.with_columns([
            ((total_bid - total_ask) / (total_bid + total_ask + 1e-10))
            .alias("total_depth_imbalance")
        ])
    
    return result


def add_depth_features_from_long(
    df: pl.DataFrame,
    windows: List[int] = [6, 12, 24]
) -> pl.DataFrame:
    """
    Add depth features from long-format book depth data.
    
    This function works directly with the raw book depth format
    without requiring pivoting (more memory efficient).
    
    Args:
        df: Book depth DataFrame in long format
        windows: Lookback windows for rolling features
        
    Returns:
        DataFrame with aggregated depth features per timestamp
    """
    # Calculate total bids and asks per timestamp
    bid_agg = (df.filter(pl.col("percentage") < 0)
               .group_by("time")
               .agg([
                   pl.col("depth").sum().alias("total_bid_depth"),
                   pl.col("notional").sum().alias("total_bid_notional"),
                   pl.col("depth").mean().alias("avg_bid_depth"),
               ]))
    
    ask_agg = (df.filter(pl.col("percentage") > 0)
               .group_by("time")
               .agg([
                   pl.col("depth").sum().alias("total_ask_depth"),
                   pl.col("notional").sum().alias("total_ask_notional"),
                   pl.col("depth").mean().alias("avg_ask_depth"),
               ]))
    
    # Join bid and ask aggregations
    result = bid_agg.join(ask_agg, on="time", how="outer").sort("time")
    
    # Fill nulls with 0
    result = result.fill_null(0)
    
    # Depth imbalance (total)
    result = result.with_columns([
        ((pl.col("total_bid_depth") - pl.col("total_ask_depth"))
         / (pl.col("total_bid_depth") + pl.col("total_ask_depth") + 1e-10))
        .alias("depth_imbalance")
    ])
    
    # Notional imbalance
    result = result.with_columns([
        ((pl.col("total_bid_notional") - pl.col("total_ask_notional"))
         / (pl.col("total_bid_notional") + pl.col("total_ask_notional") + 1e-10))
        .alias("notional_imbalance")
    ])
    
    # Total liquidity
    result = result.with_columns([
        (pl.col("total_bid_notional") + pl.col("total_ask_notional"))
        .alias("total_liquidity")
    ])
    
    # Rolling features
    for window in windows:
        # Imbalance MA
        result = result.with_columns([
            pl.col("depth_imbalance").rolling_mean(window)
            .alias(f"depth_imbalance_ma_{window}")
        ])
        
        # Liquidity MA
        result = result.with_columns([
            pl.col("total_liquidity").rolling_mean(window)
            .alias(f"liquidity_ma_{window}")
        ])
        
        # Imbalance volatility
        result = result.with_columns([
            pl.col("depth_imbalance").rolling_std(window)
            .alias(f"depth_imbalance_std_{window}")
        ])
    
    # Imbalance momentum
    result = result.with_columns([
        (pl.col("depth_imbalance") - pl.col("depth_imbalance").shift(1))
        .alias("depth_imbalance_change")
    ])
    
    return result


# ============================================================================
# LIQUIDITY FEATURES
# ============================================================================

def add_liquidity_features(
    df: pl.DataFrame,
    windows: List[int] = [6, 12, 24]
) -> pl.DataFrame:
    """
    Add liquidity-related features.
    
    Features:
    - Spread proxy (from depth levels if available)
    - Liquidity concentration
    - Depth stability
    
    Args:
        df: DataFrame with depth data
        windows: Lookback windows
        
    Returns:
        DataFrame with liquidity features
    """
    result = df.clone()
    
    # Check if we have required columns
    if "total_liquidity" not in result.columns:
        return result
    
    # Liquidity ratio to average
    for window in windows:
        if f"liquidity_ma_{window}" in result.columns:
            result = result.with_columns([
                (pl.col("total_liquidity") / (pl.col(f"liquidity_ma_{window}") + 1e-10))
                .alias(f"liquidity_ratio_{window}")
            ])
    
    # Bid-ask ratio (not spread, but asymmetry)
    if "total_bid_notional" in result.columns and "total_ask_notional" in result.columns:
        result = result.with_columns([
            (pl.col("total_bid_notional") / (pl.col("total_ask_notional") + 1e-10))
            .alias("bid_ask_ratio")
        ])
        
        result = result.with_columns([
            pl.col("bid_ask_ratio").log().alias("bid_ask_ratio_log")
        ])
    
    # Liquidity change
    result = result.with_columns([
        ((pl.col("total_liquidity") - pl.col("total_liquidity").shift(1))
         / (pl.col("total_liquidity").shift(1) + 1e-10) * 100)
        .alias("liquidity_change_pct")
    ])
    
    # Depth concentration: ratio of near levels to far levels
    # Higher = more liquidity near mid price
    if ("bid_depth_1" in result.columns and "bid_depth_5" in result.columns and
        "ask_depth_1" in result.columns and "ask_depth_5" in result.columns):
        
        near_depth = pl.col("bid_depth_1") + pl.col("ask_depth_1")
        far_depth = pl.col("bid_depth_5") + pl.col("ask_depth_5")
        
        result = result.with_columns([
            (near_depth / (far_depth + 1e-10)).alias("depth_concentration")
        ])
    
    return result


# ============================================================================
# COMPOSITE FUNCTION
# ============================================================================

def add_all_orderbook_features(
    df: pl.DataFrame,
    windows: List[int] = [6, 12, 24]
) -> pl.DataFrame:
    """
    Add all orderbook features.
    
    This function handles both long and wide format input.
    
    Args:
        df: Book depth DataFrame
        windows: Lookback windows
        
    Returns:
        DataFrame with all orderbook features
    """
    result = df.clone()
    
    # Check format
    if "percentage" in result.columns:
        # Long format - aggregate first
        result = add_depth_features_from_long(result, windows)
    else:
        # Assume already pivoted or has depth columns
        result = add_depth_imbalance(result, levels=[1, 2, 3, 5])
    
    # Add liquidity features
    result = add_liquidity_features(result, windows)
    
    return result


def prepare_orderbook_for_merge(
    book_df: pl.DataFrame,
    target_times: pl.Series,
    windows: List[int] = [6, 12, 24]
) -> pl.DataFrame:
    """
    Prepare orderbook features for merging with OHLCV data.
    
    This function:
    1. Processes raw book depth data
    2. Calculates all features
    3. Prepares for point-in-time merge
    
    Args:
        book_df: Raw book depth DataFrame
        target_times: Target timestamps to align to
        windows: Lookback windows
        
    Returns:
        DataFrame ready for asof join with OHLCV
    """
    # Calculate features
    features = add_all_orderbook_features(book_df, windows)
    
    return features
