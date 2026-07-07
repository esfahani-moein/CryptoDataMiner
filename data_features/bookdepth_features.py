import numpy as np
import polars as pl

def calculate_obi_features(book_df: pl.DataFrame, levels: list[int]) -> pl.DataFrame:
    """
    Calculates Order Book Imbalance (OBI) for specified levels from depth snapshots.
    Handles timestamp conversion, sorting, and efficient aggregation.
    
    Args:
        book_df: Polars DataFrame with 'timestamp' and 'percentage' columns.
        levels: List of levels (e.g., [1, 3, 5]) to compute OBI for.
    
    Returns:
        Polars DataFrame with OBI columns (e.g., 'OBI_L1', 'OBI_L3', 'OBI_L5').
    """
    # Ensure timestamp is datetime and sorted
    df = book_df.with_columns(
        pl.col("timestamp").str.to_datetime()
    ).sort("timestamp")
    
    # Build aggregation expressions dynamically for each level
    agg_exprs = []
    for n in levels:
        # OBI calculation: (notional at -n - notional at n) / (notional at -n + notional at n)
        obi_expr = (
            (pl.col("notional").filter(pl.col("percentage") == -n).first() - 
             pl.col("notional").filter(pl.col("percentage") == n).first()) / 
            (pl.col("notional").filter(pl.col("percentage") == -n).first() + 
             pl.col("notional").filter(pl.col("percentage") == n).first())
        ).alias(f"OBI_L{n}")
        agg_exprs.append(obi_expr)
    
    # Group by timestamp and aggregate
    obi_df = df.group_by("timestamp").agg(agg_exprs).sort("timestamp")
    
    # Handle potential division by zero (if a snapshot is empty)
    obi_df = obi_df.fill_null(0.0)
    
    return obi_df



def calc_slope_expr(prefix):
    levels = [1, 2, 3, 4, 5]
    sum_x = sum(levels)

    def safe_log_expr(col_name):
        return pl.when(pl.col(col_name) > pl.lit(0.0)) \
                 .then(pl.col(col_name).log()) \
                 .otherwise(pl.lit(np.log(1e-9)))

    sum_xy = sum((pl.lit(i) * safe_log_expr(f"{prefix}_{i}") for i in levels), pl.lit(0))
    sum_y = sum((safe_log_expr(f"{prefix}_{i}") for i in levels), pl.lit(0))

    # Denominator for OLS slope (for x=1..5): 5*sum(x^2)-sum(x)^2 = 50
    return (pl.lit(5) * sum_xy - pl.lit(sum_x) * sum_y) / pl.lit(50)


def calc_gap_expr(prefix):
    # Use expression-level when/then to avoid type inference issues
    return sum(
        (
            pl.when(pl.col(f"{prefix}_{i+1}") > pl.col(f"{prefix}_{i}") * pl.lit(1.5))
              .then(pl.lit(i))
              .otherwise(pl.lit(0))
            for i in [1, 2, 3, 4]
        ),
        pl.lit(0)
    )

def add_book_features(df: pl.DataFrame, ewma_span: int = 20) -> pl.DataFrame:
    """
    Highly optimized Polars implementation of advanced orderbook features:
    1. Logarithmic Depth Slope (OLS on log-notional)
    2. EWMA-weighted Liquidity Statistics
    3. Liquidity Gap (Distance to major volume clusters)
    """
    
    # --- PIVOT DATA ---
    # Ensure timestamp is datetime first
    pivot_input = df.with_columns(pl.col("timestamp").str.to_datetime())
    # Convert from long format to wide format for vectorization
    pivoted = pivot_input.pivot(
        index="timestamp",
        on="percentage",
        values="notional"
    ).sort("timestamp")
    
    # Rename columns for easier access (e.g., "-1" -> "bid_1", "1" -> "ask_1")
    # Note: Using absolute values for percentage logic
    mapping = {str(i): f"bid_{abs(i)}" if i < 0 else f"ask_{i}" for i in [-5,-4,-3,-2,-1,1,2,3,4,5]}
    pivoted = pivoted.rename({old: new for old, new in mapping.items() if old in pivoted.columns})

    
    # helper to sum columns safely as an Expr
    def sum_cols(prefix, levels=[1,2,3,4,5]):
        return sum((pl.coalesce([pl.col(f"{prefix}_{i}"), pl.lit(0.0)]) for i in levels), pl.lit(0.0))


    # --- APPLY ALL EXPRESSIONS ---
    result = pivoted.with_columns([
        # Slopes
        calc_slope_expr("bid").alias("bid_slope_log"),
        calc_slope_expr("ask").alias("ask_slope_log"),
        
        # Liquidity Gaps
        calc_gap_expr("bid").alias("bid_gap_score"),
        calc_gap_expr("ask").alias("ask_gap_score"),
        
        # Total Notional
        sum_cols("bid").alias("total_bid_notional"),
        sum_cols("ask").alias("total_ask_notional"),
    ])

    # --- EWMA Z-SCORES ---
    # Improvement: Using EWMA instead of simple rolling mean for better decay weighting
    result = result.with_columns([
        (
        (pl.col("total_bid_notional") - pl.col("total_bid_notional").ewm_mean(span=ewma_span))
        / pl.coalesce([pl.col("total_bid_notional").rolling_std(window_size=ewma_span), pl.lit(1e-9)])
        ).alias("bid_notional_z_score"),

        (
        (pl.col("total_ask_notional") - pl.col("total_ask_notional").ewm_mean(span=ewma_span))
        / pl.coalesce([pl.col("total_ask_notional").rolling_std(window_size=ewma_span), pl.lit(1e-9)])
        ).alias("ask_notional_z_score")
    ])

    return result