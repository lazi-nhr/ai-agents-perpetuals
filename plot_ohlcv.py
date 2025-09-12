"""
OHLCV Plotter for Parquet

This CLI tool renders candlestick + volume charts from OHLCV Parquet files.
It works on a single file or an entire folder. The chart uses:
- a shared time axis for candlesticks and volume,
- candlesticks only on buckets with trades (volume > 0),
- a thin close-line across all buckets for continuity,
- a robust time index fix so dates render correctly in both HTML and static images.
"""

import os
import re
import argparse
from glob import glob

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -------- Time index helpers -------------------------------------------------
def _ints_to_datetime(values: pd.Series | pd.Index) -> pd.DatetimeIndex:
    """Convert integer epoch timestamps to timezone-naive UTC datetimes.
    Unit is guessed from magnitude: ns > 1e14, ms > 1e11, else seconds."""
    s = pd.Series(values)
    v = s.dropna()
    if v.empty:
        return pd.to_datetime(values, errors="coerce", utc=True)
    m = int(v.iloc[0])
    if m > 10**14:
        unit = "ns"
    elif m > 10**11:
        unit = "ms"
    else:
        unit = "s"
    return pd.to_datetime(values, unit=unit, errors="coerce", utc=True)


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose index is a tz-naive UTC DatetimeIndex, sorted."""
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        idx = out.index
    else:
        if "time" in out.columns:
            idx = pd.to_datetime(out["time"], errors="coerce", utc=True)
        elif "index" in out.columns and pd.api.types.is_integer_dtype(out["index"].dtype):
            idx = _ints_to_datetime(out["index"])
        elif pd.api.types.is_integer_dtype(getattr(out.index, "dtype", None)):
            idx = _ints_to_datetime(out.index)
        else:
            idx = pd.to_datetime(out.index, errors="coerce", utc=True)

    out.index = idx
    out = out[~out.index.isna()]
    if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    out.sort_index(inplace=True)
    return out


# -------- Filename parser (asset may contain underscores) --------------------
_TF_RX = re.compile(r"^(?P<asset>.+)_(?P<tf>1m|5m|15m|1h|1d)_ohlcv$", re.IGNORECASE)


def parse_title_from_path(file_path: str) -> str:
    """Build a readable chart title from file name."""
    base = os.path.splitext(os.path.basename(file_path))[0]
    m = _TF_RX.match(base)
    if m:
        asset_name = m.group("asset")
        timeframe = m.group("tf")
        return f"Asset: {asset_name}  •  Timeframe: {timeframe}"
    return os.path.basename(file_path)


# -------- Chart builder ------------------------------------------------------
def create_ohlcv_figure(df: pd.DataFrame, file_path: str) -> go.Figure:
    """Create a candlestick + volume figure from an OHLCV DataFrame."""
    # Fix and sort the time index so Plotly gets true dates.
    df = ensure_datetime_index(df)

    # Basic column checks and numeric coercion.
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])

    # Split into "all buckets" vs "buckets with trades".
    df_trade = df[df["volume"] > 0]

    # Use ISO strings to milliseconds for a robust date axis (stable with kaleido).
    x_all = df.index.astype("datetime64[ms]").strftime("%Y-%m-%d %H:%M:%S").tolist()
    x_trd = df_trade.index.astype("datetime64[ms]").strftime("%Y-%m-%d %H:%M:%S").tolist()

    title_text = parse_title_from_path(file_path)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("OHLC Candlestick", "Volume"),
        row_heights=[0.7, 0.3]
    )

    # Candlesticks only on buckets that have trades.
    # This removes long dashed flats from O=H=L=C carry-forward buckets.
    if not df_trade.empty:
        fig.add_trace(
            go.Candlestick(
                x=x_trd,
                open=df_trade["open"],
                high=df_trade["high"],
                low=df_trade["low"],
                close=df_trade["close"],
                name="OHLC"
            ),
            row=1, col=1
        )
    """
    # A thin close-line over all buckets for context (optional but helpful).
    fig.add_trace(
        go.Scatter(
            x=x_all,
            y=df["close"],
            mode="lines",
            name="Close",
            line=dict(width=1),
            connectgaps=False
        ),
        row=1, col=1
    )
    """
    # Volume bars (0-height bars are fine; they keep the axis aligned).
    fig.add_trace(
        go.Bar(
            x=x_all,
            y=df["volume"],
            name="Volume",
            marker_color="rgba(0,100,255,0.5)"
        ),
        row=2, col=1
    )

    # Layout polish.
    fig.update_layout(
        title_text=title_text,
        title_x=0.5,
        xaxis_title=None,                 # omit redundant "Time" label
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        showlegend=False
    )
    fig.update_xaxes(matches="x", type="date")
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


# -------- CLI ----------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render OHLCV charts from Parquet files (single file or directory).",
        epilog="Charts show candlesticks on trade buckets and a close-line for continuity."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to a .parquet file or to a directory that contains .parquet files."
    )
    parser.add_argument(
        "--format",
        type=str,
        default="html",
        choices=["html", "png", "jpg", "svg"],
        help="Output format (default: html)."
    )
    parser.add_argument(
        "--cdn",
        action="store_true",
        help="Use CDN for plotly.js when writing HTML (smaller files)."
    )
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"Error: path does not exist -> {args.path}")
        raise SystemExit(1)

    # Directory mode: render every .parquet and write to <dir>/plots.
    if os.path.isdir(args.path):
        input_directory = args.path
        output_directory = os.path.join(input_directory, "plots")
        os.makedirs(output_directory, exist_ok=True)

        print(f"Scanning directory: {input_directory}")
        print(f"Saving .{args.format} files to: {output_directory}")

        parquet_files = glob(os.path.join(input_directory, "*.parquet"))
        if not parquet_files:
            print("No .parquet files found in this directory.")
        else:
            print(f"Found {len(parquet_files)} files to plot.")
            for file_path in parquet_files:
                try:
                    df = pd.read_parquet(file_path)
                    if df.empty:
                        print(f"Warning: empty file, skipping -> {file_path}")
                        continue

                    fig = create_ohlcv_figure(df, file_path)

                    base = os.path.splitext(os.path.basename(file_path))[0]
                    output_filename = f"{base}.{args.format}"
                    output_path = os.path.join(output_directory, output_filename)

                    print(f"  -> Saving chart to: {output_path}")
                    if args.format == "html":
                        if args.cdn:
                            fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
                        else:
                            fig.write_html(output_path)
                    else:
                        # Requires: pip install kaleido
                        fig.write_image(output_path, width=1600, height=900, scale=2)

                    del fig  # free memory in batch mode

                except Exception as e:
                    print(f"  -> Error processing {os.path.basename(file_path)}: {e}")

    # Single-file mode: show in browser (html) or save static image.
    else:
        try:
            df = pd.read_parquet(args.path)
            if df.empty:
                print("Error: the file is empty and cannot be plotted.")
                raise SystemExit(1)

            fig = create_ohlcv_figure(df, args.path)

            if args.format == "html":
                print("Opening interactive chart...")
                fig.show()
            else:
                base, _ = os.path.splitext(args.path)
                output_path = f"{base}.{args.format}"
                try:
                    print(f"Saving static image to: {output_path}")
                    fig.write_image(output_path, width=1600, height=900, scale=2)
                    print("Saved.")
                except Exception as e:
                    print(f"Error saving image: {e}")
                    print("Make sure kaleido is installed: pip install kaleido")

        except Exception as e:
            print(f"An error occurred: {e}")


# ------------------------------- Usage ---------------------------------------
# 1) Directory -> write HTML files:
#    python plot_ohlcv.py ./hyperliquid_data/processed_ohlcv_1d/spot
#
# 2) Directory -> write PNG files:
#    python plot_ohlcv.py ./hyperliquid_data/processed_ohlcv_1d/perp --format png
#
# 3) Single file -> open in browser:
#    python plot_ohlcv.py ./hyperliquid_data/processed_ohlcv_1d/perp/BTC_1d_ohlcv.parquet
#
# 4) Single file -> save SVG:
#    python plot_ohlcv.py ./hyperliquid_data/processed_ohlcv_1h/BTC_1h_ohlcv.parquet --format svg
