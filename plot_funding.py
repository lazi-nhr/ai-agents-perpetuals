"""
Funding rate plotter for Hyperliquid perps (single file or whole folder).

Input  (per coin Parquet made by your downloader):
  index: DatetimeIndex (UTC, hourly)
  columns: coin, fundingRate (float), premium (float)

Output:
  - Single-file mode: open HTML in browser or save a static image.
  - Directory mode: write charts under <dir>/plots.

Usage examples are at the bottom of this file.
"""

import os
import re
import argparse
from glob import glob
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------- time/index helpers ----------------
def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a sorted, tz-naive UTC DatetimeIndex named 'time'."""
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        idx = out.index
    elif "time" in out.columns:
        idx = pd.to_datetime(out["time"], errors="coerce", utc=True)
    else:
        idx = pd.to_datetime(out.index, errors="coerce", utc=True)

    out.index = idx
    out = out[~out.index.isna()]
    if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    out.index.name = "time"
    out.sort_index(inplace=True)
    return out


# ---------------- title/filename helpers ----------------
# Matches: BTC_funding_2025-03-22_2025-08-22.parquet  (asset name may have underscores)
_TITLE_RX = re.compile(r"^(?P<asset>.+?)_funding_(?P<start>\d{4}-\d{2}-\d{2})_(?P<end>\d{4}-\d{2}-\d{2})$", re.IGNORECASE)

def parse_title_from_path(file_path: str) -> str:
    base = os.path.splitext(os.path.basename(file_path))[0]
    m = _TITLE_RX.match(base)
    if m:
        asset = m.group("asset")
        start = m.group("start")
        end = m.group("end")
        return f"Perp: {asset}  •  Funding window: {start} → {end}"
    return os.path.basename(file_path)

def sanitize_filename(name: str) -> str:
    return (name.replace("/", "_").replace("\\", "_").replace(":", "_")
                .replace("*", "_").replace("?", "_").replace('"', "_")
                .replace("<", "_").replace(">", "_").replace("|", "_"))


# ---------------- core plotting ----------------
def create_funding_figure(
    df: pd.DataFrame,
    file_path: str,
    show_apr: bool = True,
    roll_hours: Optional[int] = None,
) -> go.Figure:
    """
    Build a 2-row figure:
      row1: fundingRate (fraction per hour), optional APR (%) on secondary y-axis
      row2: premium (fraction)
    """
    df = ensure_datetime_index(df)

    # Basic column checks and numeric coercion
    for col in ("fundingRate", "premium"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["fundingRate", "premium"])

    # Optional rolling mean (hours)
    if roll_hours and roll_hours > 1:
        df["fundingRate_roll"] = df["fundingRate"].rolling(roll_hours, min_periods=1).mean()
        df["premium_roll"] = df["premium"].rolling(roll_hours, min_periods=1).mean()
    else:
        df["fundingRate_roll"] = None
        df["premium_roll"] = None

    # Secondary trace: annualized APR (%) if requested.
    # Approx: hourly fraction * 24 * 365 * 100
    if show_apr:
        df["apr_pct"] = df["fundingRate"] * 24 * 365 * 100
        if df["fundingRate_roll"].isna().all():
            df["apr_roll_pct"] = None
        else:
            df["apr_roll_pct"] = df["fundingRate_roll"] * 24 * 365 * 100

    title_text = parse_title_from_path(file_path)

    # Use secondary y-axis only on the top subplot
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        subplot_titles=("Funding rate (per hour)", "Premium"),
        specs=[[{"secondary_y": show_apr}], [{"secondary_y": False}]],
        row_heights=[0.6, 0.4]
    )

    # x values as ISO strings (stable for static export)
    x_iso = df.index.astype("datetime64[ms]").strftime("%Y-%m-%d %H:%M:%S").tolist()

    # --- Row 1: fundingRate ---
    fig.add_trace(
        go.Scatter(x=x_iso, y=df["fundingRate"], mode="lines", name="fundingRate"),
        row=1, col=1, secondary_y=False
    )
    if df["fundingRate_roll"] is not None and not df["fundingRate_roll"].isna().all():
        fig.add_trace(
            go.Scatter(x=x_iso, y=df["fundingRate_roll"], mode="lines",
                       name=f"fundingRate {roll_hours}h MA", line=dict(width=2, dash="dot")),
            row=1, col=1, secondary_y=False
        )
    # zero line
    fig.add_hline(y=0.0, line_width=1, line_dash="dot", row=1, col=1)

    # APR on secondary axis
    if show_apr:
        fig.add_trace(
            go.Scatter(x=x_iso, y=df["apr_pct"], mode="lines", name="APR (%)"),
            row=1, col=1, secondary_y=True
        )
        if df.get("apr_roll_pct") is not None and not df["apr_roll_pct"].isna().all():
            fig.add_trace(
                go.Scatter(x=x_iso, y=df["apr_roll_pct"], mode="lines",
                           name=f"APR (%) {roll_hours}h MA", line=dict(width=2, dash="dash")),
                row=1, col=1, secondary_y=True
            )

    # --- Row 2: premium ---
    fig.add_trace(
        go.Scatter(x=x_iso, y=df["premium"], mode="lines", name="premium"),
        row=2, col=1
    )
    if df["premium_roll"] is not None and not df["premium_roll"].isna().all():
        fig.add_trace(
            go.Scatter(x=x_iso, y=df["premium_roll"], mode="lines",
                       name=f"premium {roll_hours}h MA", line=dict(width=2, dash="dot")),
            row=2, col=1
        )
    fig.add_hline(y=0.0, line_width=1, line_dash="dot", row=2, col=1)

    # Layout polish
    fig.update_layout(
        title_text=title_text,
        title_x=0.5,
        hovermode="x unified",
        showlegend=True,
        xaxis_title=None,                 # no redundant "Time" under the top subplot
        xaxis_rangeslider_visible=False
    )
    fig.update_xaxes(matches="x", type="date")
    fig.update_yaxes(title_text="Funding (fraction / hour)", row=1, col=1, secondary_y=False)
    if show_apr:
        fig.update_yaxes(title_text="APR (%)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Premium (fraction)", row=2, col=1)

    return fig


# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot Hyperliquid perp funding rate history from Parquet."
    )
    parser.add_argument(
        "path", type=str,
        help="Path to a single funding Parquet, or to a folder of Parquet files."
    )
    parser.add_argument(
        "--format", type=str, default="html",
        choices=["html", "png", "jpg", "svg"],
        help="Output format (default: html)."
    )
    parser.add_argument(
        "--cdn", action="store_true",
        help="Use CDN for plotly.js when writing HTML (smaller files)."
    )
    parser.add_argument(
        "--no-apr", action="store_true",
        help="Do not plot APR on secondary axis."
    )
    parser.add_argument(
        "--roll", type=int, default=0,
        help="Rolling mean window in hours (e.g., 24). 0 disables it."
    )
    args = parser.parse_args()

    show_apr = not args.no_apr
    roll_hours = args.roll if args.roll and args.roll > 1 else None

    if not os.path.exists(args.path):
        print(f"Error: path does not exist -> {args.path}")
        raise SystemExit(1)

    # Directory mode
    if os.path.isdir(args.path):
        input_directory = args.path
        output_directory = os.path.join(input_directory, "plots")
        os.makedirs(output_directory, exist_ok=True)

        print(f"Scanning directory: {input_directory}")
        files = glob(os.path.join(input_directory, "*.parquet"))
        if not files:
            print("No .parquet files found.")
            raise SystemExit(0)

        print(f"Found {len(files)} files.")
        for file_path in files:
            try:
                df = pd.read_parquet(file_path)
                if df.empty:
                    print(f"  [skip] Empty file: {file_path}")
                    continue

                fig = create_funding_figure(df, file_path, show_apr=show_apr, roll_hours=roll_hours)

                base = os.path.splitext(os.path.basename(file_path))[0]
                out_name = f"{sanitize_filename(base)}.{args.format}"
                out_path = os.path.join(output_directory, out_name)
                print(f"  -> Saving: {out_path}")
                if args.format == "html":
                    if args.cdn:
                        fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
                    else:
                        fig.write_html(out_path)
                else:
                    try:
                        # Needs: pip install kaleido   (or a working choreographer setup)
                        fig.write_image(out_path, width=1600, height=900, scale=2)
                    except Exception as e:
                        # Fallback to HTML on static export failure
                        html_path = os.path.splitext(out_path)[0] + ".html"
                        print(f"  [warn] Static export failed: {e}\n          Falling back to {html_path}")
                        fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)

                del fig
            except Exception as e:
                print(f"  [error] {os.path.basename(file_path)}: {e}")

    # Single-file mode
    else:
        try:
            df = pd.read_parquet(args.path)
            if df.empty:
                print("Error: the file is empty.")
                raise SystemExit(1)

            fig = create_funding_figure(df, args.path, show_apr=show_apr, roll_hours=roll_hours)

            if args.format == "html":
                print("Opening interactive chart...")
                fig.show()
            else:
                base, _ = os.path.splitext(args.path)
                out_path = f"{base}.{args.format}"
                try:
                    print(f"Saving static image to: {out_path}")
                    fig.write_image(out_path, width=1600, height=900, scale=2)
                    print("Saved.")
                except Exception as e:
                    print(f"[warn] Static export failed: {e}\n       Falling back to HTML.")
                    fig.write_html(f"{base}.html", include_plotlyjs="cdn", full_html=True)

        except Exception as e:
            print(f"An error occurred: {e}")

# ---------------- Examples ----------------
# 1) Plot one coin interactively:
#    python plot_funding.py ./hyperliquid_funding/BTC_funding_2025-03-22_2025-08-22.parquet
#
# 2) Plot a whole folder to HTML:
#    python plot_funding.py ./hyperliquid_funding --format html --cdn
#
# 3) Plot with annualized APR and 24-hour rolling mean, save PNG:
#    python plot_funding.py ./hyperliquid_funding/BTC_funding_2025-03-22_2025-08-22.parquet --format png --roll 24
#
# 4) Batch plot a folder, add rolling mean, fallback to HTML on static export errors:
#    python plot_funding.py ./hyperliquid_funding --format png --roll 24
