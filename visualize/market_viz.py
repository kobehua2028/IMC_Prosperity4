#!/usr/bin/env python3
"""
Market data visualization tool for IMC Prosperity.

Reads all prices_*.csv and trades_*.csv files from a round folder and generates
a self-contained interactive HTML file with synchronized price + volume charts.

Usage:
  python market_viz.py <folder>              # writes viz.html in current dir
  python market_viz.py <folder> output.html  # custom output path

Examples:
  python market_viz.py ../data/ROUND_1/ROUND1
  python market_viz.py ../data/TUTORIAL_ROUND_1 tutorial.html
"""

import csv
import sys
import os
import json
import glob
import re
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def read_prices(folder: Path) -> dict:
    """
    Returns: { product: [ {day, timestamp, bid_price_1, bid_volume_1,
                            ask_price_1, ask_volume_1, mid_price}, ... ] }
    """
    products = defaultdict(list)
    files = sorted(glob.glob(str(folder / "prices_*.csv")))
    if not files:
        print(f"  [warn] no prices_*.csv found in {folder}", file=sys.stderr)

    for fpath in files:
        with open(fpath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                try:
                    product = row["product"].strip()
                    day     = int(row["day"])
                    ts      = int(row["timestamp"])

                    def _f(key):
                        v = row.get(key, "").strip()
                        return float(v) if v else None

                    def _i(key):
                        v = row.get(key, "").strip()
                        return int(v) if v else 0

                    products[product].append({
                        "day":          day,
                        "timestamp":    ts,
                        "bid_price_1":  _f("bid_price_1"),
                        "bid_volume_1": _i("bid_volume_1"),
                        "bid_price_2":  _f("bid_price_2"),
                        "bid_volume_2": _i("bid_volume_2"),
                        "ask_price_1":  _f("ask_price_1"),
                        "ask_volume_1": _i("ask_volume_1"),
                        "ask_price_2":  _f("ask_price_2"),
                        "ask_volume_2": _i("ask_volume_2"),
                        "mid_price":    _f("mid_price"),
                    })
                except (ValueError, KeyError):
                    continue

    return dict(products)


def read_trades(folder: Path) -> dict:
    """
    Returns: { symbol: [ {day, timestamp, price, quantity, buyer, seller}, ... ] }
    Day is extracted from the filename since trades CSVs omit it.
    """
    symbols = defaultdict(list)
    files = sorted(glob.glob(str(folder / "trades_*.csv")))
    if not files:
        print(f"  [warn] no trades_*.csv found in {folder}", file=sys.stderr)

    for fpath in files:
        match = re.search(r"day_(-?\d+)\.csv$", fpath)
        file_day = int(match.group(1)) if match else 0

        with open(fpath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                try:
                    symbol   = row["symbol"].strip()
                    ts       = int(row["timestamp"])
                    price    = float(row["price"])
                    quantity = int(row["quantity"])
                    symbols[symbol].append({
                        "day":       file_day,
                        "timestamp": ts,
                        "price":     price,
                        "quantity":  quantity,
                        "buyer":     row.get("buyer", "").strip(),
                        "seller":    row.get("seller", "").strip(),
                    })
                except (ValueError, KeyError):
                    continue

    return dict(symbols)


# ---------------------------------------------------------------------------
# Data processing
# ---------------------------------------------------------------------------

def global_ts(day: int, timestamp: int, day_offset: int) -> int:
    """Flatten day+timestamp onto a single x axis (1 day = 1 000 000 units)."""
    return (day - day_offset) * 1_000_000 + timestamp


def build_payload(prices: dict, trades: dict, folder_name: str) -> dict:
    all_days: set = set()
    for recs in prices.values():
        for r in recs:
            all_days.add(r["day"])
    for recs in trades.values():
        for r in recs:
            all_days.add(r["day"])

    day_offset   = min(all_days) if all_days else 0
    sorted_days  = sorted(all_days)
    all_products = sorted(set(prices.keys()) | set(trades.keys()))

    # day -> global x of its first timestamp
    day_boundaries = {d: global_ts(d, 0, day_offset) for d in sorted_days}

    product_data = {}
    for product in all_products:
        precs = sorted(
            prices.get(product, []), key=lambda r: (r["day"], r["timestamp"])
        )
        trecs = sorted(
            trades.get(product, []), key=lambda r: (r["day"], r["timestamp"])
        )

        xs        = [global_ts(r["day"], r["timestamp"], day_offset) for r in precs]
        mid       = [r["mid_price"]    for r in precs]
        bid1      = [r["bid_price_1"]  for r in precs]
        ask1      = [r["ask_price_1"]  for r in precs]
        bid2      = [r["bid_price_2"]  for r in precs]
        ask2      = [r["ask_price_2"]  for r in precs]
        bid_vol1  = [r["bid_volume_1"] for r in precs]
        ask_vol1  = [r["ask_volume_1"] for r in precs]
        bid_vol2  = [r["bid_volume_2"] for r in precs]
        ask_vol2  = [r["ask_volume_2"] for r in precs]

        txs    = [global_ts(r["day"], r["timestamp"], day_offset) for r in trecs]
        tprice = [r["price"]    for r in trecs]
        tqty   = [r["quantity"] for r in trecs]

        product_data[product] = {
            "xs": xs, "mid": mid,
            "bid1": bid1, "ask1": ask1,
            "bid2": bid2, "ask2": ask2,
            "bid_vol1": bid_vol1, "ask_vol1": ask_vol1,
            "bid_vol2": bid_vol2, "ask_vol2": ask_vol2,
            "txs": txs, "tprice": tprice, "tqty": tqty,
        }

    return {
        "folder":         folder_name,
        "products":       all_products,
        "product_data":   product_data,
        "day_boundaries": day_boundaries,
        "sorted_days":    sorted_days,
    }


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Market Visualizer — {folder}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body   { font-family: system-ui, -apple-system, sans-serif; background: #0d1117; color: #c9d1d9; }
  header { padding: 14px 24px; background: #161b22; border-bottom: 1px solid #21262d;
           display: flex; align-items: center; gap: 14px; }
  header h1 { font-size: 1.1rem; font-weight: 600; color: #e6edf3; }
  header .sub { color: #6e7681; font-size: 0.85rem; }
  .controls { padding: 10px 24px; background: #0d1117; border-bottom: 1px solid #21262d;
              display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .controls label { font-size: 0.8rem; color: #6e7681; }
  select  { background: #161b22; color: #c9d1d9; border: 1px solid #30363d;
            border-radius: 6px; padding: 5px 10px; font-size: 0.85rem; cursor: pointer; }
  select:focus { outline: 2px solid #388bfd; outline-offset: 1px; }
  .btn-group { display: flex; gap: 5px; }
  button  { background: #21262d; color: #8b949e; border: 1px solid #30363d;
            border-radius: 6px; padding: 4px 10px; font-size: 0.78rem; cursor: pointer;
            transition: background 0.12s, color 0.12s; }
  button:hover   { background: #30363d; color: #c9d1d9; }
  button.active  { background: #1f6feb; color: #fff; border-color: #1f6feb; }
  .sep    { flex: 1; }
  .charts { padding: 14px 24px; display: flex; flex-direction: column; gap: 10px; }
  .card   { background: #161b22; border: 1px solid #21262d; border-radius: 8px; overflow: hidden; }
  .card-title { padding: 8px 14px; font-size: 0.72rem; color: #6e7681;
                border-bottom: 1px solid #21262d; text-transform: uppercase;
                letter-spacing: 0.06em; }
  .no-data { padding: 36px; text-align: center; color: #484f58; font-size: 0.85rem; }
  .stats  { padding: 8px 14px; display: flex; gap: 20px; flex-wrap: wrap;
            font-size: 0.78rem; color: #6e7681; border-top: 1px solid #21262d; }
  .stats span b { color: #c9d1d9; }
</style>
</head>
<body>
<header>
  <h1>Market Visualizer</h1>
  <span class="sub" id="folder-label"></span>
</header>

<div class="controls">
  <label for="prod-sel">Product</label>
  <select id="prod-sel"></select>
  <div class="btn-group" id="day-btns"></div>
  <div class="sep"></div>
  <button id="reset-btn">Reset Zoom</button>
</div>

<div class="charts">
  <div class="card">
    <div class="card-title">Price — Mid, Best Bid &amp; Ask (L1 + L2) with Trades</div>
    <div id="price-chart"></div>
    <div class="stats" id="price-stats"></div>
  </div>
  <div class="card">
    <div class="card-title">Trade Volume &amp; Price</div>
    <div id="vol-chart"></div>
    <div class="stats" id="vol-stats"></div>
  </div>
  <div class="card">
    <div class="card-title">Order Book Depth — Bid &amp; Ask Volume at L1 and L2</div>
    <div id="depth-chart"></div>
  </div>
</div>

<script>
const DATA = {data_json};

/* ---- layout helpers ---- */
const DARK = {
  paper_bgcolor: '#161b22',
  plot_bgcolor:  '#0d1117',
  font:   { color: '#6e7681', size: 11 },
  legend: { bgcolor: 'rgba(22,27,34,0.9)', bordercolor: '#21262d', borderwidth: 1,
            font: { size: 10 }, orientation: 'h', y: -0.18 },
  hovermode: 'x unified',
  hoverlabel: { bgcolor: '#161b22', bordercolor: '#30363d', font: { size: 11 } },
  margin: { t: 8, b: 50, l: 64, r: 16 },
};

const AXIS_BASE = {
  gridcolor:      '#21262d',
  zerolinecolor:  '#30363d',
  tickfont:       { size: 10, color: '#484f58' },
  showspikes:     true,
  spikecolor:     '#484f58',
  spikethickness: 1,
  spikemode:      'across',
};

const CFG = {
  responsive: true,
  displayModeBar: true,
  modeBarButtonsToRemove: ['toImage', 'sendDataToCloud', 'lasso2d', 'select2d'],
  displaylogo: false,
};

/* ---- state ---- */
let product      = null;
let dayFilter    = null;   // null → all days
let xRange       = null;   // shared zoom state [min, max]
let syncLock     = false;  // prevents relayout ping-pong

/* ---- helpers ---- */
function dayLabel(d) { return `Day ${d >= 0 ? '+' + d : d}`; }

function getDayShapes() {
  return Object.entries(DATA.day_boundaries).map(([d, x]) => ({
    type: 'line', xref: 'x', yref: 'paper',
    x0: x, x1: x, y0: 0, y1: 1,
    line: { color: '#30363d', width: 1, dash: 'dot' },
  }));
}

function getDayAnnotations() {
  return Object.entries(DATA.day_boundaries).map(([d, x]) => ({
    xref: 'x', yref: 'paper',
    x: x, y: 1.01, text: dayLabel(parseInt(d)),
    showarrow: false, font: { size: 9, color: '#484f58' },
    xanchor: 'left',
  }));
}

function tickConfig() {
  const vals = Object.values(DATA.day_boundaries);
  const text = Object.keys(DATA.day_boundaries).map(d => dayLabel(parseInt(d)));
  return { tickvals: vals, ticktext: text };
}

/* Filter arrays to the selected day (or return all if dayFilter === null). */
function filterToDay(...arrays) {
  if (dayFilter === null) return arrays.map(a => [...a]);
  const db = DATA.day_boundaries;
  const days = DATA.sorted_days;
  const start = db[dayFilter];
  const nextIdx = days.indexOf(dayFilter) + 1;
  const end = nextIdx < days.length ? db[days[nextIdx]] : Infinity;

  // xs is always the first array
  const xs = arrays[0];
  const keep = [];
  for (let i = 0; i < xs.length; i++) {
    if (xs[i] >= start && xs[i] < end) keep.push(i);
  }
  return arrays.map(arr => keep.map(i => arr[i]));
}

/* ---- main render ---- */
function render() {
  const pd = DATA.product_data[product];
  if (!pd) return;

  const shapes      = getDayShapes();
  const annotations = getDayAnnotations();
  const ticks       = tickConfig();

  const xaxisCommon = {
    ...AXIS_BASE,
    ...ticks,
    range:    xRange || undefined,
    title: { text: 'Global Timestamp (day × 1 000 000 + t)', font: { size: 9 } },
  };

  // ---- 1. Price chart ----
  const [pxs, mid, bid1, ask1, bid2, ask2, txsP, tpriceP, tqtyP] = filterToDay(
    pd.xs, pd.mid, pd.bid1, pd.ask1, pd.bid2, pd.ask2,
    pd.txs, pd.tprice, pd.tqty,
  );

  const priceTraces = [
    // L2 bid/ask fill first (bottom layer)
    {
      x: pxs, y: bid2,
      name: 'Bid L2', legendgroup: 'bid2',
      type: 'scatter', mode: 'lines',
      line: { color: 'rgba(63,185,80,0.3)', width: 0.8, dash: 'dot' },
      hovertemplate: '%{y:.2f}<extra>Bid L2</extra>',
    },
    {
      x: pxs, y: ask2,
      name: 'Ask L2', legendgroup: 'ask2',
      type: 'scatter', mode: 'lines',
      line: { color: 'rgba(248,81,73,0.3)', width: 0.8, dash: 'dot' },
      hovertemplate: '%{y:.2f}<extra>Ask L2</extra>',
    },
    // L1 bid/ask
    {
      x: pxs, y: bid1,
      name: 'Bid L1',
      type: 'scatter', mode: 'lines',
      line: { color: '#3fb950', width: 1, dash: 'dash' },
      hovertemplate: '%{y:.2f}<extra>Bid L1</extra>',
    },
    {
      x: pxs, y: ask1,
      name: 'Ask L1',
      type: 'scatter', mode: 'lines',
      line: { color: '#f85149', width: 1, dash: 'dash' },
      hovertemplate: '%{y:.2f}<extra>Ask L1</extra>',
    },
    // Mid price (top)
    {
      x: pxs, y: mid,
      name: 'Mid Price',
      type: 'scatter', mode: 'lines',
      line: { color: '#58a6ff', width: 2 },
      hovertemplate: '%{y:.2f}<extra>Mid</extra>',
    },
  ];

  // Trade dots overlaid on price chart
  if (txsP.length > 0) {
    priceTraces.push({
      x: txsP, y: tpriceP,
      name: 'Trades',
      type: 'scatter', mode: 'markers',
      marker: { color: '#e3b341', size: 6, symbol: 'circle-open',
                line: { width: 1.8, color: '#e3b341' } },
      customdata: tqtyP,
      hovertemplate: 'Price: %{y:.2f}<br>Qty: %{customdata}<extra>Trade</extra>',
    });
  }

  // Stats
  const midVals = mid.filter(v => v != null);
  if (midVals.length > 0) {
    const lo = Math.min(...midVals), hi = Math.max(...midVals);
    const spread = ask1.map((a, i) => a != null && bid1[i] != null ? a - bid1[i] : null).filter(v => v != null);
    const avgSpread = spread.length ? (spread.reduce((a, b) => a + b, 0) / spread.length).toFixed(2) : 'n/a';
    document.getElementById('price-stats').innerHTML =
      `<span>Low <b>${lo.toFixed(2)}</b></span>` +
      `<span>High <b>${hi.toFixed(2)}</b></span>` +
      `<span>Range <b>${(hi - lo).toFixed(2)}</b></span>` +
      `<span>Avg Spread (L1) <b>${avgSpread}</b></span>` +
      `<span>Price Points <b>${midVals.length.toLocaleString()}</b></span>`;
  }

  Plotly.react('price-chart', priceTraces, {
    ...DARK, height: 340, shapes, annotations,
    xaxis: xaxisCommon,
    yaxis: { ...AXIS_BASE, title: { text: 'Price', font: { size: 10 } } },
  }, CFG);

  // ---- 2. Volume chart ----
  const [tvxs, tvprice, tvqty] = filterToDay(pd.txs, pd.tprice, pd.tqty);

  if (tvxs.length === 0) {
    document.getElementById('vol-chart').innerHTML =
      '<div class="no-data">No trade data for this product / selection</div>';
    document.getElementById('vol-stats').innerHTML = '';
  } else {
    // Cumulative volume line
    const cumVol = [];
    let running = 0;
    for (const q of tvqty) { running += q; cumVol.push(running); }

    const volTraces = [
      {
        x: tvxs, y: tvqty,
        name: 'Trade Volume',
        type: 'bar',
        marker: { color: '#e3b341', opacity: 0.65 },
        customdata: tvprice,
        hovertemplate: 'Qty: %{y}<br>Price: %{customdata:.2f}<extra>Volume</extra>',
        yaxis: 'y',
      },
      {
        x: tvxs, y: cumVol,
        name: 'Cumulative Vol',
        type: 'scatter', mode: 'lines',
        line: { color: '#d2a8ff', width: 1.5 },
        hovertemplate: 'Cum: %{y}<extra>Cumulative</extra>',
        yaxis: 'y2',
      },
    ];

    const totalVol = tvqty.reduce((a, b) => a + b, 0);
    const avgPrice = tvprice.length
      ? (tvprice.reduce((a, b) => a + b, 0) / tvprice.length).toFixed(2) : 'n/a';
    const vwap = tvqty.reduce((acc, q, i) => acc + q * tvprice[i], 0) / (totalVol || 1);

    document.getElementById('vol-stats').innerHTML =
      `<span>Trades <b>${tvxs.length.toLocaleString()}</b></span>` +
      `<span>Total Volume <b>${totalVol.toLocaleString()}</b></span>` +
      `<span>Avg Price <b>${avgPrice}</b></span>` +
      `<span>VWAP <b>${vwap.toFixed(2)}</b></span>`;

    Plotly.react('vol-chart', volTraces, {
      ...DARK, height: 240, shapes,
      barmode: 'overlay',
      xaxis: xaxisCommon,
      yaxis:  { ...AXIS_BASE, title: { text: 'Volume', font: { size: 10 } } },
      yaxis2: { ...AXIS_BASE, title: { text: 'Cumulative', font: { size: 10 } },
                overlaying: 'y', side: 'right', showgrid: false },
    }, CFG);
  }

  // ---- 3. Depth chart ----
  const [dxs, dBid1, dAsk1, dBid2, dAsk2] = filterToDay(
    pd.xs, pd.bid_vol1, pd.ask_vol1, pd.bid_vol2, pd.ask_vol2,
  );

  const negAsk1 = dAsk1.map(v => -v);
  const negAsk2 = dAsk2.map(v => -v);

  const depthTraces = [
    {
      x: dxs, y: dBid2,
      name: 'Bid Vol L2', legendgroup: 'bvol2',
      type: 'scatter', mode: 'none',
      fill: 'tozeroy',
      fillcolor: 'rgba(63,185,80,0.08)',
      hovertemplate: '%{y}<extra>Bid L2</extra>',
    },
    {
      x: dxs, y: negAsk2,
      name: 'Ask Vol L2', legendgroup: 'avol2',
      type: 'scatter', mode: 'none',
      fill: 'tozeroy',
      fillcolor: 'rgba(248,81,73,0.08)',
      hovertemplate: '%{y:.0f}<extra>Ask L2</extra>',
    },
    {
      x: dxs, y: dBid1,
      name: 'Bid Vol L1',
      type: 'scatter', mode: 'lines',
      fill: 'tozeroy',
      line: { color: '#3fb950', width: 1 },
      fillcolor: 'rgba(63,185,80,0.2)',
      hovertemplate: '%{y}<extra>Bid L1</extra>',
    },
    {
      x: dxs, y: negAsk1,
      name: 'Ask Vol L1',
      type: 'scatter', mode: 'lines',
      fill: 'tozeroy',
      line: { color: '#f85149', width: 1 },
      fillcolor: 'rgba(248,81,73,0.2)',
      hovertemplate: '%{customdata:.0f}<extra>Ask L1</extra>',
      customdata: dAsk1,
    },
  ];

  Plotly.react('depth-chart', depthTraces, {
    ...DARK, height: 220, shapes,
    xaxis: xaxisCommon,
    yaxis: { ...AXIS_BASE, title: { text: 'Volume (bid +, ask −)', font: { size: 10 } } },
  }, CFG);

  attachZoomSync();
}

/* ---- zoom synchronisation ---- */
function attachZoomSync() {
  const ids = ['price-chart', 'vol-chart', 'depth-chart'];
  ids.forEach(id => {
    const el = document.getElementById(id);
    if (!el || !el._fullLayout) return;

    // Re-attach (Plotly.react removes old listeners)
    el.removeAllListeners && el.removeAllListeners('plotly_relayout');
    el.on('plotly_relayout', ev => {
      if (syncLock) return;
      let newRange = null;
      if (ev['xaxis.range[0]'] !== undefined && ev['xaxis.range[1]'] !== undefined) {
        newRange = [ev['xaxis.range[0]'], ev['xaxis.range[1]']];
      } else if (ev['xaxis.autorange'] === true) {
        newRange = null;
      } else {
        return; // ignore unrelated events
      }
      xRange = newRange;
      syncLock = true;
      ids.filter(c => c !== id).forEach(otherId => {
        const other = document.getElementById(otherId);
        if (!other || !other._fullLayout) return;
        if (xRange) {
          Plotly.relayout(other, { 'xaxis.range[0]': xRange[0], 'xaxis.range[1]': xRange[1] });
        } else {
          Plotly.relayout(other, { 'xaxis.autorange': true });
        }
      });
      syncLock = false;
    });
  });
}

/* ---- init ---- */
function init() {
  document.getElementById('folder-label').textContent = DATA.folder;

  // Product dropdown
  const sel = document.getElementById('prod-sel');
  DATA.products.forEach(p => {
    const o = document.createElement('option');
    o.value = p; o.textContent = p;
    sel.appendChild(o);
  });
  sel.addEventListener('change', () => {
    product = sel.value; xRange = null; render();
  });

  // Day filter buttons
  const btnGroup = document.getElementById('day-btns');
  function makeBtn(label, day) {
    const btn = document.createElement('button');
    btn.textContent = label;
    if (day === null) btn.classList.add('active');
    btn.addEventListener('click', () => {
      dayFilter = day; xRange = null;
      document.querySelectorAll('#day-btns button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      render();
    });
    btnGroup.appendChild(btn);
  }
  makeBtn('All Days', null);
  DATA.sorted_days.forEach(d => makeBtn(dayLabel(d), d));

  document.getElementById('reset-btn').addEventListener('click', () => {
    xRange = null; render();
  });

  if (DATA.products.length > 0) {
    product = DATA.products[0];
    sel.value = product;
    render();
  }
}

window.addEventListener('load', init);
</script>
</body>
</html>
"""


def build_html(payload: dict) -> str:
    data_json = json.dumps(payload, separators=(",", ":"))
    return HTML_TEMPLATE.replace("{data_json}", data_json).replace("{folder}", payload["folder"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    folder = Path(sys.argv[1]).resolve()
    if not folder.is_dir():
        print(f"Error: '{folder}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    output = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path("viz.html")

    folder_name = folder.name

    print(f"Loading prices from  {folder} ...")
    prices = read_prices(folder)
    print(f"  -> {sum(len(v) for v in prices.values()):,} rows across {len(prices)} products")

    print(f"Loading trades from  {folder} ...")
    trades = read_trades(folder)
    print(f"  -> {sum(len(v) for v in trades.values()):,} rows across {len(trades)} symbols")

    print("Building payload ...")
    payload = build_payload(prices, trades, folder_name)
    print(f"  -> products: {', '.join(payload['products'])}")
    print(f"  -> days:     {payload['sorted_days']}")

    print(f"Writing {output} ...")
    html = build_html(payload)
    output.write_text(html, encoding="utf-8")

    size_kb = output.stat().st_size / 1024
    print(f"Done! {output}  ({size_kb:.0f} KB)")
    print(f"\nOpen in browser:  file:///{output.resolve().as_posix()}")


if __name__ == "__main__":
    main()
