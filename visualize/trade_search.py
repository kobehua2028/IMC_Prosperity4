#!/usr/bin/env python3
"""
Trade search server for IMC Prosperity data.

Usage:
  python trade_search.py            # loads ../data, serves on http://localhost:5050
  python trade_search.py --port 8080
  python trade_search.py --data path/to/data
"""

import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

from flask import Flask, jsonify, render_template_string, request

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_trades(data_dir: Path) -> list[dict]:
    trades = []
    files = sorted(data_dir.rglob("trades_*.csv"))
    if not files:
        print(f"[warn] no trades_*.csv found under {data_dir}", file=sys.stderr)
    for fpath in files:
        rel = fpath.relative_to(data_dir)
        round_name = rel.parts[0]
        match = re.search(r"day_(-?\d+)\.csv$", fpath.name)
        file_day = int(match.group(1)) if match else 0
        with open(fpath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                try:
                    trades.append({
                        "round":     round_name,
                        "day":       file_day,
                        "timestamp": int(row["timestamp"]),
                        "symbol":    row["symbol"].strip(),
                        "price":     float(row["price"]),
                        "quantity":  int(row["quantity"]),
                        "buyer":     row.get("buyer", "").strip(),
                        "seller":    row.get("seller", "").strip(),
                        "currency":  row.get("currency", "").strip(),
                    })
                except (ValueError, KeyError):
                    continue
    trades.sort(key=lambda r: (r["round"], r["day"], r["timestamp"]))
    return trades


def build_meta(trades: list[dict]) -> dict:
    rounds  = sorted({t["round"]  for t in trades})
    days    = sorted({t["day"]    for t in trades})
    symbols = sorted({t["symbol"] for t in trades})
    buyers  = sorted({t["buyer"]  for t in trades if t["buyer"]})
    sellers = sorted({t["seller"] for t in trades if t["seller"]})
    prices  = [t["price"]    for t in trades]
    qtys    = [t["quantity"] for t in trades]
    return {
        "total":    len(trades),
        "rounds":   rounds,
        "days":     days,
        "symbols":  symbols,
        "buyers":   buyers,
        "sellers":  sellers,
        "price_min": min(prices) if prices else 0,
        "price_max": max(prices) if prices else 0,
        "qty_min":  min(qtys)   if qtys   else 0,
        "qty_max":  max(qtys)   if qtys   else 0,
    }


# ---------------------------------------------------------------------------
# Filtering + stats
# ---------------------------------------------------------------------------

def filter_trades(trades, params):
    q         = params.get("q", "").strip().lower()
    rounds    = set(params.getlist("round"))
    days      = {int(d) for d in params.getlist("day") if d}
    symbols   = set(params.getlist("symbol"))
    min_price = _float(params.get("min_price"))
    max_price = _float(params.get("max_price"))
    min_qty   = _int(params.get("min_qty"))
    max_qty   = _int(params.get("max_qty"))
    buyer     = params.get("buyer", "").strip().lower()
    seller    = params.get("seller", "").strip().lower()

    out = []
    for t in trades:
        if rounds  and t["round"]  not in rounds:  continue
        if days    and t["day"]    not in days:     continue
        if symbols and t["symbol"] not in symbols:  continue
        if min_price is not None and t["price"]    < min_price: continue
        if max_price is not None and t["price"]    > max_price: continue
        if min_qty   is not None and t["quantity"] < min_qty:   continue
        if max_qty   is not None and t["quantity"] > max_qty:   continue
        if buyer  and buyer  not in t["buyer"].lower():   continue
        if seller and seller not in t["seller"].lower():  continue
        if q and not (
            q in t["symbol"].lower() or
            q in t["buyer"].lower()  or
            q in t["seller"].lower() or
            q in str(t["price"])     or
            q in t["round"].lower()
        ):
            continue
        out.append(t)
    return out


def compute_stats(trades):
    if not trades:
        return {"count": 0, "total_volume": 0, "avg_price": None, "vwap": None,
                "min_price": None, "max_price": None, "min_qty": None, "max_qty": None}
    prices = [t["price"]    for t in trades]
    qtys   = [t["quantity"] for t in trades]
    total_vol   = sum(qtys)
    vwap = sum(p * q for p, q in zip(prices, qtys)) / total_vol if total_vol else None
    return {
        "count":        len(trades),
        "total_volume": total_vol,
        "avg_price":    round(sum(prices) / len(prices), 4),
        "vwap":         round(vwap, 4) if vwap else None,
        "min_price":    min(prices),
        "max_price":    max(prices),
        "min_qty":      min(qtys),
        "max_qty":      max(qtys),
    }


def _float(v):
    try: return float(v)
    except (TypeError, ValueError): return None

def _int(v):
    try: return int(v)
    except (TypeError, ValueError): return None


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
TRADES: list[dict] = []
META:   dict       = {}

PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trade Search — IMC Prosperity</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:      #0d1117;
  --surface: #161b22;
  --border:  #21262d;
  --text:    #c9d1d9;
  --muted:   #6e7681;
  --accent:  #1f6feb;
  --green:   #3fb950;
  --red:     #f85149;
  --yellow:  #e3b341;
  --purple:  #d2a8ff;
}
body { font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); display: flex; flex-direction: column; min-height: 100vh; }

/* ---- header ---- */
header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 12px 20px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
header h1 { font-size: 1rem; font-weight: 600; color: #e6edf3; white-space: nowrap; }
.badge { background: var(--accent); color: #fff; font-size: 0.72rem; padding: 2px 8px; border-radius: 20px; }
header .right { margin-left: auto; display: flex; gap: 8px; align-items: center; }
.export-btn { background: #21262d; color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 5px 12px; font-size: 0.78rem; cursor: pointer; }
.export-btn:hover { background: var(--accent); border-color: var(--accent); color: #fff; }

/* ---- layout ---- */
.layout { display: flex; flex: 1; overflow: hidden; }
aside { width: 240px; min-width: 200px; background: var(--surface); border-right: 1px solid var(--border); padding: 14px 12px; overflow-y: auto; display: flex; flex-direction: column; gap: 14px; }
main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

/* ---- sidebar filters ---- */
.filter-section h3 { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; color: var(--muted); margin-bottom: 6px; }
.search-wrap { position: relative; }
.search-wrap input { width: 100%; }
.search-wrap .clear-btn { position: absolute; right: 7px; top: 50%; transform: translateY(-50%); background: none; border: none; color: var(--muted); cursor: pointer; font-size: 0.9rem; padding: 0; line-height: 1; }
.search-wrap .clear-btn:hover { color: var(--text); }
input[type=text], input[type=number] {
  width: 100%; background: var(--bg); color: var(--text); border: 1px solid var(--border);
  border-radius: 6px; padding: 5px 8px; font-size: 0.82rem;
}
input:focus { outline: 2px solid var(--accent); outline-offset: 1px; border-color: transparent; }
.range-row { display: flex; gap: 5px; align-items: center; }
.range-row span { font-size: 0.75rem; color: var(--muted); flex-shrink: 0; }
.checkgroup { display: flex; flex-direction: column; gap: 4px; max-height: 120px; overflow-y: auto; }
.checkgroup label { display: flex; align-items: center; gap: 6px; font-size: 0.8rem; cursor: pointer; padding: 2px 0; }
.checkgroup input[type=checkbox] { accent-color: var(--accent); cursor: pointer; flex-shrink: 0; }
.reset-btn { width: 100%; background: #21262d; color: var(--muted); border: 1px solid var(--border); border-radius: 6px; padding: 6px; font-size: 0.78rem; cursor: pointer; margin-top: 4px; }
.reset-btn:hover { color: var(--text); background: var(--border); }

/* ---- stats bar ---- */
.stats-bar { background: var(--surface); border-bottom: 1px solid var(--border); padding: 8px 16px; display: flex; gap: 20px; flex-wrap: wrap; font-size: 0.78rem; color: var(--muted); align-items: center; }
.stat { display: flex; flex-direction: column; gap: 1px; }
.stat .val { font-size: 0.92rem; font-weight: 600; color: var(--text); }
.stat .lbl { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; }
.stat.green .val { color: var(--green); }
.stat.yellow .val { color: var(--yellow); }
.stat.purple .val { color: var(--purple); }

/* ---- table area ---- */
.table-wrap { flex: 1; overflow: auto; }
table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
thead { position: sticky; top: 0; z-index: 10; background: var(--surface); }
th { padding: 9px 12px; text-align: left; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); border-bottom: 1px solid var(--border); cursor: pointer; white-space: nowrap; user-select: none; }
th:hover { color: var(--text); }
th .sort-icon { margin-left: 4px; opacity: 0.4; }
th.sorted .sort-icon { opacity: 1; color: var(--accent); }
td { padding: 7px 12px; border-bottom: 1px solid #0d1117; vertical-align: middle; }
tr:hover td { background: #161b22; }
.sym-chip { background: rgba(31,111,235,0.15); color: #58a6ff; border-radius: 4px; padding: 1px 7px; font-size: 0.75rem; font-weight: 600; white-space: nowrap; }
.round-chip { background: rgba(63,185,80,0.12); color: var(--green); border-radius: 4px; padding: 1px 6px; font-size: 0.72rem; }
.price-val { color: var(--yellow); font-variant-numeric: tabular-nums; }
.qty-val { color: var(--purple); font-variant-numeric: tabular-nums; }
.party { color: var(--muted); font-size: 0.77rem; max-width: 100px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.empty { padding: 60px 20px; text-align: center; color: var(--muted); }

/* ---- pagination ---- */
.pagination { background: var(--surface); border-top: 1px solid var(--border); padding: 8px 16px; display: flex; align-items: center; gap: 8px; font-size: 0.8rem; color: var(--muted); }
.pagination .pg-info { flex: 1; }
.pg-btn { background: var(--bg); border: 1px solid var(--border); color: var(--text); border-radius: 5px; padding: 3px 10px; cursor: pointer; font-size: 0.78rem; }
.pg-btn:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
.pg-btn:disabled { opacity: 0.35; cursor: default; }
.pg-num { font-weight: 600; color: var(--text); }
select.pg-size { background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 5px; padding: 3px 6px; font-size: 0.78rem; }
</style>
</head>
<body>
<header>
  <h1>Trade Search</h1>
  <span class="badge" id="total-badge">loading...</span>
  <div class="right">
    <button class="export-btn" id="export-btn">Export CSV</button>
  </div>
</header>

<div class="layout">
  <!-- Sidebar -->
  <aside>
    <div class="filter-section">
      <h3>Search</h3>
      <div class="search-wrap">
        <input type="text" id="q" placeholder="symbol, buyer, seller, price..." autocomplete="off">
        <button class="clear-btn" id="q-clear" title="Clear">&#x2715;</button>
      </div>
    </div>

    <div class="filter-section">
      <h3>Round</h3>
      <div class="checkgroup" id="round-checks"></div>
    </div>

    <div class="filter-section">
      <h3>Day</h3>
      <div class="checkgroup" id="day-checks"></div>
    </div>

    <div class="filter-section">
      <h3>Symbol</h3>
      <div class="checkgroup" id="symbol-checks"></div>
    </div>

    <div class="filter-section">
      <h3>Price Range</h3>
      <div class="range-row">
        <input type="number" id="min-price" placeholder="Min" step="any">
        <span>to</span>
        <input type="number" id="max-price" placeholder="Max" step="any">
      </div>
    </div>

    <div class="filter-section">
      <h3>Quantity Range</h3>
      <div class="range-row">
        <input type="number" id="min-qty" placeholder="Min" min="0">
        <span>to</span>
        <input type="number" id="max-qty" placeholder="Max" min="0">
      </div>
    </div>

    <div class="filter-section">
      <h3>Buyer</h3>
      <input type="text" id="buyer" placeholder="filter buyer...">
    </div>

    <div class="filter-section">
      <h3>Seller</h3>
      <input type="text" id="seller" placeholder="filter seller...">
    </div>

    <button class="reset-btn" id="reset-btn">Reset All Filters</button>
  </aside>

  <main>
    <!-- Stats bar -->
    <div class="stats-bar" id="stats-bar">
      <div class="stat"><span class="val" id="s-count">—</span><span class="lbl">Trades</span></div>
      <div class="stat green"><span class="val" id="s-vol">—</span><span class="lbl">Total Volume</span></div>
      <div class="stat yellow"><span class="val" id="s-vwap">—</span><span class="lbl">VWAP</span></div>
      <div class="stat"><span class="val" id="s-avg">—</span><span class="lbl">Avg Price</span></div>
      <div class="stat"><span class="val" id="s-lo">—</span><span class="lbl">Low</span></div>
      <div class="stat"><span class="val" id="s-hi">—</span><span class="lbl">High</span></div>
      <div class="stat purple"><span class="val" id="s-qlo">—</span><span class="lbl">Min Qty</span></div>
      <div class="stat purple"><span class="val" id="s-qhi">—</span><span class="lbl">Max Qty</span></div>
    </div>

    <!-- Table -->
    <div class="table-wrap" id="table-wrap">
      <table>
        <thead>
          <tr id="thead-row"></tr>
        </thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>

    <!-- Pagination -->
    <div class="pagination">
      <span class="pg-info" id="pg-info"></span>
      <select class="pg-size" id="pg-size">
        <option value="50">50 / page</option>
        <option value="100" selected>100 / page</option>
        <option value="250">250 / page</option>
        <option value="500">500 / page</option>
        <option value="0">All</option>
      </select>
      <button class="pg-btn" id="pg-first">&#171; First</button>
      <button class="pg-btn" id="pg-prev">&#8249; Prev</button>
      <span class="pg-num" id="pg-label">1</span>
      <button class="pg-btn" id="pg-next">Next &#8250;</button>
      <button class="pg-btn" id="pg-last">Last &#187;</button>
    </div>
  </main>
</div>

<script>
/* ---- state ---- */
const COLUMNS = [
  { key: 'round',     label: 'Round',     render: r => `<span class="round-chip">${r.round}</span>` },
  { key: 'day',       label: 'Day',       render: r => r.day >= 0 ? `+${r.day}` : r.day },
  { key: 'timestamp', label: 'Timestamp', render: r => r.timestamp.toLocaleString() },
  { key: 'symbol',    label: 'Symbol',    render: r => `<span class="sym-chip">${r.symbol}</span>` },
  { key: 'price',     label: 'Price',     render: r => `<span class="price-val">${r.price.toFixed(2)}</span>` },
  { key: 'quantity',  label: 'Qty',       render: r => `<span class="qty-val">${r.quantity}</span>` },
  { key: 'buyer',     label: 'Buyer',     render: r => `<span class="party" title="${r.buyer}">${r.buyer || '—'}</span>` },
  { key: 'seller',    label: 'Seller',    render: r => `<span class="party" title="${r.seller}">${r.seller || '—'}</span>` },
  { key: 'currency',  label: 'Currency',  render: r => `<span class="party">${r.currency || '—'}</span>` },
];

let meta      = null;
let allRows   = [];    // current filtered rows (full set)
let sortKey   = 'timestamp';
let sortAsc   = true;
let page      = 1;
let pageSize  = 100;
let debounceTimer = null;

/* ---- build sidebar once we have meta ---- */
function buildSidebar(m) {
  buildChecks('round-checks',  m.rounds,  'round');
  buildChecks('day-checks',    m.days,    'day',   d => (d >= 0 ? '+' : '') + d);
  buildChecks('symbol-checks', m.symbols, 'symbol');
}

function buildChecks(containerId, values, name, labelFn) {
  const el = document.getElementById(containerId);
  el.innerHTML = '';
  values.forEach(v => {
    const label = document.createElement('label');
    const cb = document.createElement('input');
    cb.type = 'checkbox'; cb.name = name; cb.value = v;
    cb.addEventListener('change', triggerSearch);
    label.appendChild(cb);
    label.appendChild(document.createTextNode(' ' + (labelFn ? labelFn(v) : v)));
    el.appendChild(label);
  });
}

/* ---- build table header once ---- */
function buildHeader() {
  const tr = document.getElementById('thead-row');
  COLUMNS.forEach(col => {
    const th = document.createElement('th');
    th.dataset.key = col.key;
    th.innerHTML = col.label + ' <span class="sort-icon">&#8597;</span>';
    if (col.key === sortKey) {
      th.classList.add('sorted');
      th.querySelector('.sort-icon').innerHTML = sortAsc ? '&#8593;' : '&#8595;';
    }
    th.addEventListener('click', () => {
      if (sortKey === col.key) sortAsc = !sortAsc;
      else { sortKey = col.key; sortAsc = true; }
      renderTable();
    });
    tr.appendChild(th);
  });
}

/* ---- fetch + render ---- */
function buildParams() {
  const params = new URLSearchParams();
  const q = document.getElementById('q').value.trim();
  if (q) params.set('q', q);

  document.querySelectorAll('#round-checks  input:checked').forEach(cb => params.append('round',  cb.value));
  document.querySelectorAll('#day-checks    input:checked').forEach(cb => params.append('day',    cb.value));
  document.querySelectorAll('#symbol-checks input:checked').forEach(cb => params.append('symbol', cb.value));

  const minP = document.getElementById('min-price').value;
  const maxP = document.getElementById('max-price').value;
  const minQ = document.getElementById('min-qty').value;
  const maxQ = document.getElementById('max-qty').value;
  const buyer  = document.getElementById('buyer').value.trim();
  const seller = document.getElementById('seller').value.trim();

  if (minP) params.set('min_price', minP);
  if (maxP) params.set('max_price', maxP);
  if (minQ) params.set('min_qty',   minQ);
  if (maxQ) params.set('max_qty',   maxQ);
  if (buyer)  params.set('buyer',  buyer);
  if (seller) params.set('seller', seller);

  return params;
}

async function fetchData() {
  const params = buildParams();
  const [tradesRes, statsRes] = await Promise.all([
    fetch('/api/trades?' + params),
    fetch('/api/stats?'  + params),
  ]);
  allRows = await tradesRes.json();
  const stats = await statsRes.json();
  updateStats(stats);
  page = 1;
  renderTable();
}

function triggerSearch() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(fetchData, 180);
}

/* ---- stats ---- */
function fmt(n, decimals = 2) {
  if (n == null) return '—';
  if (typeof n === 'number') return n.toLocaleString(undefined, { maximumFractionDigits: decimals });
  return n;
}

function updateStats(s) {
  document.getElementById('s-count').textContent = fmt(s.count, 0);
  document.getElementById('s-vol').textContent   = fmt(s.total_volume, 0);
  document.getElementById('s-vwap').textContent  = fmt(s.vwap);
  document.getElementById('s-avg').textContent   = fmt(s.avg_price);
  document.getElementById('s-lo').textContent    = fmt(s.min_price);
  document.getElementById('s-hi').textContent    = fmt(s.max_price);
  document.getElementById('s-qlo').textContent   = fmt(s.min_qty, 0);
  document.getElementById('s-qhi').textContent   = fmt(s.max_qty, 0);
  document.getElementById('total-badge').textContent = `${(s.count || 0).toLocaleString()} trades`;
}

/* ---- table render (client-side sort + paginate) ---- */
function renderTable() {
  // Update sort icons
  document.querySelectorAll('#thead-row th').forEach(th => {
    th.classList.toggle('sorted', th.dataset.key === sortKey);
    const icon = th.querySelector('.sort-icon');
    if (th.dataset.key === sortKey) icon.innerHTML = sortAsc ? '&#8593;' : '&#8595;';
    else icon.innerHTML = '&#8597;';
  });

  // Sort
  const sorted = [...allRows].sort((a, b) => {
    let av = a[sortKey], bv = b[sortKey];
    if (typeof av === 'string') av = av.toLowerCase();
    if (typeof bv === 'string') bv = bv.toLowerCase();
    if (av < bv) return sortAsc ? -1 :  1;
    if (av > bv) return sortAsc ?  1 : -1;
    return 0;
  });

  // Paginate
  const total = sorted.length;
  const ps    = pageSize === 0 ? total : pageSize;
  const pages = ps > 0 ? Math.max(1, Math.ceil(total / ps)) : 1;
  page = Math.min(Math.max(page, 1), pages);

  const start = (page - 1) * ps;
  const slice = ps > 0 ? sorted.slice(start, start + ps) : sorted;

  const tbody = document.getElementById('tbody');
  tbody.innerHTML = '';

  if (slice.length === 0) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td colspan="${COLUMNS.length}" class="empty">No trades match the current filters.</td>`;
    tbody.appendChild(tr);
  } else {
    slice.forEach(row => {
      const tr = document.createElement('tr');
      tr.innerHTML = COLUMNS.map(col => `<td>${col.render(row)}</td>`).join('');
      tbody.appendChild(tr);
    });
  }

  // Pagination controls
  const endRow = ps > 0 ? Math.min(start + ps, total) : total;
  document.getElementById('pg-info').textContent =
    total === 0 ? 'No results' : `${(start + 1).toLocaleString()}–${endRow.toLocaleString()} of ${total.toLocaleString()}`;
  document.getElementById('pg-label').textContent = `${page} / ${pages}`;
  document.getElementById('pg-first').disabled = page <= 1;
  document.getElementById('pg-prev').disabled  = page <= 1;
  document.getElementById('pg-next').disabled  = page >= pages;
  document.getElementById('pg-last').disabled  = page >= pages;
}

/* ---- export ---- */
document.getElementById('export-btn').addEventListener('click', async () => {
  const params = buildParams();
  const rows = await (await fetch('/api/trades?' + params)).json();
  const header = COLUMNS.map(c => c.key).join(',');
  const lines  = rows.map(r => COLUMNS.map(c => {
    const v = r[c.key];
    const s = v === null || v === undefined ? '' : String(v);
    return s.includes(',') ? `"${s}"` : s;
  }).join(','));
  const csv = [header, ...lines].join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'trades_export.csv';
  a.click();
});

/* ---- pagination buttons ---- */
document.getElementById('pg-first').addEventListener('click', () => { page = 1;     renderTable(); });
document.getElementById('pg-prev' ).addEventListener('click', () => { page--;       renderTable(); });
document.getElementById('pg-next' ).addEventListener('click', () => { page++;       renderTable(); });
document.getElementById('pg-last' ).addEventListener('click', () => { page = 99999; renderTable(); });
document.getElementById('pg-size' ).addEventListener('change', e => {
  pageSize = parseInt(e.target.value); page = 1; renderTable();
});

/* ---- reset ---- */
document.getElementById('reset-btn').addEventListener('click', () => {
  document.getElementById('q').value = '';
  document.getElementById('min-price').value = '';
  document.getElementById('max-price').value = '';
  document.getElementById('min-qty').value   = '';
  document.getElementById('max-qty').value   = '';
  document.getElementById('buyer').value     = '';
  document.getElementById('seller').value    = '';
  document.querySelectorAll('aside input[type=checkbox]').forEach(cb => cb.checked = false);
  fetchData();
});

/* ---- clear search ---- */
document.getElementById('q-clear').addEventListener('click', () => {
  document.getElementById('q').value = ''; triggerSearch();
});

/* ---- attach all filter listeners ---- */
['q','min-price','max-price','min-qty','max-qty','buyer','seller'].forEach(id => {
  document.getElementById(id).addEventListener('input', triggerSearch);
});

/* ---- init ---- */
async function init() {
  meta = await (await fetch('/api/meta')).json();
  buildSidebar(meta);
  buildHeader();
  await fetchData();
}

init();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE_HTML)


@app.route("/api/meta")
def api_meta():
    return jsonify(META)


@app.route("/api/trades")
def api_trades():
    rows = filter_trades(TRADES, request.args)
    return jsonify(rows)


@app.route("/api/stats")
def api_stats():
    rows = filter_trades(TRADES, request.args)
    return jsonify(compute_stats(rows))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None,
                        help="Path to data directory (default: ../data relative to this script)")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    data_dir = Path(args.data).resolve() if args.data else (Path(__file__).parent.parent / "data").resolve()
    if not data_dir.is_dir():
        print(f"Error: data directory '{data_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading trades from {data_dir} ...")
    global TRADES, META
    TRADES = load_all_trades(data_dir)
    META   = build_meta(TRADES)
    print(f"  {len(TRADES):,} trades loaded across {len(META['rounds'])} round(s): {', '.join(META['rounds'])}")
    print(f"  Symbols: {', '.join(META['symbols'])}")
    print(f"\nServing at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
