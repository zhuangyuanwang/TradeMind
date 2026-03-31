import os
import json
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

try:
    import anthropic
except ImportError:
    anthropic = None


st.set_page_config(page_title="TradeMind AI", page_icon="📈", layout="wide")

# ─────────────────────────── Session State Initialization ───────────────────────────

if "cash" not in st.session_state:
    st.session_state.cash = 10000.0
if "positions" not in st.session_state:
    st.session_state.positions = {}       # {ticker: {company, shares, avg_price}}
if "trade_log" not in st.session_state:
    st.session_state.trade_log = []
if "last_data" not in st.session_state:
    st.session_state.last_data = None
if "last_decision" not in st.session_state:
    st.session_state.last_decision = None
if "mark_prices" not in st.session_state:
    # Tracks the latest known price for every held ticker,
    # so portfolio value stays accurate across multi-stock sessions.
    st.session_state.mark_prices = {}

API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
client = None

if anthropic is not None and API_KEY:
    try:
        client = anthropic.Anthropic(api_key=API_KEY)
    except Exception:
        client = None


# ─────────────────────────── Utility Helpers ───────────────────────────

def safe_series(obj: Any) -> pd.Series:
    """Coerce yfinance output (Series or DataFrame) to a plain Series."""
    if isinstance(obj, pd.Series):
        return obj
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 0:
            return pd.Series(dtype="float64")
        return obj.iloc[:, 0]
    return pd.Series(obj)


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely extract a scalar float from a value that may be a Series or DataFrame."""
    try:
        if isinstance(value, pd.Series):
            if value.empty:
                return default
            value = value.iloc[0]
        elif isinstance(value, pd.DataFrame):
            if value.empty:
                return default
            value = value.iloc[0, 0]
        return float(value)
    except Exception:
        return default


def parse_json_from_text(text: str) -> Dict[str, Any]:
    """Extract a JSON object from model output, stripping markdown fences if present."""
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except Exception:
            pass
    raise ValueError(f"Could not parse JSON from model output: {text}")


# ─────────────────────────── News Fetching (Cached) ───────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_tavily_news(ticker: str, company_name: str) -> List[Dict[str, str]]:
    """Fetch latest news via Tavily. Results are cached for 5 minutes to avoid redundant API calls."""
    if not TAVILY_API_KEY:
        return []
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": f"latest stock news for {ticker} {company_name}",
        "topic": "news",
        "search_depth": "advanced",
        "max_results": 5,
        "include_answer": False,
        "include_raw_content": False,
    }
    try:
        response = requests.post("https://api.tavily.com/search", json=payload, timeout=20)
        response.raise_for_status()
        results = response.json().get("results", [])
        return [
            {
                "title": item.get("title", "Untitled"),
                # 500 chars gives the model richer context than the original 280
                "summary": item.get("content", "")[:500],
                "url": item.get("url", ""),
            }
            for item in results[:5]
        ]
    except Exception:
        return []


# ─────────────────────────── News Classification ───────────────────────────

# Maps each category to a display label and emoji
NEWS_CATEGORIES: Dict[str, str] = {
    "Company":  "🏢 Company",
    "Industry": "🏭 Industry",
    "Macro":    "🌍 Macro",
}

@st.cache_data(ttl=300, show_spinner=False)
def classify_news(
    ticker: str, news_items: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """
    Use Claude to label each news item as Company, Industry, or Macro.
    Returns the same list with a 'category' key added to every item.
    Falls back to 'Company' for all items when Claude is unavailable.
    Results are cached for 5 minutes alongside the news cache.
    """
    if not news_items:
        return news_items

    # Fallback: no API key available
    if client is None:
        return [{**item, "category": "Company"} for item in news_items]

    # Build a compact numbered list for the prompt
    numbered = "\n".join(
        f"{i+1}. {item['title']}" for i, item in enumerate(news_items)
    )

    prompt = f"""
You are a financial news classifier.

Classify each headline below for the stock ticker "{ticker}" into exactly one of:
- Company  : news specific to this company (earnings, leadership, products, lawsuits, etc.)
- Industry : news about the broader sector or direct competitors
- Macro    : macroeconomic, geopolitical, interest rate, or market-wide news

Return ONLY a JSON array of objects with "index" (1-based) and "category".
Example: [{{"index": 1, "category": "Company"}}, {{"index": 2, "category": "Macro"}}]

Headlines:
{numbered}
""".strip()

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        labels = json.loads(raw)
        # Build index → category lookup
        label_map = {
            entry["index"]: entry["category"]
            for entry in labels
            if isinstance(entry.get("index"), int)
        }
    except Exception:
        # Any failure → fall back to "Company" for all
        return [{**item, "category": "Company"} for item in news_items]

    valid = set(NEWS_CATEGORIES.keys())
    result = []
    for i, item in enumerate(news_items):
        cat = label_map.get(i + 1, "Company")
        if cat not in valid:
            cat = "Company"
        result.append({**item, "category": cat})
    return result


# ─────────────────────────── Stock Data Fetching ───────────────────────────

def _fetch_price_data(ticker: str) -> Dict[str, Any]:
    """Download OHLCV data and compute price/volume metrics."""
    try:
        df = yf.download(
            ticker,
            period="1mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as e:
        raise ValueError(f"Failed to download market data. Check your connection and try again. ({e})")

    if df is None or df.empty:
        raise ValueError(f"Ticker '{ticker}' not found. Please verify the symbol and try again.")

    close = safe_series(df.get("Close")).dropna()
    volume = safe_series(df.get("Volume")).dropna()

    if close.empty:
        raise ValueError(f"No valid close-price data available for '{ticker}'.")

    current_price = safe_float(close.iloc[-1])
    prev_close = safe_float(close.iloc[-2], current_price) if len(close) > 1 else current_price
    change_1d = ((current_price - prev_close) / prev_close * 100) if prev_close else 0.0

    base_10d = (
        safe_float(close.iloc[-10], safe_float(close.iloc[0]))
        if len(close) >= 10
        else safe_float(close.iloc[0])
    )
    change_10d = ((current_price - base_10d) / base_10d * 100) if base_10d else 0.0

    hist = close.iloc[-20:] if len(close) >= 20 else close
    price_history: List[Dict[str, Any]] = [
        {"date": str(idx)[:10], "price": round(safe_float(price), 2)}
        for idx, price in hist.items()
    ]

    volume_today = int(safe_float(volume.iloc[-1], 0)) if not volume.empty else 0
    avg_volume = safe_float(volume.mean(), 0.0) if not volume.empty else 0.0
    volume_ratio = round(volume_today / avg_volume, 2) if avg_volume else 1.0

    return {
        "current_price": round(current_price, 2),
        "prev_close": round(prev_close, 2),
        "change_1d_pct": round(change_1d, 2),
        "change_10d_pct": round(change_10d, 2),
        "volume_today": volume_today,
        "volume_ratio": volume_ratio,
        "price_history": price_history,
    }


def _fetch_meta_data(ticker: str) -> Dict[str, Any]:
    """Fetch company metadata: name, sector, and trailing P/E ratio."""
    company_name, sector, pe_ratio = ticker, "Unknown", "N/A"
    try:
        info: Dict = {}
        try:
            maybe_info = yf.Ticker(ticker).info
            if isinstance(maybe_info, dict):
                info = maybe_info
        except Exception:
            pass
        company_name = info.get("shortName") or info.get("longName") or ticker
        sector = info.get("sector") or "Unknown"
        trailing_pe = info.get("trailingPE")
        if isinstance(trailing_pe, (int, float)):
            pe_ratio = round(float(trailing_pe), 2)
    except Exception:
        pass
    return {"company_name": company_name, "sector": sector, "pe_ratio": pe_ratio}


def fetch_stock_data(ticker: str) -> Dict[str, Any]:
    """Entry point: fetch price data, metadata, and news for a given ticker."""
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Please enter a ticker symbol.")

    price_data = _fetch_price_data(ticker)
    meta_data = _fetch_meta_data(ticker)

    direction = "upward" if price_data["change_10d_pct"] >= 0 else "downward"
    market_context = (
        f"The stock has shown a {direction} short-term trend over the last several sessions. "
        "Use both market data and any live news headlines in the recommendation."
    )

    news_items = fetch_tavily_news(ticker, meta_data["company_name"])

    return {
        "ticker": ticker,
        **meta_data,
        **price_data,
        "market_context": market_context,
        "news_items": news_items,
    }


# ─────────────────────────── Claude Analysis ───────────────────────────

def ask_claude(data: Dict[str, Any]) -> Dict[str, Any]:
    """Send stock data to Claude and return a structured BUY / SELL / HOLD decision."""

    # Fallback mode when no API key is configured
    if client is None:
        action = (
            "BUY" if data["change_10d_pct"] > 2
            else "SELL" if data["change_10d_pct"] < -2
            else "HOLD"
        )
        confidence = 0.68 if action != "HOLD" else 0.55
        if confidence < 0.6:
            action = "HOLD"
        return {
            "action": action,
            "confidence": confidence,
            "reason": "Fallback mode: decision based on recent price trend and volume only.",
            "news_summary": (
                " | ".join(item["title"] for item in data["news_items"][:2])
                or "No Claude API key detected."
            ),
            "risk": "medium",
        }

    price_table = "\n".join(
        f"  {row['date']}: ${row['price']:.2f}" for row in data["price_history"][-10:]
    )
    news_block = (
        "\n".join(f"- {item['title']}: {item['summary']}" for item in data["news_items"])
        or "- No live news results found."
    )

    prompt = f"""
You are an expert quantitative stock analyst.

Analyze the following stock and return ONLY valid JSON.

STOCK DATA
Ticker: {data['ticker']}
Company: {data['company_name']}
Sector: {data['sector']}
Current Price: {data['current_price']:.2f}
Prev Close: {data['prev_close']:.2f}
1D Change: {data['change_1d_pct']:+.2f}%
10D Change: {data['change_10d_pct']:+.2f}%
Volume Ratio: {data['volume_ratio']:.2f}x

Recent Price History:
{price_table}

Market Context:
{data['market_context']}

Live News Headlines:
{news_block}

Rules:
- If confidence < 0.6, action MUST be HOLD
- Be conservative
- reason should be 2-3 sentences
- news_summary should be 1-2 sentences based on the live news headlines
- risk must be low, medium, or high

Return exactly:
{{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": 0.0,
  "reason": "",
  "news_summary": "",
  "risk": "low" | "medium" | "high"
}}
""".strip()

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=350,
            messages=[{"role": "user", "content": prompt}],
        )
        result = parse_json_from_text(msg.content[0].text.strip())
    except Exception as e:
        return {
            "action": "HOLD",
            "confidence": 0.5,
            "reason": f"Claude response failed; switched to safe fallback. Error: {e}",
            "news_summary": (
                " | ".join(item["title"] for item in data["news_items"][:2])
                or data["market_context"]
            ),
            "risk": "medium",
        }

    # Sanitize and enforce confidence rule
    try:
        result["confidence"] = float(result.get("confidence", 0.5))
    except Exception:
        result["confidence"] = 0.5

    result["action"] = str(result.get("action", "HOLD")).upper()
    if result["action"] not in {"BUY", "SELL", "HOLD"}:
        result["action"] = "HOLD"
    if result["confidence"] < 0.6:
        result["action"] = "HOLD"

    result["reason"] = str(result.get("reason", "No reason provided."))
    result["news_summary"] = str(result.get("news_summary", data["market_context"]))
    risk = str(result.get("risk", "medium")).lower()
    result["risk"] = risk if risk in {"low", "medium", "high"} else "medium"

    return result


# ─────────────────────────── Paper Trading ───────────────────────────

def execute_buy(ticker: str, company_name: str, price: float, amount: float) -> None:
    """Simulate a market buy order for the given notional amount."""
    if amount <= 0:
        st.warning("Buy amount must be greater than zero.")
        return
    if st.session_state.cash < amount:
        st.warning(f"Insufficient cash. Available: ${st.session_state.cash:,.2f}")
        return

    shares = amount / price
    pos = st.session_state.positions.get(ticker)
    if pos:
        # Update weighted average cost basis
        total_shares = pos["shares"] + shares
        new_avg = (pos["shares"] * pos["avg_price"] + shares * price) / total_shares
        st.session_state.positions[ticker] = {
            "company": company_name,
            "shares": total_shares,
            "avg_price": new_avg,
        }
    else:
        st.session_state.positions[ticker] = {
            "company": company_name,
            "shares": shares,
            "avg_price": price,
        }

    st.session_state.cash -= amount
    st.session_state.trade_log.append({
        "time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker,
        "action": "BUY",
        "shares": round(shares, 4),
        "price": round(price, 2),
        "notional": round(amount, 2),
    })


def execute_sell(ticker: str, price: float) -> None:
    """Simulate closing the entire position for a ticker."""
    pos = st.session_state.positions.get(ticker)
    if not pos:
        st.warning(f"No open position found for {ticker}.")
        return

    shares = pos["shares"]
    notional = shares * price
    pnl = (price - pos["avg_price"]) * shares

    st.session_state.cash += notional
    del st.session_state.positions[ticker]
    # Remove stale mark price for the closed position
    st.session_state.mark_prices.pop(ticker, None)

    st.session_state.trade_log.append({
        "time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker,
        "action": "SELL",
        "shares": round(shares, 4),
        "price": round(price, 2),
        "notional": round(notional, 2),
        "pnl": round(pnl, 2),
    })


def portfolio_market_value() -> float:
    """
    Compute total market value of all open positions.
    Uses mark_prices (updated on every successful analysis) so multi-ticker
    portfolios stay accurate even when switching between symbols.
    Falls back to avg_price for positions that have not been re-priced yet.
    """
    total = 0.0
    for ticker, pos in st.session_state.positions.items():
        mark = st.session_state.mark_prices.get(ticker, pos["avg_price"])
        total += pos["shares"] * mark
    return total


def action_color(action: str) -> str:
    return {"BUY": "green", "SELL": "red", "HOLD": "orange"}.get(action, "gray")


# ─────────────────────────── UI: Analysis Panel ───────────────────────────

def render_analysis(data: Dict[str, Any], decision: Dict[str, Any]) -> None:
    # ── Price metrics ──
    metric_cols = st.columns(4)
    metric_cols[0].metric("Price", f"${data['current_price']:.2f}")
    metric_cols[1].metric("1D Change", f"{data['change_1d_pct']:+.2f}%")
    metric_cols[2].metric("10D Change", f"{data['change_10d_pct']:+.2f}%")
    metric_cols[3].metric("Volume Ratio", f"{data['volume_ratio']:.2f}x")

    st.subheader(f"{data['company_name']} ({data['ticker']})")
    st.caption(f"Sector: {data['sector']}  |  P/E: {data['pe_ratio']}")

    # ── AI recommendation ──
    a1, a2, a3 = st.columns([1, 1, 2])
    a1.markdown(f"### :{action_color(decision['action'])}[{decision['action']}]")
    a2.metric("Confidence", f"{float(decision['confidence']) * 100:.0f}%")
    a3.write(f"**Risk:** {decision['risk'].upper()}")

    # ── Paper trade controls with user-configurable buy amount ──
    st.markdown("#### Paper Trade")
    trade_col1, trade_col2, trade_col3 = st.columns([1, 1, 2])
    buy_amount = trade_col3.number_input(
        "Buy amount ($)",
        min_value=100.0,
        max_value=float(max(st.session_state.cash, 100.0)),
        value=min(1000.0, float(max(st.session_state.cash, 100.0))),
        step=100.0,
    )
    if trade_col1.button("Simulate BUY", use_container_width=True):
        execute_buy(data["ticker"], data["company_name"], data["current_price"], buy_amount)
        st.rerun()
    if trade_col2.button("Sell Position", use_container_width=True):
        execute_sell(data["ticker"], data["current_price"])
        st.rerun()

    # ── Price chart ──
    chart_df = pd.DataFrame(data["price_history"])
    chart_df["date"] = pd.to_datetime(chart_df["date"], errors="coerce")
    chart_df = chart_df.dropna(subset=["date"]).set_index("date")
    if not chart_df.empty:
        st.line_chart(chart_df["price"])
    else:
        st.warning("Price chart unavailable.")

    # ── Analysis reasoning ──
    left, right = st.columns(2)
    with left:
        st.markdown("#### Why?")
        st.write(decision["reason"])
    with right:
        st.markdown("#### Market Context")
        st.write(decision["news_summary"])

    # ── Live news (categorized into tabs) ──
    st.markdown("#### Live News")
    if data["news_items"]:
        with st.spinner("Classifying news..."):
            # classify_news is cached, so repeat calls for the same ticker are free
            classified = classify_news(data["ticker"], data["news_items"])

        # Group by category, preserving order within each group
        grouped: Dict[str, List[Dict]] = {cat: [] for cat in NEWS_CATEGORIES}
        for item in classified:
            cat = item.get("category", "Company")
            grouped.setdefault(cat, []).append(item)

        # Only render tabs that have at least one article
        active_cats = [cat for cat in NEWS_CATEGORIES if grouped.get(cat)]
        if active_cats:
            tabs = st.tabs([NEWS_CATEGORIES[cat] for cat in active_cats])
            for tab, cat in zip(tabs, active_cats):
                with tab:
                    for item in grouped[cat]:
                        st.markdown(f"**{item['title']}**")
                        if item["summary"]:
                            st.caption(item["summary"])
                        if item["url"]:
                            st.markdown(f"[Open article]({item['url']})")
                        st.divider()
    else:
        st.write("No live news results found.")

    # ── Portfolio summary ──
    st.markdown("#### Paper Portfolio")
    port_value = portfolio_market_value()
    total_equity = st.session_state.cash + port_value
    total_pnl = total_equity - 10000.0

    p1, p2, p3 = st.columns(3)
    p1.metric("Cash", f"${st.session_state.cash:,.2f}")
    p2.metric("Portfolio Value", f"${port_value:,.2f}")
    p3.metric("Total PnL", f"${total_pnl:+,.2f}")

    if st.session_state.positions:
        rows = []
        for sym, pos in st.session_state.positions.items():
            mark = st.session_state.mark_prices.get(sym, pos["avg_price"])
            unrealized = (mark - pos["avg_price"]) * pos["shares"]
            rows.append({
                "Ticker": sym,
                "Company": pos["company"],
                "Shares": round(pos["shares"], 4),
                "Avg Price": round(pos["avg_price"], 2),
                "Mark": round(mark, 2),
                "Unrealized PnL": round(unrealized, 2),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.write("No open positions yet.")

    # ── Trade history ──
    st.markdown("#### Trade Log")
    if st.session_state.trade_log:
        st.dataframe(pd.DataFrame(st.session_state.trade_log[::-1]), use_container_width=True)
    else:
        st.write("No simulated trades yet.")

    # ── Raw JSON output ──
    st.markdown("#### Decision Engine Output")
    st.code(
        json.dumps(
            {
                "ticker": data["ticker"],
                "company": data["company_name"],
                "decision": decision,
                "news_items": data["news_items"],
            },
            indent=2,
        ),
        language="json",
    )


# ─────────────────────────── App Layout ───────────────────────────

st.title("TradeMind AI 📈")
st.caption("AI-powered trading decision demo · yfinance + Tavily live news + Claude")

# Disclaimer — especially important now that buy/sell buttons exist
st.warning(
    "**Disclaimer:** This tool is for educational and demonstration purposes only. "
    "All analysis results **do not constitute investment advice**. "
    "Invest at your own risk.",
    icon="⚠️",
)

# ── Sidebar ──
with st.sidebar:
    st.header("Setup")
    st.write("1. Install: `pip install streamlit yfinance pandas anthropic requests`")
    st.write("2. Export: `ANTHROPIC_API_KEY` and `TAVILY_API_KEY`")
    st.write("3. Run: `streamlit run app.py`")
    st.divider()
    st.write("Rule: confidence < 0.60 → HOLD")
    if not API_KEY:
        st.info("No Claude API key detected. Running in fallback mode.")
    if not TAVILY_API_KEY:
        st.info("No Tavily API key detected. Live news unavailable.")

    st.divider()
    st.subheader("Paper Portfolio")
    port_value_sidebar = portfolio_market_value()
    st.metric("Cash", f"${st.session_state.cash:,.2f}")
    st.metric("Portfolio Value", f"${port_value_sidebar:,.2f}")
    st.metric("Total Equity", f"${st.session_state.cash + port_value_sidebar:,.2f}")

    if st.button("Reset Portfolio", use_container_width=True):
        st.session_state.cash = 10000.0
        st.session_state.positions = {}
        st.session_state.trade_log = []
        st.session_state.last_data = None
        st.session_state.last_decision = None
        st.session_state.mark_prices = {}
        st.rerun()

# ── Ticker input ──
col1, col2 = st.columns([3, 1])
with col1:
    ticker_input = st.text_input("Ticker", value="NVDA", max_chars=10)
with col2:
    analyze = st.button("Analyze", use_container_width=True)

# ── Quick-pick buttons ──
# FIX: write to session_state and rerun immediately so the ticker value is
# never cleared before the analysis block reads it (original bug).
quick_cols = st.columns(6)
for i, symbol in enumerate(["NVDA", "AAPL", "TSLA", "MSFT", "AMD", "META"]):
    if quick_cols[i].button(symbol, use_container_width=True):
        st.session_state["active_ticker"] = symbol
        st.rerun()

# Resolve which ticker to analyze: manual input takes priority when Analyze is clicked
if analyze:
    st.session_state["active_ticker"] = ticker_input.strip().upper()

ticker = st.session_state.get("active_ticker", ticker_input.strip().upper())

# ── Run analysis whenever a ticker is active ──
if "active_ticker" in st.session_state:
    try:
        with st.spinner("Fetching market data, live news, and generating recommendation..."):
            data = fetch_stock_data(ticker)
            decision = ask_claude(data)

        # Persist results and update mark price for this ticker
        st.session_state.last_data = data
        st.session_state.last_decision = decision
        st.session_state.mark_prices[ticker] = data["current_price"]

    except ValueError as e:
        # User-facing business errors (bad ticker, no data, etc.)
        st.error(f"⚠️ {e}")
    except Exception as e:
        # Unexpected errors
        st.error(f"❌ An unexpected error occurred. Please try again. ({e})")

if st.session_state.last_data and st.session_state.last_decision:
    render_analysis(st.session_state.last_data, st.session_state.last_decision)
