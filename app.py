import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import json
import os
import re
import feedparser
from bs4 import BeautifulSoup
import urllib.parse
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ページ設定
st.set_page_config(
    page_title="AI株式投資バトル ＆ シミュレーション",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_FILE = "data.json"

# --- データ永続化関数 ---
def get_default_users():
    return {
        "User_A (慎重型)": {
            "system_prompt": "あなたはリスク管理を最優先する慎重な投資アナリストです。テクニカル指標の過熱感やニュースのリスク要因を厳しく評価し、確実性が極めて高いと判断した場合のみ『買い』と提示してください。疑わしい材料や高値圏では迷わず『様子見』または『売り』を選択してください。",
            "cash": 1000000.0,
            "initial_cash": 1000000.0,
            "holdings": {},
            "history": []
        },
        "User_B (積極型)": {
            "system_prompt": "あなたは成長性と上昇モメンタムを重視する積極的な投資アナリストです。将来性や好材料ニュース、ブレイクアウトの兆候を前向きに捉え、上昇余地があれば多少のリスクを取ってでも積極的に『買い』と提示してください。",
            "cash": 1000000.0,
            "initial_cash": 1000000.0,
            "holdings": {},
            "history": []
        }
    }

def load_data():
    if not os.path.exists(DATA_FILE):
        data = {"users": get_default_users(), "timeline": []}
        save_data(data)
        return data
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
            # 旧フォーマット（直下がユーザー辞書）からの移行互換
            if "users" not in raw:
                data = {"users": raw, "timeline": []}
            else:
                data = raw
            data.setdefault("timeline", [])
            for user, val in data["users"].items():
                val.setdefault("initial_cash", 1000000.0)
                val.setdefault("cash", 1000000.0)
                val.setdefault("holdings", {})
                val.setdefault("history", [])
                val.setdefault("system_prompt", "あなたはプロの投資アナリストです。")
            return data
    except Exception as e:
        st.error(f"データファイル読み込みエラー: {e}")
        return {"users": get_default_users(), "timeline": []}

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

app_state = load_data()
users_data = app_state["users"]
timeline_logs = app_state["timeline"]

# 主要銘柄マスター
POPULAR_STOCKS = {
    "トヨタ自動車 (7203.T)": {"code": "7203.T", "name": "トヨタ自動車", "query": "トヨタ自動車"},
    "ソニーグループ (6758.T)": {"code": "6758.T", "name": "ソニーグループ", "query": "ソニーグループ"},
    "任天堂 (7974.T)": {"code": "7974.T", "name": "任天堂", "query": "任天堂"},
    "ソフトバンクグループ (9984.T)": {"code": "9984.T", "name": "ソフトバンクG", "query": "ソフトバンクグループ"},
    "Apple (AAPL)": {"code": "AAPL", "name": "Apple", "query": "Apple 株"},
    "Microsoft (MSFT)": {"code": "MSFT", "name": "Microsoft", "query": "マイクロソフト 株"},
    "NVIDIA (NVDA)": {"code": "NVDA", "name": "NVIDIA", "query": "エヌビディア 株"},
}

# --- サイドバー設定 ---
st.sidebar.header("👤 プレイヤー選択 ＆ 設定")
user_names = list(users_data.keys())
selected_user = st.sidebar.selectbox("アクティブプレイヤー", user_names, index=0)

# 新規プレイヤー追加
with st.sidebar.expander("➕ 新規プレイヤーの登録"):
    new_user_name = st.text_input("プレイヤー名", placeholder="例: User_C (テクニカル型)")
    new_user_prompt = st.text_area(
        "初期AI判断基準（プロンプト）",
        value="あなたは移動平均線と出来高のブレイクを最重視するテクニカルアナリストです。",
        height=70
    )
    if st.button("登録する", width="stretch"):
        clean_name = new_user_name.strip()
        if not clean_name:
            st.warning("プレイヤー名を入力してください。")
        elif clean_name in users_data:
            st.warning("同名のプレイヤーが既に存在します。")
        else:
            users_data[clean_name] = {
                "system_prompt": new_user_prompt.strip(),
                "cash": 1000000.0,
                "initial_cash": 1000000.0,
                "holdings": {},
                "history": []
            }
            save_data(app_state)
            st.success(f"プレイヤー '{clean_name}' を追加しました！")
            st.rerun()

current_user_profile = users_data[selected_user]

# AIプロンプト編集
st.sidebar.subheader("🎯 投資判断基準（システムプロンプト）")
edited_prompt = st.sidebar.text_area(
    f"{selected_user} の判断基準",
    value=current_user_profile.get("system_prompt", ""),
    height=100,
    help="このプロンプトがAIの売買判断方針になります。編集すると自動保存されます。"
)
if edited_prompt != current_user_profile.get("system_prompt"):
    current_user_profile["system_prompt"] = edited_prompt
    users_data[selected_user] = current_user_profile
    save_data(app_state)
    st.sidebar.caption("💾 プロンプトを自動保存しました")

# --- 共通キャッシュ関数 ---
@st.cache_data(ttl=60)
def get_ollama_models(base_url: str):
    """ローカルOllamaにインストールされているモデル一覧を自動取得"""
    try:
        res = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=3)
        if res.status_code == 200:
            names = [m["name"] for m in res.json().get("models", [])]
            if names:
                return names
    except Exception:
        pass
    return ["qwen2.5:7b", "deepseek-r1:8b", "llama3.3", "mistral"]

# --- Streamlit Secretsからキーを自動取得 ---
def get_secret(key: str, section: str = "api_keys", fallback: str = "") -> str:
    """Streamlit SecretsまたはSectionから安全にキーを取得"""
    try:
        return st.secrets[section][key]
    except (KeyError, FileNotFoundError):
        return fallback

# Secretsに設定済みのキーを確認
SECRET_GEMINI_KEY = get_secret("GEMINI_API_KEY")
SECRET_GROQ_KEY = get_secret("GROQ_API_KEY")
SECRET_DEFAULT_PROVIDER = get_secret("default_provider", section="settings", fallback="")
SECRET_GEMINI_MODEL = get_secret("default_gemini_model", section="settings", fallback="gemini-1.5-flash")
SECRET_GROQ_MODEL = get_secret("default_groq_model", section="settings", fallback="llama-3.3-70b-versatile")

# クラウドにSecretsが設定されているか確認
has_gemini_secret = bool(SECRET_GEMINI_KEY and SECRET_GEMINI_KEY != "ここにGemini APIキーを貼り付け")
has_groq_secret = bool(SECRET_GROQ_KEY and SECRET_GROQ_KEY != "ここにGroq APIキーを貼り付け")

# デフォルトプロバイダーの決定（Secrets設定 > ローカルOllama）
if SECRET_DEFAULT_PROVIDER == "gemini" and has_gemini_secret:
    default_provider_index = 1
elif SECRET_DEFAULT_PROVIDER == "groq" and has_groq_secret:
    default_provider_index = 2
elif has_gemini_secret:
    default_provider_index = 1
elif has_groq_secret:
    default_provider_index = 2
else:
    default_provider_index = 0  # ローカルOllamaがデフォルト

# クラウド公開用 API / ローカル切替設定
st.sidebar.header("🌐 AIエンジン・モデル設定")
llm_provider = st.sidebar.selectbox(
    "AIプロバイダー",
    ["Ollama (ローカル)", "Google Gemini API (クラウド)", "Groq API (高速クラウド)", "OpenAI互換 API"],
    index=default_provider_index,
    help="お友達のPC（RTX 4070など）で動かす場合は Ollama を選択できます。"
)

api_key = ""
custom_model = ""
custom_url = ""
temperature = st.sidebar.slider("AIの温度 (リスク志向・創造性)", min_value=0.0, max_value=1.0, value=0.3, step=0.1, help="低いほど論理的・堅実、高いほど積極的・柔軟な分析になります。")

if llm_provider == "Ollama (ローカル)":
    ollama_url = st.sidebar.text_input("Ollama URL", value="http://localhost:11434")
    detected_models = get_ollama_models(ollama_url)
    model_choice_mode = st.sidebar.radio("モデル指定", ["検出されたモデルから選択", "自由に入力"], horizontal=True)
    if model_choice_mode == "検出されたモデルから選択":
        ollama_model = st.sidebar.selectbox("使用モデル", detected_models, index=0)
    else:
        ollama_model = st.sidebar.text_input("モデル名を手動入力", value="qwen2.5:7b")
    st.sidebar.caption("💡 RTX 4070なら `qwen2.5:14b` や `deepseek-r1:8b` も超高速で動作します！")

elif llm_provider == "Google Gemini API (クラウド)":
    if has_gemini_secret:
        api_key = SECRET_GEMINI_KEY
        st.sidebar.success("✅ Gemini APIキーは設定済みです（入力不要）")
    else:
        api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Google AI Studioで取得したAPIキー")
    # gemini-1.5-proは無料枠では利用不可のため除外
    gemini_options = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]
    default_gemini_idx = gemini_options.index(SECRET_GEMINI_MODEL) if SECRET_GEMINI_MODEL in gemini_options else 0
    custom_model = st.sidebar.selectbox("Gemini モデル", gemini_options, index=default_gemini_idx)
    st.sidebar.caption("💡 無料枠で使えるモデルのみ表示しています")

elif llm_provider == "Groq API (高速クラウド)":
    if has_groq_secret:
        api_key = SECRET_GROQ_KEY
        st.sidebar.success("✅ Groq APIキーは設定済みです（入力不要）")
    else:
        api_key = st.sidebar.text_input("Groq API Key", type="password", help="Groq Consoleで取得したAPIキー")
    groq_options = ["llama-3.3-70b-versatile", "qwen-2.5-32b", "deepseek-r1-distill-llama-70b", "gemma2-9b-it"]
    default_groq_idx = groq_options.index(SECRET_GROQ_MODEL) if SECRET_GROQ_MODEL in groq_options else 0
    custom_model = st.sidebar.selectbox("Groq モデル", groq_options, index=default_groq_idx)

else:
    api_key = st.sidebar.text_input("API Key", type="password")
    custom_url = st.sidebar.text_input("Endpoint URL", value="https://api.openai.com/v1/chat/completions")
    custom_model = st.sidebar.text_input("モデル名", value="gpt-4o-mini")

# サイドバー: 銘柄選択
st.sidebar.markdown("---")
st.sidebar.header("🔍 銘柄選択")
selection_mode = st.sidebar.radio("選択方法", ["主要銘柄から選択", "コード手動入力"], index=0)

if selection_mode == "主要銘柄から選択":
    selected_name = st.sidebar.selectbox("銘柄", list(POPULAR_STOCKS.keys()), index=0)
    ticker_input = POPULAR_STOCKS[selected_name]["code"]
    default_news_query = POPULAR_STOCKS[selected_name]["query"]
else:
    ticker_input = st.sidebar.text_input("銘柄コード（例: 7203.T, NVDA）", value="7203.T").strip().upper()
    default_news_query = ticker_input

period_options = {"1ヶ月": "1mo", "3ヶ月": "3mo", "6ヶ月": "6mo", "1年": "1y", "年初来": "ytd"}
selected_period_label = st.sidebar.selectbox("期間", list(period_options.keys()), index=2)
selected_period = period_options[selected_period_label]

# --- 共通キャッシュ関数 ---
@st.cache_data(ttl=300)
def load_stock_data(symbol: str, period: str):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    info = ticker.info if hasattr(ticker, "info") else {}
    return df, info

@st.cache_data(ttl=300)
def get_cached_price(symbol: str):
    try:
        t = yf.Ticker(symbol)
        h = t.history(period="5d")
        if not h.empty:
            return float(h["Close"].iloc[-1])
    except Exception:
        pass
    return 0.0

@st.cache_data(ttl=600)
def fetch_stock_news(query: str, max_items: int = 4):
    encoded_query = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ja&gl=JP&ceid=JP:ja"
    feed = feedparser.parse(rss_url)
    news_items = []
    for entry in feed.entries[:max_items]:
        raw_summary = getattr(entry, "summary", "")
        soup = BeautifulSoup(raw_summary, "html.parser")
        clean_summary = soup.get_text(separator=" ", strip=True)
        source_title = "ニュース"
        if hasattr(entry, "source") and hasattr(entry.source, "title"):
            source_title = entry.source.title
        news_items.append({
            "title": entry.title,
            "link": entry.link,
            "published": getattr(entry, "published", "日時不明"),
            "source": source_title,
            "summary": clean_summary
        })
    return news_items

# --- 統一LLM呼び出し関数（Ollama / Gemini / Groq / OpenAI互換） ---
def request_llm(prompt_text: str):
    if llm_provider == "Ollama (ローカル)":
        endpoint = f"{ollama_url.rstrip('/')}/api/generate"
        try:
            payload = {
                "model": ollama_model,
                "prompt": prompt_text,
                "stream": False,
                "options": {"temperature": float(temperature)}
            }
            res = requests.post(endpoint, json=payload, timeout=90)
            res.raise_for_status()
            return res.json().get("response", "")
        except requests.exceptions.ConnectionError:
            return "⚠️ Ollamaサーバーに接続できません。ローカルでOllamaが起動しているか確認してください。"
        except Exception as e:
            return f"⚠️ エラー: {e}"

    elif llm_provider == "Google Gemini API (クラウド)":
        if not api_key:
            return "⚠️ Gemini APIキーが設定されていません。サイドバーに入力してください。"
        # 新旧両方のキー形式（AIza... / AQ.）に対応するためヘッダーで認証
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "contents": [{"parts": [{"text": prompt_text}]}],
            "generationConfig": {"temperature": float(temperature)}
        }
        # v1 → v1beta の順で試みる
        for api_version in ["v1beta", "v1"]:
            url = f"https://generativelanguage.googleapis.com/{api_version}/models/{custom_model}:generateContent"
            try:
                res = requests.post(url, headers=headers, json=payload, timeout=60)
                if res.status_code == 404:
                    continue  # 次のバージョンを試す
                res.raise_for_status()
                data = res.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except requests.exceptions.HTTPError as e:
                if res.status_code == 404:
                    continue
                return f"⚠️ Gemini APIエラー ({api_version}): {e}"
            except Exception as e:
                return f"⚠️ Gemini APIエラー: {e}"
        return f"⚠️ Gemini APIエラー: モデル '{custom_model}' が見つかりませんでした。別のモデルをお試しください。"

    elif llm_provider == "Groq API (高速クラウド)":
        if not api_key:
            return "⚠️ Groq APIキーが設定されていません。サイドバーに入力してください。"
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        try:
            payload = {
                "model": custom_model,
                "messages": [{"role": "user", "content": prompt_text}],
                "temperature": float(temperature)
            }
            res = requests.post(url, headers=headers, json=payload, timeout=60)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"⚠️ Groq APIエラー: {e}"

    else: # OpenAI互換
        if not api_key:
            return "⚠️ APIキーが設定されていません。サイドバーに入力してください。"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        try:
            payload = {
                "model": custom_model,
                "messages": [{"role": "user", "content": prompt_text}],
                "temperature": float(temperature)
            }
            res = requests.post(custom_url, headers=headers, json=payload, timeout=60)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"⚠️ OpenAI APIエラー: {e}"

# 銘柄データ取得
with st.spinner("銘柄データをロード中..."):
    try:
        df, info = load_stock_data(ticker_input, selected_period)
    except Exception as e:
        st.error(f"データ取得エラー: {e}")
        st.stop()

if df.empty:
    st.error(f"銘柄コード '{ticker_input}' のデータが見つかりませんでした。")
    st.stop()

company_name = info.get("shortName") or info.get("longName") or ticker_input
currency = info.get("currency", "JPY")
currency_symbol = "円" if currency == "JPY" else f" {currency}"
latest_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else latest_close
price_change = latest_close - prev_close
pct_change = (price_change / prev_close) * 100 if prev_close > 0 else 0.0

search_query = default_news_query if selection_mode == "主要銘柄から選択" else f"{company_name} 株"
news_data = fetch_stock_news(search_query, max_items=4)

# ==========================================
# 画面トップ: コンパクトステータスバー
# ==========================================
st.markdown(f"## 📈 AI株式投資バトル ＆ シミュレーション")
top_c1, top_c2, top_c3, top_c4 = st.columns([1.2, 1, 1, 1.2])

# 全ユーザーの成績集計
leaderboard_data = []
for uname, udata in users_data.items():
    init_c = udata.get("initial_cash", 1000000.0)
    c_bal = udata.get("cash", 1000000.0)
    h_val = 0.0
    wins = 0
    total_trades = len(udata.get("history", []))
    for scode, hinfo in udata.get("holdings", {}).items():
        sh = hinfo.get("shares", 0)
        if sh > 0:
            p = latest_close if scode == ticker_input else get_cached_price(scode)
            h_val += sh * (p if p > 0 else hinfo.get("avg_price", 0.0))
    for h in udata.get("history", []):
        if h.get("action") == "SELL" and h.get("profit", 0) > 0:
            wins += 1
    total_w = c_bal + h_val
    profit = total_w - init_c
    profit_pct = (profit / init_c) * 100
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    leaderboard_data.append({
        "name": uname,
        "total_wealth": total_w,
        "profit": profit,
        "profit_pct": profit_pct,
        "cash": c_bal,
        "stock_val": h_val,
        "trades": total_trades,
        "win_rate": win_rate,
        "prompt": udata.get("system_prompt", "")
    })

leaderboard_data.sort(key=lambda x: x["profit_pct"], reverse=True)
top_player = leaderboard_data[0] if leaderboard_data else None

# アクティブユーザーの資産
cur_user_stats = next((x for x in leaderboard_data if x["name"] == selected_user), None)
top_c1.metric("アクティブプレイヤー", selected_user, f"総資産: {cur_user_stats['total_wealth']:,.0f}円" if cur_user_stats else "")
top_c2.metric("プレイヤー通算損益", f"{cur_user_stats['profit']:+,.0f} 円" if cur_user_stats else "0 円", f"{cur_user_stats['profit_pct']:+.2f}%" if cur_user_stats else "0%")
top_c3.metric("🥇 現在のランキング1位", f"{top_player['name']}" if top_player else "---", f"{top_player['profit_pct']:+.2f}%" if top_player else "")
top_c4.metric("AIエンジン", f"{llm_provider.split(' ')[0]}", f"モデル: {custom_model or ollama_model}")

st.markdown("---")

# ==========================================
# メイン画面: 横2分割レイアウト
# ==========================================
col_left, col_right = st.columns([1.15, 0.85], gap="large")

# ------------------------------------------
# 左側カラム: 株価推移・AI分析・手動トレード
# ------------------------------------------
with col_left:
    st.subheader(f"📊 {company_name} ({ticker_input})")
    
    # 指標カード
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("現在終値", f"{latest_close:,.2f}{currency_symbol}", f"{price_change:+,.2f}{currency_symbol} ({pct_change:+.2f}%)")
    m2.metric("期間最高値", f"{df['High'].max():,.2f}{currency_symbol}")
    m3.metric("期間最安値", f"{df['Low'].min():,.2f}{currency_symbol}")
    m4.metric("直近出来高", f"{int(df['Volume'].iloc[-1]):,} 株")

    # チャート表示形式の切替
    c_type1, c_type2 = st.columns([1.2, 1])
    with c_type1:
        chart_style = st.radio("表示スタイル", ["ローソク足", "エリア折れ線"], horizontal=True, label_visibility="collapsed")
    with c_type2:
        show_ma = st.checkbox("移動平均線 (5日/25日)", value=True)

    # Plotlyでローソク足 ＋ 出来高チャートを構築
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25]
    )

    dates = df.index.strftime("%Y-%m-%d")

    if chart_style == "ローソク足":
        fig.add_trace(
            go.Candlestick(
                x=dates,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="株価",
                increasing_line_color="#00C805",
                decreasing_line_color="#FF333A"
            ),
            row=1, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df["Close"],
                mode="lines",
                name="終値",
                line=dict(color="#2962FF", width=2),
                fill="tozeroy",
                fillcolor="rgba(41, 98, 255, 0.1)"
            ),
            row=1, col=1
        )

    # 移動平均線
    if show_ma:
        if len(df) >= 5:
            ma5 = df["Close"].rolling(5).mean()
            fig.add_trace(go.Scatter(x=dates, y=ma5, mode="lines", name="5日線", line=dict(color="#FF9800", width=1.5)), row=1, col=1)
        if len(df) >= 25:
            ma25 = df["Close"].rolling(25).mean()
            fig.add_trace(go.Scatter(x=dates, y=ma25, mode="lines", name="25日線", line=dict(color="#9C27B0", width=1.5)), row=1, col=1)

    # 出来高バー
    vol_colors = ["#00C805" if c >= o else "#FF333A" for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(
        go.Bar(x=dates, y=df["Volume"], name="出来高", marker_color=vol_colors, opacity=0.6),
        row=2, col=1
    )

    fig.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, width="stretch")

    # 左側タブ: AI分析 / 参照ニュース / 手動トレード
    tab_ai, tab_news, tab_trade = st.tabs(["🧠 AI投資分析 (RAG)", "📰 参照ニュース", "⚡ 手動トレード"])

    # プロンプト準備
    recent_days = min(7, len(df))
    recent_df = df.tail(recent_days)[["Open", "High", "Low", "Close", "Volume"]]
    recent_summary_str = "\n".join([
        f"- {idx.strftime('%m/%d')}: 始={r['Open']:,.1f}, 高={r['High']:,.1f}, 安={r['Low']:,.1f}, 終={r['Close']:,.1f}, 出来高={int(r['Volume']):,}"
        for idx, r in recent_df.iterrows()
    ])
    news_prompt_str = "\n".join([
        f"・{n['title']} ({n['source']}): {n['summary'][:100]}"
        for n in news_data
    ]) if news_data else "最新ニュースなし"

    analysis_prompt = f"""あなたはプロの株式アナリストです。以下の投資判断基準に従い、株価と最新ニュースを分析して売買判断を提示してください。

【あなたの投資判断基準・ポリシー】
{current_user_profile.get("system_prompt", "リスクとリターンを総合判断してください。")}

【銘柄】{company_name} ({ticker_input}) 終値: {latest_close:,.2f}{currency_symbol} (前日比: {price_change:+,.2f}{currency_symbol}, {pct_change:+.2f}%)
【直近{recent_days}日データ】
{recent_summary_str}
【最新Webニュース（RAG）】
{news_prompt_str}

【回答フォーマット】必ず1行目に売買判断を記述してください。
【売買判断】: 買い（または 売り、様子見 のいずれか一つ）
【総合判断理由】: 
【テクニカル分析】: 
【ニュース・ファンダ考察】: 
【リスク要因】: 
"""
    session_analysis_key = f"analysis_{selected_user}_{ticker_input}"
    if session_analysis_key not in st.session_state:
        st.session_state[session_analysis_key] = None

    with tab_ai:
        c_btn, c_note = st.columns([1.5, 3])
        with c_btn:
            if st.button("🚀 AI分析を実行", type="primary", width="stretch"):
                with st.spinner("AIアナリストが分析中..."):
                    res = request_llm(analysis_prompt)
                    st.session_state[session_analysis_key] = res
                    # タイムラインにAI判断を記録
                    verdict = "買い" if "買い" in res[:150] else ("売り" if "売り" in res[:150] else "様子見")
                    timeline_logs.insert(0, {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "user": selected_user,
                        "type": "AI_ANALYSIS",
                        "title": f"AI判断: 【{verdict}】 推奨",
                        "detail": f"{company_name} ({ticker_input}) @ {latest_close:,.1f}円"
                    })
                    app_state["timeline"] = timeline_logs[:50]
                    save_data(app_state)
        with c_note:
            st.caption(f"適用ポリシー: {current_user_profile.get('system_prompt', '')[:40]}...")

        ai_out = st.session_state.get(session_analysis_key)
        if ai_out:
            first_part = ai_out[:180]
            if "買い" in first_part:
                st.success("### 🟢 AI推奨アクション: 【 買い 】")
            elif "売り" in first_part:
                st.error("### 🔴 AI推奨アクション: 【 売り 】")
            elif "様子見" in first_part:
                st.warning("### 🟡 AI推奨アクション: 【 様子見 】")
            else:
                st.info("### ⚪ AI推奨アクション: 【 判定中 】")
            st.markdown(ai_out)
        else:
            st.info("「🚀 AI分析を実行」ボタンを押すと、このプレイヤーの基準で売買判断が下されます。")

    with tab_news:
        if news_data:
            for item in news_data:
                st.markdown(f"**[{item['title']}]({item['link']})**")
                st.caption(f"📡 {item['source']} ｜ 🕒 {item['published']}")
                if item["summary"]:
                    st.write(item["summary"])
                st.divider()
        else:
            st.info("関連ニュースは見つかりませんでした。")

    with tab_trade:
        user_c = current_user_profile["cash"]
        user_h = current_user_profile["holdings"].get(ticker_input, {"shares": 0, "avg_price": 0.0})
        h_sh = user_h["shares"]
        h_avg = user_h["avg_price"]
        cur_eval = h_sh * latest_close
        cur_gain = (latest_close - h_avg) * h_sh if h_sh > 0 else 0.0
        cur_gain_pct = ((latest_close - h_avg) / h_avg * 100) if (h_sh > 0 and h_avg > 0) else 0.0

        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("保有株数", f"{h_sh:,} 株")
        tc2.metric("平均取得単価", f"{h_avg:,.2f} 円" if h_sh > 0 else "---")
        tc3.metric("評価損益", f"{cur_eval:,.0f} 円", delta=f"{cur_gain:+,.0f} 円 ({cur_gain_pct:+.2f}%)" if h_sh > 0 else None)

        qty_col, buy_col, sell_col = st.columns([2, 1, 1])
        with qty_col:
            trade_qty = st.number_input("取引株数", min_value=10, max_value=50000, value=100, step=100)
            trade_total = trade_qty * latest_close
            st.caption(f"概算約定代金: **{trade_total:,.0f} 円**")

        with buy_col:
            st.write("")
            st.write("")
            if st.button("🟢 買付実行", width="stretch"):
                if user_c < trade_total:
                    st.error("現金が不足しています！")
                else:
                    new_sh = h_sh + trade_qty
                    new_avg = ((h_sh * h_avg) + trade_total) / new_sh
                    current_user_profile["cash"] -= trade_total
                    current_user_profile["holdings"][ticker_input] = {"shares": new_sh, "avg_price": round(new_avg, 2), "name": company_name}
                    now_s = datetime.now().strftime("%H:%M:%S")
                    current_user_profile["history"].insert(0, {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": ticker_input,
                        "name": company_name,
                        "action": "BUY",
                        "shares": trade_qty,
                        "price": round(latest_close, 2),
                        "total": round(trade_total, 2),
                        "cash_after": round(current_user_profile["cash"], 2)
                    })
                    timeline_logs.insert(0, {
                        "time": now_s,
                        "user": selected_user,
                        "type": "BUY",
                        "title": f"買付: {company_name} {trade_qty:,}株",
                        "detail": f"約定: {latest_close:,.1f}円 (計: {trade_total:,.0f}円)"
                    })
                    app_state["timeline"] = timeline_logs[:50]
                    save_data(app_state)
                    st.success("買付完了！")
                    st.rerun()

        with sell_col:
            st.write("")
            st.write("")
            if st.button("🔴 売却実行", width="stretch"):
                if h_sh < trade_qty:
                    st.error("保有株数が不足しています！")
                else:
                    realized_gain = (latest_close - h_avg) * trade_qty
                    new_sh = h_sh - trade_qty
                    current_user_profile["cash"] += trade_total
                    if new_sh == 0:
                        current_user_profile["holdings"][ticker_input] = {"shares": 0, "avg_price": 0.0, "name": company_name}
                    else:
                        current_user_profile["holdings"][ticker_input]["shares"] = new_sh
                    now_s = datetime.now().strftime("%H:%M:%S")
                    current_user_profile["history"].insert(0, {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": ticker_input,
                        "name": company_name,
                        "action": "SELL",
                        "shares": trade_qty,
                        "price": round(latest_close, 2),
                        "total": round(trade_total, 2),
                        "profit": round(realized_gain, 2),
                        "cash_after": round(current_user_profile["cash"], 2)
                    })
                    timeline_logs.insert(0, {
                        "time": now_s,
                        "user": selected_user,
                        "type": "SELL",
                        "title": f"売却: {company_name} {trade_qty:,}株",
                        "detail": f"損益: {realized_gain:+,.0f}円 (約定: {latest_close:,.1f}円)"
                    })
                    app_state["timeline"] = timeline_logs[:50]
                    save_data(app_state)
                    st.success("売却完了！")
                    st.rerun()

# ------------------------------------------
# 右側カラム: リーダーボード・完全自動化・タイムライン
# ------------------------------------------
with col_right:
    # 1. 🏆 リアルタイムリーダーボード
    st.subheader("🏆 リアルタイム・リーダーボード")
    
    # 上位3名の表彰カード
    rank_icons = ["🥇", "🥈", "🥉"]
    rank_cols = st.columns(min(3, len(leaderboard_data)))
    for idx, p in enumerate(leaderboard_data[:3]):
        with rank_cols[idx]:
            icon = rank_icons[idx]
            st.markdown(f"**{icon} 第{idx+1}位**")
            st.caption(f"**{p['name']}**")
            st.metric("総資産", f"{p['total_wealth']:,.0f}円", f"{p['profit_pct']:+.2f}%")

    # 全員の一覧（テーブル）
    with st.expander("📊 全プレイヤー成績一覧表", expanded=False):
        disp_df = pd.DataFrame([{
            "順位": f"第{i+1}位",
            "プレイヤー": p["name"],
            "総資産 (円)": f"{p['total_wealth']:,.0f}",
            "損益額 (円)": f"{p['profit']:+,.0f}",
            "損益率 (%)": f"{p['profit_pct']:+.2f}%",
            "勝率 (%)": f"{p['win_rate']:.1f}%",
            "取引数": p["trades"]
        } for i, p in enumerate(leaderboard_data)])
        st.dataframe(disp_df, width="stretch")

    st.markdown("---")

    # 2. 🤖 完全放置・複数銘柄の一括自動トレード
    st.subheader("🤖 完全放置・一括自動トレード")
    st.caption("AIが複数銘柄を同時にスキャンし、各プレイヤーの方針に従って自動売買を実行します。")

    auto_targets = st.multiselect(
        "自動取引の対象銘柄",
        options=list(POPULAR_STOCKS.keys()),
        default=["トヨタ自動車 (7203.T)", "ソニーグループ (6758.T)", "任天堂 (7974.T)", "ソフトバンクグループ (9984.T)"]
    )

    col_auto1, col_auto2 = st.columns(2)
    with col_auto1:
        run_my_auto = st.button("▶️ 選択中プレイヤーで自動実行", width="stretch")
    with col_auto2:
        run_battle_auto = st.button("⚔️ 全プレイヤー一斉対戦自動実行", width="stretch", type="primary")

    # 自動トレード実行ロジック
    def execute_auto_trade_for_user(uname: str, target_stocks: list):
        u_profile = users_data[uname]
        actions_done = 0
        now_str = datetime.now().strftime("%H:%M:%S")

        for s_label in target_stocks:
            stock_info = POPULAR_STOCKS[s_label]
            s_code = stock_info["code"]
            s_name = stock_info["name"]
            
            p_price = get_cached_price(s_code)
            if p_price <= 0:
                continue

            # 直近データからの簡易AIシグナル判定
            u_prompt = u_profile.get("system_prompt", "")
            # 慎重型・積極型のキーワード判定やシグナル
            df_sub, _ = load_stock_data(s_code, "1mo")
            if len(df_sub) < 3:
                continue
            ret_3d = (df_sub["Close"].iloc[-1] - df_sub["Close"].iloc[-3]) / df_sub["Close"].iloc[-3] * 100

            action = "HOLD"
            # プロンプトに応じた判定ルール
            if "慎重" in u_prompt or "リスク" in u_prompt:
                # 慎重型: 下落からの反発確認時のみ買い、上昇しすぎは様子見または売り
                if -5.0 < ret_3d < -1.0:
                    action = "BUY"
                elif ret_3d > 4.0:
                    action = "SELL"
            else:
                # 積極型: 上昇モメンタムがあれば積極買い
                if ret_3d > 0.5:
                    action = "BUY"
                elif ret_3d < -4.0:
                    action = "SELL"

            u_cash = u_profile["cash"]
            u_held = u_profile["holdings"].get(s_code, {"shares": 0, "avg_price": 0.0})
            sh_count = 100
            order_cost = sh_count * p_price

            if action == "BUY" and u_cash >= order_cost:
                new_sh = u_held["shares"] + sh_count
                new_avg = ((u_held["shares"] * u_held["avg_price"]) + order_cost) / new_sh
                u_profile["cash"] -= order_cost
                u_profile["holdings"][s_code] = {"shares": new_sh, "avg_price": round(new_avg, 2), "name": s_name}
                u_profile["history"].insert(0, {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": s_code,
                    "name": s_name,
                    "action": "BUY",
                    "shares": sh_count,
                    "price": round(p_price, 2),
                    "total": round(order_cost, 2),
                    "cash_after": round(u_profile["cash"], 2)
                })
                timeline_logs.insert(0, {
                    "time": now_str,
                    "user": uname,
                    "type": "BUY",
                    "title": f"🤖 自動買付: {s_name} {sh_count}株",
                    "detail": f"約定: {p_price:,.1f}円 (AIシグナル: 買い)"
                })
                actions_done += 1

            elif action == "SELL" and u_held["shares"] >= sh_count:
                real_p = (p_price - u_held["avg_price"]) * sh_count
                new_sh = u_held["shares"] - sh_count
                u_profile["cash"] += order_cost
                if new_sh == 0:
                    u_profile["holdings"][s_code] = {"shares": 0, "avg_price": 0.0, "name": s_name}
                else:
                    u_profile["holdings"][s_code]["shares"] = new_sh
                u_profile["history"].insert(0, {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": s_code,
                    "name": s_name,
                    "action": "SELL",
                    "shares": sh_count,
                    "price": round(p_price, 2),
                    "total": round(order_cost, 2),
                    "profit": round(real_p, 2),
                    "cash_after": round(u_profile["cash"], 2)
                })
                timeline_logs.insert(0, {
                    "time": now_str,
                    "user": uname,
                    "type": "SELL",
                    "title": f"🤖 自動売却: {s_name} {sh_count}株",
                    "detail": f"確定損益: {real_p:+,.0f}円 (AIシグナル: 売り)"
                })
                actions_done += 1
            else:
                timeline_logs.insert(0, {
                    "time": now_str,
                    "user": uname,
                    "type": "HOLD",
                    "title": f"🤖 様子見: {s_name}",
                    "detail": f"条件不一致または資金/株数不足のため見送り"
                })

        users_data[uname] = u_profile
        return actions_done

    if run_my_auto:
        with st.spinner(f"{selected_user} の自動トレードを実行中..."):
            cnt = execute_auto_trade_for_user(selected_user, auto_targets)
            app_state["timeline"] = timeline_logs[:50]
            save_data(app_state)
            st.success(f"{selected_user}: {cnt} 件の売買を実行しました！")
            st.rerun()

    if run_battle_auto:
        with st.spinner("全プレイヤーの自動対戦トレードを実行中..."):
            total_act = 0
            for u in users_data.keys():
                total_act += execute_auto_trade_for_user(u, auto_targets)
            app_state["timeline"] = timeline_logs[:50]
            save_data(app_state)
            st.success(f"全プレイヤー一斉対戦完了！ 合計 {total_act} 件の注文が約定しました！")
            st.rerun()

    st.markdown("---")

    # 3. 💬 リアルタイム・タイムライン（フィード形式）
    st.subheader("💬 アクティビティ・タイムライン")
    st.caption("AIの判断とプレイヤーたちの売買がリアルタイムに流れるフィードです。")

    if timeline_logs:
        for item in timeline_logs[:12]:
            t_type = item.get("type", "")
            if t_type == "BUY":
                badge = "🟢 **[買付]**"
            elif t_type == "SELL":
                badge = "🔴 **[売却]**"
            elif t_type == "AI_ANALYSIS":
                badge = "🧠 **[AI判断]**"
            else:
                badge = "🟡 **[様子見]**"

            with st.container():
                st.markdown(f"{badge} **{item.get('title')}**")
                st.caption(f"👤 {item.get('user')} ｜ 🕒 {item.get('time')} ｜ {item.get('detail')}")
                st.divider()
    else:
        st.info("まだアクティビティはありません。取引やAI分析を実行するとここに流れます。")
