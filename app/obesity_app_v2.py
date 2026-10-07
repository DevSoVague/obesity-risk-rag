"""
obesity_app_v2.py
=================
Streamlit frontend for the obesity classifier + RAG nutrition chat.

Changes from v1:
  - Helper functions (_patch_indexer_for_gateway, _gemini_web_search) moved
    ABOVE the sidebar/tab code so they are defined before use.
  - Auto-reconnect detects collection dim from schema — no more dimension mismatch.
  - Gemini path works independently (no Anthropic key needed):
      _ensure_gemini() is patched to be a no-op when key is injected at runtime,
      and _call_claude() is monkey-patched to use Gemini when running in Gemini mode.
  - CMU Gateway path works independently (no Gemini key needed):
      gateway mode monkey-patches _call_claude() to route through the gateway,
      and suppresses the _ensure_gemini() check.
  - Empty gateway key shows a clear st.error instead of a cryptic SDK exception.

Folder layout (all in same directory):
    obesity_app_v2.py
    model_pipeline.py
    main.py
    indexer.py
    obesity_model_bundle.joblib

Run:
    streamlit run obesity_app_v2.py
"""

import os, json, math, re
from pathlib import Path

import requests as http_requests
import streamlit as st

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"

try:
    from indexer import PDFIndexer, GEN_MODELS, EMBED_MODEL, PAPER_TYPES, INDEX_TYPES
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    PAPER_TYPES  = ["Clinical Guidelines", "Nutrition Study", "Intervention Research",
                    "Systematic Review", "Other"]
    EMBED_MODEL  = "BAAI/bge-large-en-v1.5"
    INDEX_TYPES  = ["HNSW", "IVF_PQ", "DiskANN"]

OBESITY_API = os.environ.get("OBESITY_API_URL", "http://localhost:8001")

OB_ORDER = [
    "Insufficient_Weight", "Normal_Weight",
    "Overweight_Level_I",  "Overweight_Level_II",
    "Obesity_Type_I",      "Obesity_Type_II",  "Obesity_Type_III",
]

CLASS_COLOR = {
    "Insufficient_Weight":  "#60a5fa",
    "Normal_Weight":        "#34d399",
    "Overweight_Level_I":   "#fbbf24",
    "Overweight_Level_II":  "#f97316",
    "Obesity_Type_I":       "#fb923c",
    "Obesity_Type_II":      "#ef4444",
    "Obesity_Type_III":     "#dc2626",
}

TIER_COLOR = {
    "High confidence":                          "#34d399",
    "Moderate confidence — consider follow-up": "#fbbf24",
    "Low confidence — borderline case":         "#f97316",
    "Very low confidence — inconclusive":       "#f87171",
}

CLASS_DESC = {
    "Insufficient_Weight":  "Body weight is below the healthy range. Focus on nutrient-dense foods and building muscle mass.",
    "Normal_Weight":        "Weight is within the healthy range. Maintain current diet and activity habits.",
    "Overweight_Level_I":   "Slightly above healthy weight. Small dietary adjustments and increased activity can help.",
    "Overweight_Level_II":  "Moderately above healthy weight. Structured diet and regular exercise are recommended.",
    "Obesity_Type_I":       "Class I obesity. Medical guidance and a supervised diet and exercise plan are advised.",
    "Obesity_Type_II":      "Class II obesity. Healthcare provider involvement is strongly recommended.",
    "Obesity_Type_III":     "Class III obesity. Comprehensive medical intervention is recommended as a priority.",
}

CLASS_GENERIC_ADVICE = {
    "Insufficient Weight":   "Focus on calorie-dense whole foods, strength training, and regular meals.",
    "Normal Weight":         "Maintain your current habits — balanced diet and regular physical activity.",
    "Overweight Level I":    "Aim for a modest calorie deficit, increase vegetable intake, and add 150+ minutes of activity per week.",
    "Overweight Level II":   "A structured diet plan and consistent exercise routine are recommended. Consider consulting a dietitian.",
    "Obesity Type I":        "Medical guidance is advisable. A supervised diet and exercise programme can make a significant difference.",
    "Obesity Type II":       "Healthcare provider involvement is strongly recommended for a personalised intervention plan.",
    "Obesity Type III":      "Comprehensive medical evaluation and intervention should be the first priority.",
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Obesity Risk Classifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
*,*::before,*::after{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif!important;box-sizing:border-box;}
.stApp{background:#212121;color:#ececec;}.main{background:#212121;}
#MainMenu,footer{visibility:hidden!important;}.stDeployButton{display:none!important;}
header{display:none!important;}
[data-testid="collapsedControl"],[data-testid="stSidebarCollapseButton"]{display:none!important;}
section[data-testid="stSidebar"]{background:#171717!important;border-right:1px solid #2f2f2f!important;min-width:260px!important;max-width:260px!important;}
section[data-testid="stSidebar"]>div{padding:0 12px!important;overflow-x:hidden;}
.block-container{max-width:1100px!important;margin:0 auto!important;padding:0 48px 140px 48px!important;}

/* tabs */
div[data-testid="stTabs"]{border-bottom:1px solid #2f2f2f!important;margin-bottom:24px!important;}
div[data-testid="stTabs"] [role="tablist"]{justify-content:center!important;display:flex!important;}
div[data-testid="stTabs"] [role="tab"]{font-size:.95rem!important;font-weight:500!important;color:#666!important;border:none!important;background:transparent!important;padding:10px 40px!important;}
div[data-testid="stTabs"] [role="tab"][aria-selected="true"]{color:#ececec!important;border-bottom:2px solid #e53935!important;}
div[data-testid="stTabs"] [role="tab"] p,div[data-testid="stTabs"] [role="tab"] span{font-size:.95rem!important;font-weight:500!important;}

/* buttons */
.block-container .stButton>button{background:#2a2a2a!important;border:1px solid #333!important;color:#9b9b9b!important;border-radius:6px!important;font-size:.83rem!important;}
.block-container .stButton>button:hover{background:#333!important;color:#ececec!important;border-color:#444!important;}
.block-container .stButton>button[kind="primary"]{background:#e53935!important;color:#fff!important;border-color:#e53935!important;}
.block-container .stButton>button[kind="primary"]:hover{background:#c62828!important;}
section[data-testid="stSidebar"] .stButton>button{width:100%!important;background:#2a2a2a!important;border:1px solid #333!important;color:#9b9b9b!important;border-radius:6px!important;font-size:.8rem!important;padding:7px 8px!important;}
section[data-testid="stSidebar"] .stButton>button[kind="primary"]{background:#e53935!important;color:#fff!important;border-color:#e53935!important;}

/* inputs */
.stSelectbox>div>div,div[data-testid="stTextInput"] input,div[data-testid="stNumberInput"] input{background:#2a2a2a!important;border:1px solid #333!important;color:#ececec!important;border-radius:6px!important;}
section[data-testid="stSidebar"] .stSelectbox>div>div{background:#2a2a2a!important;border:1px solid #333!important;color:#ececec!important;border-radius:6px!important;}
div[data-testid="stCheckbox"] label,div[data-testid="stRadio"] label{color:#9b9b9b!important;font-size:.84rem!important;}
.stSelectbox label,div[data-testid="stNumberInput"] label,div[data-testid="stSlider"] label{color:#9b9b9b!important;font-size:.8rem!important;}
div[data-testid="stRadio"]>div{gap:6px!important;}
div[data-testid="stRadio"]>div>label{background:#2a2a2a!important;border:1px solid #333!important;border-radius:8px!important;padding:8px 14px!important;font-size:.82rem!important;color:#9b9b9b!important;cursor:pointer!important;transition:all .12s!important;}
div[data-testid="stRadio"]>div>label:hover{border-color:#555!important;color:#ececec!important;}

/* chat */
div[data-testid="stChatMessage"]{background:transparent!important;border:none!important;padding:14px 0!important;border-bottom:1px solid #2f2f2f!important;}
div[data-testid="stChatMessage"] p{color:#ececec!important;font-size:.93rem!important;line-height:1.65!important;}
div[data-testid="stChatInput"]{position:fixed!important;bottom:0!important;left:calc(50% + 130px)!important;transform:translateX(-50%)!important;width:min(1000px,calc(100vw - 280px))!important;background:#212121!important;padding:16px 0 28px 0!important;z-index:999!important;}
div[data-testid="stChatInput"]>div{background:transparent!important;border:1px solid #3f3f3f!important;border-radius:16px!important;padding:14px 20px!important;min-height:64px!important;}
div[data-testid="stChatInput"]>div:focus-within{border-color:#555!important;box-shadow:0 0 0 1px #555!important;}
div[data-testid="stChatInput"] textarea{background:transparent!important;color:#ececec!important;font-size:1rem!important;border:none!important;outline:none!important;}
div[data-testid="stChatInput"] textarea::placeholder{color:#555!important;}
div[data-testid="stChatInput"] button{background:#e53935!important;border-radius:10px!important;color:#fff!important;}

/* cards */
.result-card{background:#2a2a2a;border:1px solid #333;border-radius:12px;padding:22px 24px;margin:8px 0;}
.paper-card{background:#2a2a2a;border:1px solid #333;border-radius:8px;padding:12px 14px;margin:5px 0;}
.paper-card .ptitle{font-weight:600;font-size:.84rem;color:#ececec;margin-bottom:4px;}
.paper-card .purl{font-size:.71rem;color:#60a5fa;}
.paper-card .pscore{font-size:.67rem;color:#555;margin-top:3px;font-family:monospace;}
.chunk-preview{background:#2a2a2a;border-left:2px solid #444;padding:.65rem .9rem;border-radius:0 6px 6px 0;margin:.4rem 0;font-size:.78rem;color:#bbb;white-space:pre-wrap;font-family:monospace;line-height:1.5;}

/* badges */
.badge{display:inline-block;background:#2a2a2a;color:#9b9b9b;padding:1px 7px;border-radius:3px;font-size:.66rem;font-weight:500;margin-right:3px;border:1px solid #333;font-family:monospace;}
.badge-type{background:#1e1533;color:#a78bfa;border-color:#3a2f55;}
.badge-url{background:#0f1e33;color:#60a5fa;border-color:#1e3a55;}
.badge-score{background:#0f2218;color:#34d399;border-color:#1a3d2b;}
.reflect-pass{background:#0f2218;border-left:2px solid #34d399;padding:.4rem .7rem;border-radius:0 5px 5px 0;font-size:.74rem;color:#34d399;margin:.25rem 0;}
.reflect-fail{background:#220f0f;border-left:2px solid #f87171;padding:.4rem .7rem;border-radius:0 5px 5px 0;font-size:.74rem;color:#f87171;margin:.25rem 0;}
.reflect-fixed{background:#1f1a0f;border-left:2px solid #fbbf24;padding:.4rem .7rem;border-radius:0 5px 5px 0;font-size:.74rem;color:#fbbf24;margin:.25rem 0;}

/* alerts */
div[data-testid="stSuccess"]{background:#0f2218!important;border:1px solid #1a3d2b!important;color:#34d399!important;border-radius:6px!important;}
div[data-testid="stError"]{background:#220f0f!important;border:1px solid #3d1a1a!important;color:#f87171!important;border-radius:6px!important;}
div[data-testid="stInfo"]{background:#0f1e33!important;border:1px solid #1a3050!important;color:#60a5fa!important;border-radius:6px!important;}
div[data-testid="stWarning"]{background:#1f1a0f!important;border:1px solid #3d300f!important;color:#fbbf24!important;border-radius:6px!important;}

/* misc */
hr{border-color:#2f2f2f!important;margin:16px 0!important;}
div[data-testid="stMetric"]{background:#2a2a2a!important;border:1px solid #333!important;border-radius:8px!important;padding:12px!important;}
div[data-testid="stMetric"] label{color:#666!important;}
div[data-testid="stMetricValue"]{color:#ececec!important;}
details summary{font-size:.79rem!important;font-weight:500!important;color:#666!important;}
.stCaption,small{color:#555!important;font-size:.72rem!important;}
.block-container h2{color:#ececec!important;font-size:1rem!important;font-weight:600!important;margin:20px 0 10px 0!important;}
.block-container h3{color:#ececec!important;font-size:.88rem!important;font-weight:600!important;}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:#1a1a1a;}
::-webkit-scrollbar-thumb{background:#333;border-radius:3px;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════════
_CHAT_PATH    = Path("/tmp/obesity_chat.json")
_INDEXED_PATH = Path("/tmp/obesity_indexed.json")

def _save_chat():
    try:
        _CHAT_PATH.write_text(
            json.dumps([{"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chat_history]),
            encoding="utf-8",
        )
    except Exception:
        pass

def _load_chat():
    try:
        if _CHAT_PATH.exists():
            return json.loads(_CHAT_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []

def _save_indexed():
    try:
        _INDEXED_PATH.write_text(
            json.dumps({"files": st.session_state.indexed_files,
                        "url_map": st.session_state.file_url_map}),
            encoding="utf-8",
        )
    except Exception:
        pass

def _load_indexed():
    try:
        if _INDEXED_PATH.exists():
            d = json.loads(_INDEXED_PATH.read_text(encoding="utf-8"))
            return d.get("files", []), d.get("url_map", {})
    except Exception:
        pass
    return [], {}

_idx_files, _idx_urls = _load_indexed()

_defaults = [
    ("chat_history",      _load_chat()),
    ("prediction_result", None),
    ("prediction_ctx",    ""),
    ("full_result",       None),
    ("m1_payload",        {}),
    ("gender_int",        0),
    ("age_val",           25.0),
    ("family_int",        0),
    ("exact_bmi",         None),
    ("indexer",           None),
    ("indexed_files",     _idx_files),
    ("file_url_map",      _idx_urls),
    ("index_stats",       {}),
    ("sidebar_tab",       "rag"),
    ("cfg_model",         "gemini-2.5-flash"),
    ("cfg_top_k",         4),
    ("cfg_iterations",    2),
    ("cfg_min_words",     80),
    ("cfg_max_words",     400),
    ("cfg_word_count",    500),
    ("cfg_language",      "English"),
    ("cfg_index_type",    "HNSW"),
    ("cfg_chunk_size",    400),
    ("cfg_chunk_overlap", 50),
    ("cfg_use_gateway",   False),
    ("cfg_gateway_key",   os.environ.get("GATEWAY_API_KEY", "")),
    ("cfg_gateway_url",   os.environ.get("GATEWAY_URL", "")),
    ("interrupt_requested", False),
    ("is_indexing",         False),
    ("chat_interrupt_requested", False),
    ("is_chatting",              False),
    ("_show_size_guide",         None),
    ("custom_paper_types",  []),
    ("file_assignments",    {}),
    ("web_search_enabled",  False),
    ("cfg_embed_model",     "BGE (local · 1024-dim)"),
    ("cfg_gemini_api_key",  ""),
]
for k, v in _defaults:
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# GATEWAY LLM HELPER  (CMU Andrew AI Gateway)
# Defined here — ABOVE sidebar and tab code — so it is always in scope.
# ══════════════════════════════════════════════════════════════════════════════
def _patch_indexer_for_gateway(indexer_obj, model: str, gateway_key: str,
                                gateway_url: str, word_count: int):
    """
    Monkey-patch the indexer so ALL LLM calls (generate, reflect, revise)
    route through the CMU AI Gateway instead of Gemini or a bare Anthropic key.

    - No ANTHROPIC_API_KEY required — the gateway_key IS the auth credential.
    - No GEMINI_API_KEY required — _ensure_gemini() is suppressed.
    - Reflection judge: always claude-sonnet-4-20250514-v1:0
    """
    import anthropic as _anthropic

    # Guard: catch empty key before the SDK throws a cryptic error
    if not gateway_key or not gateway_key.strip():
        raise ValueError(
            "CMU AI Gateway is enabled but no API key was provided. "
            "Enter your gateway key in the sidebar under 'Gateway API key'."
        )
    if not gateway_url or not gateway_url.strip():
        raise ValueError(
            "CMU AI Gateway is enabled but GATEWAY_URL is not set. "
            "Export GATEWAY_URL before launching Streamlit."
        )

    JUDGE_MODEL = "claude-sonnet-4-20250514-v1:0"

    def _gateway_call(prompt: str, model_name: str = model) -> str:
        client = _anthropic.Anthropic(
            api_key=gateway_key,
            base_url=gateway_url,
        )
        resp = client.messages.create(
            model=model_name,
            max_tokens=max(1024, word_count * 2),
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text if resp.content else ""

    # Patch _call_claude so generate / reflect / revise all use the gateway
    indexer_obj._call_claude = lambda prompt: _gateway_call(prompt, JUDGE_MODEL)

    # Suppress the Gemini key check — not needed in gateway mode
    indexer_obj._ensure_gemini = lambda: None

    # Store gateway metadata
    indexer_obj._gateway_call   = _gateway_call
    indexer_obj._gateway_model  = model
    indexer_obj._judge_model    = JUDGE_MODEL
    indexer_obj._word_count     = word_count
    indexer_obj._use_gateway    = True

    return indexer_obj


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI MODE HELPER
# Patches the indexer so generate/reflect/revise use Gemini directly,
# with NO Anthropic key required.
# ══════════════════════════════════════════════════════════════════════════════
def _patch_indexer_for_gemini(indexer_obj, model: str, gemini_key: str, word_count: int):
    """
    Monkey-patch the indexer so ALL LLM calls go through Gemini.
    - No ANTHROPIC_API_KEY required.
    - gemini_key is injected into the environment for this session.
    """
    if not gemini_key or not gemini_key.strip():
        raise ValueError(
            "Gemini mode requires a Gemini API key. "
            "Set GEMINI_API_KEY in your environment or enter it in the Index PDFs tab."
        )

    # Inject key into environment so LangChain picks it up
    os.environ["GEMINI_API_KEY"]  = gemini_key
    os.environ["GOOGLE_API_KEY"]  = gemini_key

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.output_parsers import StrOutputParser

    _parser = StrOutputParser()

    # Map short model name to full Gemini model string
    _GEN_MODELS = {
        "gemini-2.5-flash":      "gemini-2.5-flash",
        "gemini-2.5-pro":        "gemini-2.5-pro",
        "gemini-2.0-flash-lite": "gemini-2.0-flash-lite",
    }
    resolved = _GEN_MODELS.get(model, model)

    def _gemini_llm_call(prompt: str) -> str:
        llm = ChatGoogleGenerativeAI(
            model=resolved,
            temperature=0,
            google_api_key=gemini_key,
        )
        resp = llm.invoke(prompt)
        content = getattr(resp, "content", "")
        if isinstance(content, list):
            content = " ".join(str(x) for x in content)
        return str(content).strip()

    # Patch _call_claude to use Gemini — no Anthropic key needed
    indexer_obj._call_claude    = _gemini_llm_call
    # Suppress the Gemini key check (already handled above)
    indexer_obj._ensure_gemini  = lambda: None
    indexer_obj._use_gateway    = False

    return indexer_obj


# ══════════════════════════════════════════════════════════════════════════════
# WEB SEARCH HELPER  (Gemini grounding)
# ══════════════════════════════════════════════════════════════════════════════
def _gemini_web_search(query: str, model_name: str = "gemini-2.5-flash") -> str:
    """
    Use Gemini with Google Search grounding to fetch a brief web summary.
    Returns a plain-text snippet to prepend to the RAG query.
    Requires GEMINI_API_KEY in environment.
    """
    try:
        import google.generativeai as genai
        key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            return ""
        genai.configure(api_key=key)

        tools = [{"google_search": {}}]
        model = genai.GenerativeModel(model_name, tools=tools)
        resp  = model.generate_content(
            f"Search the web and give a concise 3-5 sentence summary relevant to: {query}"
        )
        return resp.text.strip() if hasattr(resp, "text") else ""
    except Exception as e:
        return f"[Web search failed: {e}]"


# ══════════════════════════════════════════════════════════════════════════════
# Auto-reconnect to existing Milvus collection
# Detects stored vector dimension from schema to avoid dimension mismatch.
# ══════════════════════════════════════════════════════════════════════════════
if RAG_AVAILABLE and st.session_state.indexer is None:
    try:
        from pymilvus import Collection, utility

        _reconnect_collection = None
        _reconnect_embed      = "BGE (local · 1024-dim)"

        # Check Gemini collection first (higher quality), then BGE
        for _cname, _embed_label in [
            ("papers_rag_gemini",      "Gemini embedding-001 (3072-dim)"),
            ("papers_rag_interactive", "BGE (local · 1024-dim)"),
        ]:
            try:
                if utility.has_collection(_cname):
                    _tmp = Collection(_cname)
                    if _tmp.num_entities > 0:
                        # Read actual stored dim from schema to avoid mismatch
                        for _field in _tmp.schema.fields:
                            if _field.dtype.name in ("FLOAT_VECTOR", "BINARY_VECTOR"):
                                _stored_dim = _field.params.get("dim", 0)
                                _reconnect_embed = (
                                    "Gemini embedding-001 (3072-dim)"
                                    if _stored_dim == 3072
                                    else "BGE (local · 1024-dim)"
                                )
                                break
                        _reconnect_collection = _cname
                        break
            except Exception:
                continue

        if _reconnect_collection:
            import indexer as _idx_mod
            _idx_mod.EMBED_DIM = 3072 if "Gemini" in _reconnect_embed else 1024
            st.session_state.cfg_embed_model = _reconnect_embed

            _c = PDFIndexer(
                chunk_size=st.session_state.cfg_chunk_size,
                chunk_overlap=st.session_state.cfg_chunk_overlap,
                model=st.session_state.cfg_model,
                index_type=st.session_state.cfg_index_type,
                collection_name=_reconnect_collection,
                drop_old_collection=False,
            )
            if _c._collection and _c._collection.num_entities > 0:
                _c._collection.load()
                _c._build_graph()
                st.session_state.indexer = _c
                if not st.session_state.index_stats.get("total_chunks"):
                    st.session_state.index_stats = {
                        "total_chunks": _c._collection.num_entities,
                        "total_files":  len(st.session_state.indexed_files),
                        "index_type":   st.session_state.cfg_index_type,
                        "embed_model":  _reconnect_embed,
                    }
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def call_predict(payload):
    try:
        r = http_requests.post(f"{OBESITY_API}/predict", json=payload, timeout=10)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def _unique_papers(sources, url_map, k=2):
    seen, out = set(), []
    for s in sorted(sources, key=lambda x: float(x.get("score", 0)), reverse=True):
        n = s.get("source", "")
        if not n or n in seen:
            continue
        seen.add(n)
        out.append({"paper": n, "url": url_map.get(n, ""),
                    "paper_type": s.get("paper_type", ""),
                    "best_score": float(s.get("score", 0))})
        if len(out) >= k:
            break
    return out


def _render_recs(recs, url_trace=None):
    if not recs:
        return
    st.markdown('<div style="font-size:.8rem;font-weight:600;color:#9b9b9b;margin:12px 0 6px 0;">Recommended Papers</div>', unsafe_allow_html=True)
    for i, r in enumerate(recs, 1):
        url   = r.get("url", "")
        uhtml = f'<div class="purl">{url}</div>' if url else '<div class="pscore">No URL available</div>'
        st.markdown(
            f'<div class="paper-card"><div class="ptitle">{i}. {r["paper"]}'
            f'  <span class="badge badge-type">{r.get("paper_type","")}</span></div>'
            f'{uhtml}<div class="pscore">score: {r["best_score"]:.4f}</div></div>',
            unsafe_allow_html=True,
        )
    if url_trace:
        for t in url_trace:
            s   = t.get("status", "")
            css = "reflect-pass" if s == "PASS" else "reflect-fixed" if s == "FIXED" else "reflect-fail"
            st.markdown(
                f'<div class="{css}"><strong>{t.get("paper","")}</strong> — {s}<br>'
                f'Original: {t.get("original_url","") or "(none)"}<br>'
                f'Final: {t.get("final_url","") or "(none)"}</div>',
                unsafe_allow_html=True,
            )


def _render_sources(sources):
    if not sources:
        return
    st.markdown(f'<div style="font-size:.76rem;font-weight:600;color:#9b9b9b;margin:12px 0 6px 0;">{len(sources)} retrieved chunks</div>', unsafe_allow_html=True)
    for s in sources:
        url  = st.session_state.file_url_map.get(s["source"], "")
        ul   = f"<br><span class='badge badge-url'>URL</span> {url}" if url else ""
        st.markdown(
            f'<div class="chunk-preview">'
            f'<span class="badge">{s["source"]}</span>'
            f'<span class="badge">p{s["page"]}</span>'
            f'<span class="badge badge-type">{s.get("paper_type","?")}</span>'
            f'<span class="badge badge-score">score {s["score"]:.3f}</span>'
            f'{ul}<br><br>{s["text"][:500]}{"..." if len(s["text"])>500 else ""}</div>',
            unsafe_allow_html=True,
        )


def render_top3(result):
    top3  = result.get("top3", [])
    comps = result.get("score_components", {})
    if not top3:
        return

    for i, item in enumerate(top3):
        cls   = item["class"]
        prob  = item["probability"] * 100
        conf  = item["confidence_pct"]
        tier  = item["tier"]
        cc    = CLASS_COLOR.get(cls, "#9b9b9b")
        tc    = TIER_COLOR.get(tier, "#9b9b9b")
        desc  = CLASS_DESC.get(cls, "")
        label = cls.replace("_", " ")
        badges = ["🥇 PRIMARY PREDICTION", "🥈 RUNNER-UP", "🥉 ALTERNATIVE"]

        st.markdown(f"""
        <div class="result-card" style="border-left:4px solid {cc};{'opacity:.78;' if i>0 else ''}">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;">
                <div>
                    <div style="font-size:.63rem;text-transform:uppercase;letter-spacing:.1em;color:#555;margin-bottom:4px;">{badges[i]}</div>
                    <div style="font-size:{'1.5rem' if i==0 else '1.15rem'};font-weight:700;color:{cc};letter-spacing:-.02em;">{label}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:{'1.4rem' if i==0 else '1.1rem'};font-weight:700;color:{tc};">{conf:.1f}%</div>
                    <div style="font-size:.63rem;color:#555;margin-top:2px;">confidence</div>
                </div>
            </div>
            <div style="margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;font-size:.7rem;color:#555;margin-bottom:4px;">
                    <span>Model probability</span><span>{prob:.1f}%</span>
                </div>
                <div style="background:#1a1a1a;border-radius:4px;height:6px;overflow:hidden;border:1px solid #2f2f2f;">
                    <div style="width:{prob}%;height:100%;background:{cc};border-radius:4px;"></div>
                </div>
                <div style="font-size:.68rem;color:{tc};margin-top:4px;">{tier}</div>
            </div>
            <div style="font-size:.78rem;color:#9b9b9b;line-height:1.55;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

        if i == 0 and comps:
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;margin:-4px 0 12px 0;">
                <div style="background:#1a1a1a;border:1px solid #2f2f2f;border-radius:6px;padding:10px;text-align:center;">
                    <div style="font-size:.58rem;text-transform:uppercase;letter-spacing:.08em;color:#555;margin-bottom:3px;">x · Cosine Sim</div>
                    <div style="font-size:1rem;font-weight:600;color:#ececec;">{comps.get('x_cosine_sim', 0):.3f}</div>
                </div>
                <div style="background:#1a1a1a;border:1px solid #2f2f2f;border-radius:6px;padding:10px;text-align:center;">
                    <div style="font-size:.58rem;text-transform:uppercase;letter-spacing:.08em;color:#555;margin-bottom:3px;">y · Model Prob</div>
                    <div style="font-size:1rem;font-weight:600;color:#ececec;">{comps.get('y_model_proba', 0):.3f}</div>
                </div>
                <div style="background:#1a1a1a;border:1px solid #2f2f2f;border-radius:6px;padding:10px;text-align:center;">
                    <div style="font-size:.58rem;text-transform:uppercase;letter-spacing:.08em;color:#555;margin-bottom:3px;">z · Confusion Risk</div>
                    <div style="font-size:1rem;font-weight:600;color:#ececec;">{comps.get('z_confusion_risk', 0):.3f}</div>
                </div>
                <div style="background:#1a1a1a;border:1px solid #2f2f2f;border-radius:6px;padding:10px;text-align:center;">
                    <div style="font-size:.58rem;text-transform:uppercase;letter-spacing:.08em;color:#555;margin-bottom:3px;">m · Demographic</div>
                    <div style="font-size:1rem;font-weight:600;color:#ececec;">{comps.get('m_demographic', 0):.3f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def build_ctx(result, inputs):
    cls  = result["predicted_class"].replace("_", " ")
    conf = result["confidence_pct"]
    bmi_lbl  = {0:"underweight (BMI<18.5)", 1:"normal weight (BMI 18-25)", 2:"overweight (BMI 25-30)", 3:"obese (BMI>30)"}
    caec_lbl = {0:"no", 1:"sometimes", 2:"frequently", 3:"always"}
    calc_lbl = {0:"never", 1:"sometimes", 2:"frequently", 3:"always"}
    fcvc_lbl = {1:"never", 2:"sometimes", 3:"always"}
    ch2o_lbl = {1:"<1L/day", 2:"1-2L/day", 3:">2L/day"}
    tue_lbl  = {0:"0-2h", 1:"3-5h", 2:"5h+"}
    top3_str = " | ".join(f"{t['class'].replace('_',' ')} ({t['confidence_pct']:.1f}%)"
                           for t in result.get("top3", []))
    return (
        f"Patient: {'Male' if inputs['gender']==1 else 'Female'}, age {inputs['age']}, "
        f"family history {('yes' if inputs['family_history']==1 else 'no')}. "
        f"BMI category: {bmi_lbl.get(inputs['bmi_bucket'],'unknown')}. "
        f"Veg consumption: {fcvc_lbl.get(int(inputs['fcvc']),inputs['fcvc'])}. "
        f"Meals/day: {inputs['ncp']}. Snacking: {caec_lbl.get(inputs['caec'],inputs['caec'])}. "
        f"Water: {ch2o_lbl.get(int(inputs['ch2o']),inputs['ch2o'])}. "
        f"Activity: {inputs['faf']} days/week. Screen time: {tue_lbl.get(int(inputs['tue']),inputs['tue'])}. "
        f"Alcohol: {calc_lbl.get(inputs['calc'],inputs['calc'])}. "
        f"Classified as: {cls} ({conf:.1f}% confidence). Top 3: {top3_str}."
    )


def render_msg(msg):
    if msg["role"] == "user":
        st.markdown(
            f'<div style="display:flex;align-items:flex-start;gap:10px;margin:14px 0;">'
            f'<div style="width:30px;height:30px;border-radius:50%;background:#e53935;'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-size:.7rem;font-weight:700;color:#fff;flex-shrink:0;">You</div>'
            f'<div style="color:#ececec;font-size:.93rem;line-height:1.65;padding-top:3px;">'
            f'{msg["content"]}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="display:flex;align-items:flex-start;gap:10px;margin:14px 0;">'
            f'<div style="width:30px;height:30px;border-radius:50%;background:#2a2a2a;'
            f'border:1px solid #333;display:flex;align-items:center;justify-content:center;'
            f'font-size:.65rem;font-weight:700;color:#9b9b9b;flex-shrink:0;">AI</div>'
            f'<div style="color:#ececec;font-size:.93rem;line-height:1.65;padding-top:3px;width:100%;">',
            unsafe_allow_html=True,
        )
        st.markdown(msg["content"])
        if msg.get("recs"):
            _render_recs(msg["recs"])
        if msg.get("sources"):
            _render_sources(msg["sources"])
        st.markdown("</div></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:0 16px 12px 16px;text-align:center;">
        <div style="display:inline-block;background:#050505;border:1px solid #00eaff;
                    border-radius:10px;padding:12px 20px;margin-bottom:12px;
                    box-shadow:0 0 12px rgba(0,234,255,.4),0 0 28px rgba(0,234,255,.15);">
            <div style="font-size:1.1rem;font-weight:600;color:#00eaff;letter-spacing:.12em;
                        text-transform:uppercase;font-family:'Georgia',serif;
                        text-shadow:0 0 8px rgba(0,234,255,.8);line-height:1.5;">
                Devavrath<br>Sandeep's
            </div>
        </div>
    </div>
    <div style="text-align:center;font-size:1.15rem;font-weight:700;letter-spacing:-.02em;
                padding-bottom:16px;border-bottom:1px solid #2f2f2f;color:#ececec;">
        Obesity Classifier
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

    for tid, tlabel in [("rag", "RAG Config"), ("files", "Indexed Files")]:
        active = st.session_state.sidebar_tab == tid
        if st.button(tlabel, key=f"vtab_{tid}", use_container_width=True,
                     type="primary" if active else "secondary"):
            st.session_state.sidebar_tab = tid
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    if st.button("Reset All", use_container_width=True, key="reset_all"):
        for k, v in [("chat_history", []), ("prediction_result", None),
                     ("prediction_ctx", ""), ("full_result", None),
                     ("m1_payload", {}), ("indexer", None),
                     ("indexed_files", []), ("file_url_map", {}), ("index_stats", {})]:
            st.session_state[k] = v
        _save_chat(); _save_indexed(); st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    def _sblabel(t):
        st.markdown(f'<div style="font-size:.63rem;font-weight:600;letter-spacing:.1em;'
                    f'text-transform:uppercase;color:#666;padding:8px 0 4px 0;">{t}</div>',
                    unsafe_allow_html=True)

    if st.session_state.sidebar_tab == "rag":

        # ── API Backend ───────────────────────────────────────────────────────
        _sblabel("API Backend")
        use_gw = st.checkbox(
            "Use CMU AI Gateway",
            value=st.session_state.cfg_use_gateway,
            key="sb_use_gateway",
            help="Routes generation through the CMU Andrew AI Gateway. "
                 "Requires your CMU gateway API key. No Gemini key needed.",
        )
        st.session_state.cfg_use_gateway = use_gw

        if use_gw:
            gw_key = st.text_input(
                "Gateway API key", value=st.session_state.cfg_gateway_key,
                type="password", placeholder="sk-...",
                key="sb_gateway_key", label_visibility="collapsed",
            )
            st.session_state.cfg_gateway_key = gw_key
            if gw_key:
                st.markdown('<div style="font-size:.68rem;color:#34d399;padding-bottom:4px;">✓ Key set</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div style="font-size:.68rem;color:#f87171;padding-bottom:4px;">⚠ Key required</div>',
                            unsafe_allow_html=True)

            # Gateway model list
            _sblabel("Generation Model")
            GATEWAY_MODELS = [
                "claude-sonnet-4-20250514-v1:0",
                "claude-haiku-4-5-20251001-v1:0",
                "claude-opus-4-20250514-v1:0",
                "gemini-2.5-flash",
                "gpt-5.4-pro",
                "gpt-5",
            ]
            if st.session_state.cfg_model not in GATEWAY_MODELS:
                st.session_state.cfg_model = GATEWAY_MODELS[0]
            st.session_state.cfg_model = st.selectbox(
                "gw_model", GATEWAY_MODELS,
                index=GATEWAY_MODELS.index(st.session_state.cfg_model),
                label_visibility="collapsed",
            )
            st.markdown(
                '<div style="font-size:.65rem;color:#555;padding-bottom:4px;">'
                '🔒 Reflection judge: always Claude Sonnet</div>',
                unsafe_allow_html=True,
            )
        else:
            # Standard Gemini models — key from env or Index PDFs tab
            _sblabel("Generation Model")
            GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash-lite"]
            if st.session_state.cfg_model not in GEMINI_MODELS:
                st.session_state.cfg_model = GEMINI_MODELS[0]
            st.session_state.cfg_model = st.selectbox(
                "gem_model", GEMINI_MODELS,
                index=GEMINI_MODELS.index(st.session_state.cfg_model),
                label_visibility="collapsed",
            )
            # Show Gemini key status
            _gkey = (os.environ.get("GEMINI_API_KEY", "")
                     or os.environ.get("GOOGLE_API_KEY", "")
                     or st.session_state.cfg_gemini_api_key)
            if _gkey:
                st.markdown('<div style="font-size:.68rem;color:#34d399;padding-bottom:4px;">✓ Gemini key ready</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div style="font-size:.68rem;color:#fbbf24;padding-bottom:4px;">'
                            '⚠ Enter Gemini key in Index PDFs tab</div>',
                            unsafe_allow_html=True)

        # ── Retrieval ─────────────────────────────────────────────────────────
        _sblabel("Retrieval")
        st.session_state.cfg_top_k      = st.slider("Top-K chunks",          1, 10, st.session_state.cfg_top_k)
        st.session_state.cfg_iterations = st.slider("Reflection iterations", 1,  4, st.session_state.cfg_iterations)

        # ── Response length ───────────────────────────────────────────────────
        _sblabel("Response Length (words)")
        st.session_state.cfg_word_count = st.slider(
            "words", 300, 2000, st.session_state.cfg_word_count, 50,
            label_visibility="collapsed",
        )

        # ── Index algorithm ───────────────────────────────────────────────────
        _sblabel("Index Algorithm")
        st.session_state.cfg_index_type = st.selectbox(
            "idx", ["HNSW","IVF_PQ","DiskANN"],
            index=["HNSW","IVF_PQ","DiskANN"].index(st.session_state.cfg_index_type),
            label_visibility="collapsed",
        )

        # ── Status ────────────────────────────────────────────────────────────
        if not RAG_AVAILABLE:
            st.markdown('<div style="font-size:.72rem;color:#555;padding:8px 0;">indexer.py not found — RAG disabled</div>', unsafe_allow_html=True)
        elif st.session_state.indexer is None:
            st.markdown('<div style="font-size:.72rem;color:#fbbf24;padding:8px 0;">No Milvus collection found.<br>Index PDFs in the Index PDFs tab first.</div>', unsafe_allow_html=True)
        else:
            s = st.session_state.index_stats
            st.markdown(
                f'<div style="font-size:.72rem;color:#34d399;padding:8px 0;">'
                f'✓ Connected · {s.get("total_chunks",0)} chunks · '
                f'{s.get("embed_model", EMBED_MODEL).split("/")[-1]}</div>',
                unsafe_allow_html=True,
            )

    elif st.session_state.sidebar_tab == "files":
        if not st.session_state.indexed_files:
            st.markdown('<div style="font-size:.78rem;color:#555;padding:12px 0;">No files indexed yet.</div>', unsafe_allow_html=True)
        else:
            s = st.session_state.index_stats
            st.markdown(f"""
            <div style="display:flex;gap:6px;padding:8px 0 12px 0;">
                <div style="flex:1;background:#2a2a2a;border:1px solid #333;border-radius:8px;padding:10px 6px;text-align:center;">
                    <div style="font-size:1rem;font-weight:600;color:#ececec;">{len(st.session_state.indexed_files)}</div>
                    <div style="font-size:.58rem;text-transform:uppercase;color:#666;margin-top:3px;">files</div>
                </div>
                <div style="flex:1;background:#2a2a2a;border:1px solid #333;border-radius:8px;padding:10px 6px;text-align:center;">
                    <div style="font-size:1rem;font-weight:600;color:#ececec;">{s.get('total_chunks',0)}</div>
                    <div style="font-size:.58rem;text-transform:uppercase;color:#666;margin-top:3px;">chunks</div>
                </div>
            </div>""", unsafe_allow_html=True)
            for name, ptype in st.session_state.indexed_files:
                st.markdown(
                    f'<div style="background:#2a2a2a;border:1px solid #333;border-radius:6px;'
                    f'padding:6px 10px;font-size:.71rem;color:#bbb;margin:2px 0;'
                    f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">'
                    f'{name} <span style="background:#1e1533;color:#a78bfa;padding:1px 5px;'
                    f'border-radius:3px;font-size:.58rem;">{ptype}</span></div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_assess, tab_chat, tab_index = st.tabs(["Assessment", "Nutrition Plan", "Index PDFs"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Assessment form
# ══════════════════════════════════════════════════════════════════════════════
with tab_assess:
    st.markdown("""
    <div style="text-align:center;padding:36px 0 24px 0;">
        <h1 style="font-size:1.9rem;font-weight:600;color:#ececec;letter-spacing:-.03em;margin:0 0 6px 0;">
            Obesity Risk Assessment</h1>
        <p style="font-size:.75rem;color:#666;letter-spacing:.06em;text-transform:uppercase;margin:0;">
            Random Forest · 12 features · 7-class classification</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("assessment_form", clear_on_submit=False):

        # Section 1 — Demographics
        st.markdown("## Demographics")
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.radio("Gender", ["Female", "Male"], horizontal=True,
                              help="Biological sex at birth")
        with c2:
            age = st.number_input("Age (years)", min_value=10, max_value=100,
                                  value=25, step=1)
        with c3:
            family_history = st.radio(
                "Family history of overweight?", ["No", "Yes"], horizontal=True,
                help="Has a parent, sibling or grandparent been overweight or obese?",
            )

        st.markdown("<hr>", unsafe_allow_html=True)

        # Section 2 — Weight category
        st.markdown("## Weight Category")
        st.caption("Enter your height and weight to calculate BMI, or estimate your category directly below")

        hw1, hw2, hw3 = st.columns([2, 2, 3])
        with hw1:
            height_cm = st.number_input(
                "Height (cm)", min_value=100, max_value=250, value=170, step=1,
                help="Your height in centimetres",
            )
        with hw2:
            weight_kg = st.number_input(
                "Weight (kg)", min_value=30, max_value=300, value=70, step=1,
                help="Your weight in kilograms",
            )
        with hw3:
            bmi_val = weight_kg / ((height_cm / 100) ** 2)
            if bmi_val < 18.5:
                bmi_cat = "Underweight"; bmi_col = "#60a5fa"; auto_bmi = 0
            elif bmi_val < 25:
                bmi_cat = "Normal weight"; bmi_col = "#34d399"; auto_bmi = 1
            elif bmi_val < 30:
                bmi_cat = "Overweight"; bmi_col = "#fbbf24"; auto_bmi = 2
            else:
                bmi_cat = "Obese"; bmi_col = "#ef4444"; auto_bmi = 3
            st.markdown(f"""
            <div style="background:#1a1a1a;border:1px solid #2f2f2f;border-radius:8px;
                        padding:14px 18px;margin-top:22px;">
                <div style="font-size:.62rem;text-transform:uppercase;letter-spacing:.08em;
                            color:#555;margin-bottom:4px;">Calculated BMI</div>
                <div style="font-size:1.6rem;font-weight:700;color:{bmi_col};line-height:1;">
                    {bmi_val:.1f}
                </div>
                <div style="font-size:.75rem;color:{bmi_col};margin-top:3px;">{bmi_cat}</div>
            </div>
            """, unsafe_allow_html=True)

        st.caption("The BMI category below is pre-selected based on your height and weight — you can override it if needed")

        bmi_options = [
            "Underweight — I weigh less than is healthy for my height  (BMI < 18.5)",
            "Normal weight — My weight is in the healthy range  (BMI 18.5 – 25)",
            "Overweight — I weigh a bit more than is ideal  (BMI 25 – 30)",
            "Obese — My weight is significantly above the healthy range  (BMI > 30)",
        ]
        bmi_bucket = st.radio(
            "Which best describes your current weight?",
            bmi_options,
            index=auto_bmi,
        )
        bmi_map = {
            "Underweight — I weigh less than is healthy for my height  (BMI < 18.5)": 0,
            "Normal weight — My weight is in the healthy range  (BMI 18.5 – 25)":     1,
            "Overweight — I weigh a bit more than is ideal  (BMI 25 – 30)":           2,
            "Obese — My weight is significantly above the healthy range  (BMI > 30)":  3,
        }

        st.markdown("<hr>", unsafe_allow_html=True)

        # Section 3 — Diet
        st.markdown("## Diet & Eating Habits")
        d1, d2 = st.columns(2)
        with d1:
            fcvc = st.radio(
                "How often do you eat vegetables with meals?",
                ["Never — I rarely eat vegetables  (1)",
                 "Sometimes — I include vegetables in some meals  (2)",
                 "Always — Vegetables are part of most meals  (3)"],
                index=1, help="Salads, cooked veg, raw veg etc.",
            )
            fcvc_map = {
                "Never — I rarely eat vegetables  (1)":                1.0,
                "Sometimes — I include vegetables in some meals  (2)": 2.0,
                "Always — Vegetables are part of most meals  (3)":     3.0,
            }
            caec = st.radio(
                "Do you eat snacks or food between main meals?",
                ["No — I only eat at mealtimes",
                 "Sometimes — I occasionally have a snack",
                 "Frequently — I snack most days",
                 "Always — I snack throughout the day"],
                index=1, help="Includes biscuits, fruit, crisps, sweets etc.",
            )
            caec_map = {
                "No — I only eat at mealtimes":            0,
                "Sometimes — I occasionally have a snack": 1,
                "Frequently — I snack most days":          2,
                "Always — I snack throughout the day":     3,
            }
        with d2:
            ncp = st.radio(
                "How many main meals do you eat per day?",
                ["1 meal per day", "2 meals per day",
                 "3 meals per day  (typical)", "4 or more meals per day"],
                index=2, help="Breakfast, lunch and dinner count as separate meals",
            )
            ncp_map = {
                "1 meal per day":             1.0,
                "2 meals per day":            2.0,
                "3 meals per day  (typical)": 3.0,
                "4 or more meals per day":    4.0,
            }
            calc = st.radio(
                "How often do you drink alcohol?",
                ["Never",
                 "Sometimes — a few times a month",
                 "Frequently — several times a week",
                 "Always — daily"],
                index=1,
            )
            calc_map = {
                "Never":                              0,
                "Sometimes — a few times a month":   1,
                "Frequently — several times a week": 2,
                "Always — daily":                    3,
            }

        st.markdown("<hr>", unsafe_allow_html=True)

        # Section 4 — Lifestyle
        st.markdown("## Lifestyle")
        l1, l2, l3 = st.columns(3)
        with l1:
            faf = st.radio(
                "How many days per week are you physically active?",
                ["None — I do not exercise  (0)",
                 "1–2 days per week  (1)",
                 "3–4 days per week  (2)",
                 "5 or more days per week  (3)"],
                index=1,
                help="Walking, gym, sport, cycling — anything that raises your heart rate",
            )
            faf_map = {
                "None — I do not exercise  (0)":  0.0,
                "1–2 days per week  (1)":         1.0,
                "3–4 days per week  (2)":         2.0,
                "5 or more days per week  (3)":   3.0,
            }
        with l2:
            ch2o = st.radio(
                "How much water do you drink per day?",
                ["Less than 1 litre  (1)",
                 "Between 1 and 2 litres  (2)",
                 "More than 2 litres  (3)"],
                index=1, help="Plain water — not sugary drinks or coffee",
            )
            ch2o_map = {
                "Less than 1 litre  (1)":      1.0,
                "Between 1 and 2 litres  (2)": 2.0,
                "More than 2 litres  (3)":     3.0,
            }
        with l3:
            tue = st.radio(
                "Hours per day on screens or tech?",
                ["0 – 2 hours  (low)",
                 "3 – 5 hours  (moderate)",
                 "More than 5 hours  (high)"],
                index=1, help="Phone, computer, TV — combined daily total",
            )
            tue_map = {
                "0 – 2 hours  (low)":        0.0,
                "3 – 5 hours  (moderate)":   1.0,
                "More than 5 hours  (high)": 2.0,
            }

        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "Run Assessment →", type="primary", use_container_width=True,
        )

    # ── On submit ────────────────────────────────────────────────────────────
    if submitted:
        gender_int     = 1 if gender == "Male" else 0
        family_int     = 1 if family_history == "Yes" else 0
        bmi_bucket_int = bmi_map[bmi_bucket]

        m1_payload = {
            "gender": gender_int, "age": float(age),
            "family_history": family_int,
            "fcvc": fcvc_map[fcvc], "ncp": ncp_map[ncp],
            "caec": caec_map[caec], "ch2o": ch2o_map[ch2o],
            "faf":  faf_map[faf],  "tue":  tue_map[tue],
            "calc": calc_map[calc], "bmi_bucket": bmi_bucket_int,
        }
        with st.spinner("Running Model 1 inference..."):
            m1_result, err = call_predict(m1_payload)

        if err:
            st.error(f"API error: {err}")
            st.info(f"Make sure FastAPI is running at `{OBESITY_API}`\n\n"
                    f"`uvicorn main:app --reload --port 8001`")
        else:
            st.session_state.prediction_result  = m1_result
            st.session_state.m1_payload         = m1_payload
            st.session_state.gender_int         = gender_int
            st.session_state.age_val            = float(age)
            st.session_state.family_int         = family_int
            st.session_state.exact_bmi          = round(bmi_val, 2)
            st.session_state.full_result        = None
            st.session_state.prediction_ctx     = build_ctx(m1_result, m1_payload)
            st.session_state.chat_history       = []
            _save_chat()

    # ── Results block ─────────────────────────────────────────────────────────
    if st.session_state.prediction_result:
        m1r = st.session_state.prediction_result
        ow_classes = {"Overweight_Level_I", "Overweight_Level_II"}

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("## Results")

        m1_top       = m1r.get("top3", [{}])[0]
        m1_cls       = m1_top.get("class", "")
        m1_conf      = m1_top.get("confidence_pct", 100)
        m1_runner    = m1r.get("top3", [{}, {}])[1].get("class", "") if len(m1r.get("top3", [])) > 1 else ""
        needs_m2     = (m1_cls in ow_classes and
                        (m1_conf < 75 or m1_runner in ow_classes))
        full_r       = st.session_state.get("full_result")

        render_top3(m1r)

        if needs_m2 and full_r is None:
            st.markdown("""
            <div style="background:#1f1a0f;border:1px solid #78350f;border-radius:10px;
                        padding:18px 22px;margin:18px 0;">
                <div style="font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;
                            color:#f59e0b;margin-bottom:6px;">⚡ Clinical Escalation Available</div>
                <div style="font-size:.88rem;color:#d4b57a;line-height:1.65;margin-bottom:12px;">
                    Model 1 flagged a close call between
                    <strong style="color:#fbbf24;">Overweight Level I</strong> and
                    <strong style="color:#f97316;">Overweight Level II</strong>
                    (confidence {conf:.1f}%). Providing clinical measurements unlocks
                    Model 2 — trained on {n:,} NHANES participants — for a refined classification.
                    Blood glucose / HbA1c also enables a diabetes risk assessment.
                </div>
            </div>
            """.format(conf=m1_conf, n=2289), unsafe_allow_html=True)

            with st.form("clinical_form"):
                _m2_hdr_col, _m2_info_col = st.columns([5, 1])
                with _m2_hdr_col:
                    st.markdown("### Clinical Measurements")
                    st.caption("All fields optional — provide what you have. More inputs = higher confidence.")
                with _m2_info_col:
                    st.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)
                    show_size_guide = st.form_submit_button(
                        "ℹ️ Size guide", help="Show general body measurement reference charts"
                    )

                if show_size_guide:
                    _gender_for_guide = "Male" if st.session_state.get("gender_int", 0) == 1 else "Female"
                    st.session_state["_show_size_guide"] = _gender_for_guide

                if st.session_state.get("_show_size_guide"):
                    _g = st.session_state["_show_size_guide"]
                    _other = "Female" if _g == "Male" else "Male"
                    _gc1, _gc2 = st.columns(2)
                    with _gc1:
                        st.markdown(f"**{_g} — General Size Reference**")
                        if _g == "Male":
                            st.markdown("""
<div style="background:#1a1a1a;border:1px solid #2f2f2f;border-radius:8px;padding:12px 16px;font-size:.78rem;line-height:1.9;color:#9b9b9b;">
<div style="font-weight:600;color:#60a5fa;margin-bottom:6px;">👕 T-Shirt / Top</div>
<div><span style="color:#ececec;font-weight:600;min-width:32px;display:inline-block;">XS</span> Chest 32–34″ · Waist 26–28″ · <strong style="color:#fbbf24;">Waist cm ≈ 66–71</strong></div>
<div><span style="color:#ececec;font-weight:600;min-width:32px;display:inline-block;">S</span>  Chest 35–37″ · Waist 29–31″ · <strong style="color:#fbbf24;">Waist cm ≈ 74–79</strong></div>
<div><span style="color:#ececec;font-weight:600;min-width:32px;display:inline-block;">M</span>  Chest 38–40″ · Waist 32–34″ · <strong style="color:#fbbf24;">Waist cm ≈ 81–86</strong></div>
<div><span style="color:#ececec;font-weight:600;min-width:32px;display:inline-block;">L</span>  Chest 41–43″ · Waist 35–37″ · <strong style="color:#fbbf24;">Waist cm ≈ 89–94</strong></div>
<div><span style="color:#ececec;font-weight:600;min-width:32px;display:inline-block;">XL</span> Chest 44–46″ · Waist 38–40″ · <strong style="color:#fbbf24;">Waist cm ≈ 97–102</strong></div>
<div><span style="color:#ececec;font-weight:600;min-width:32px;display:inline-block;">2XL</span>Chest 47–49″ · Waist 41–43″ · <strong style="color:#fbbf24;">Waist cm ≈ 104–109</strong></div>
<div style="margin-top:8px;font-weight:600;color:#34d399;">📏 Hip guide</div>
<div>S/M ≈ 86–94 cm · L/XL ≈ 96–104 cm · 2XL+ ≈ 106+ cm</div>
<div style="margin-top:8px;font-weight:600;color:#a78bfa;">💪 Mid-upper arm</div>
<div>S ≈ 28–30 cm · M ≈ 31–33 cm · L ≈ 34–36 cm · XL ≈ 37–39 cm</div>
</div>""", unsafe_allow_html=True)
                        else:
                            st.markdown("""
<div style="background:#1a1a1a;border:1px solid #2f2f2f;border-radius:8px;padding:12px 16px;font-size:.78rem;line-height:1.9;color:#9b9b9b;">
<div style="font-weight:600;color:#f472b6;margin-bottom:6px;">👕 T-Shirt / Top</div>
<div><span style="color:#ececec;font-weight:600;min-width:32px;display:inline-block;">XS</span> Chest 31–33″ · Waist 23–25″ · <strong style="color:#fbbf24;">Waist cm ≈ 58–64</strong></div>
<div><span style="color:#ececec;font-weight:600;min-width:32px;display:inline-block;">S</span>  Chest 34–36″ · Waist 26–28″ · <strong style="color:#fbbf24;">Waist cm ≈ 66–71</strong></div>
<div><span style="color:#ececec;font-weight:600;min-width:32px;display:inline-block;">M</span>  Chest 37–39″ · Waist 29–31″ · <strong style="color:#fbbf24;">Waist cm ≈ 74–79</strong></div>
<div><span style="color:#ececec;font-weight:600;min-width:32px;display:inline-block;">L</span>  Chest 40–42″ · Waist 32–34″ · <strong style="color:#fbbf24;">Waist cm ≈ 81–86</strong></div>
<div><span style="color:#ececec;font-weight:600;min-width:32px;display:inline-block;">XL</span> Chest 43–45″ · Waist 35–37″ · <strong style="color:#fbbf24;">Waist cm ≈ 89–94</strong></div>
<div><span style="color:#ececec;font-weight:600;min-width:32px;display:inline-block;">2XL</span>Chest 46–48″ · Waist 38–40″ · <strong style="color:#fbbf24;">Waist cm ≈ 97–102</strong></div>
<div style="margin-top:8px;font-weight:600;color:#34d399;">📏 Hip guide</div>
<div>S/M ≈ 84–94 cm · L/XL ≈ 96–106 cm · 2XL+ ≈ 108+ cm</div>
<div style="margin-top:8px;font-weight:600;color:#a78bfa;">💪 Mid-upper arm</div>
<div>XS/S ≈ 24–27 cm · M ≈ 28–30 cm · L ≈ 31–33 cm · XL ≈ 34–36 cm</div>
</div>""", unsafe_allow_html=True)
                    with _gc2:
                        st.markdown(f"**{_other} — General Size Reference**")
                        if _other == "Male":
                            st.markdown("""
<div style="background:#1a1a1a;border:1px solid #2f2f2f;border-radius:8px;padding:12px 16px;font-size:.78rem;line-height:1.9;color:#9b9b9b;">
<div style="font-weight:600;color:#60a5fa;margin-bottom:6px;">👕 T-Shirt / Top</div>
<div><span style="color:#ececec;font-weight:600;min-width:32px;display:inline-block;">XS</span> Chest 32–34″ · Waist 26–28″ · <strong style="color:#fbbf24;">Waist cm ≈ 66–71</strong></div>
<div><span style="color:#ececec;font-weight:600;min-width:32px;display:inline-block;">S</span>  Chest 35–37″ · Waist 29–31″ · <strong style="color:#fbbf24;">Waist cm ≈ 74–79</strong></div>
<div><span style="color:#ececec;font-weight:600;min-width:32px;display:inline-block;">M</span>  Chest 38–40″ · Waist 32–34″ · <strong style="color:#fbbf24;">Waist cm ≈ 81–86</strong></div>
<div><span style="color:#ececec;font-weight:600;min-width:32px;display:inline-block;">L</span>  Chest 41–43″ · Waist 35–37″ · <strong style="color:#fbbf24;">Waist cm ≈ 89–94</strong></div>
<div><span style="color:#ececec;font-weight:600;min-width:32px;display:inline-block;">XL</span> Chest 44–46″ · Waist 38–40″ · <strong style="color:#fbbf24;">Waist cm ≈ 97–102</strong></div>
<div><span style="color:#ececec;font-weight:600;min-width:32px;display:inline-block;">2XL</span>Chest 47–49″ · Waist 41–43″ · <strong style="color:#fbbf24;">Waist cm ≈ 104–109</strong></div>
<div style="margin-top:8px;font-weight:600;color:#34d399;">📏 Hip guide</div>
<div>S/M ≈ 86–94 cm · L/XL ≈ 96–104 cm · 2XL+ ≈ 106+ cm</div>
<div style="margin-top:8px;font-weight:600;color:#a78bfa;">💪 Mid-upper arm</div>
<div>S ≈ 28–30 cm · M ≈ 31–33 cm · L ≈ 34–36 cm · XL ≈ 37–39 cm</div>
</div>""", unsafe_allow_html=True)
                        else:
                            st.markdown("""
<div style="background:#1a1a1a;border:1px solid #2f2f2f;border-radius:8px;padding:12px 16px;font-size:.78rem;line-height:1.9;color:#9b9b9b;">
<div style="font-weight:600;color:#f472b6;margin-bottom:6px;">👕 T-Shirt / Top</div>
<div><span style="color:#ececec;font-weight:600;min-width:32px;display:inline-block;">XS</span> Chest 31–33″ · Waist 23–25″ · <strong style="color:#fbbf24;">Waist cm ≈ 58–64</strong></div>
<div><span style="color:#ececec;font-weight:600;min-width:32px;display:inline-block;">S</span>  Chest 34–36″ · Waist 26–28″ · <strong style="color:#fbbf24;">Waist cm ≈ 66–71</strong></div>
<div><span style="color:#ececec;font-weight:600;min-width:32px;display:inline-block;">M</span>  Chest 37–39″ · Waist 29–31″ · <strong style="color:#fbbf24;">Waist cm ≈ 74–79</strong></div>
<div><span style="color:#ececec;font-weight:600;min-width:32px;display:inline-block;">L</span>  Chest 40–42″ · Waist 32–34″ · <strong style="color:#fbbf24;">Waist cm ≈ 81–86</strong></div>
<div><span style="color:#ececec;font-weight:600;min-width:32px;display:inline-block;">XL</span> Chest 43–45″ · Waist 35–37″ · <strong style="color:#fbbf24;">Waist cm ≈ 89–94</strong></div>
<div><span style="color:#ececec;font-weight:600;min-width:32px;display:inline-block;">2XL</span>Chest 46–48″ · Waist 38–40″ · <strong style="color:#fbbf24;">Waist cm ≈ 97–102</strong></div>
<div style="margin-top:8px;font-weight:600;color:#34d399;">📏 Hip guide</div>
<div>S/M ≈ 84–94 cm · L/XL ≈ 96–106 cm · 2XL+ ≈ 108+ cm</div>
<div style="margin-top:8px;font-weight:600;color:#a78bfa;">💪 Mid-upper arm</div>
<div>XS/S ≈ 24–27 cm · M ≈ 28–30 cm · L ≈ 31–33 cm · XL ≈ 34–36 cm</div>
</div>""", unsafe_allow_html=True)
                    st.markdown(
                        '<div style="font-size:.7rem;color:#555;padding:6px 0 4px 0;">'
                        '* These are general ready-to-wear size averages. '
                        'Measure at the widest points using a soft tape measure held snug but not tight.</div>',
                        unsafe_allow_html=True,
                    )
                    if st.form_submit_button("✕ Hide size guide"):
                        del st.session_state["_show_size_guide"]

                st.markdown("**Body Measurements**")
                bc1, bc2, bc3 = st.columns(3)
                with bc1:
                    waist_cm = st.number_input("Waist circumference (cm)", min_value=40.0, max_value=200.0, value=None, placeholder="e.g. 95")
                with bc2:
                    hip_cm   = st.number_input("Hip circumference (cm)",   min_value=40.0, max_value=200.0, value=None, placeholder="e.g. 100")
                with bc3:
                    arm_cm   = st.number_input("Arm circumference (cm)",   min_value=10.0, max_value=60.0,  value=None, placeholder="e.g. 35")

                st.markdown("**Blood Pressure**")
                bp1, bp2 = st.columns(2)
                with bp1:
                    sys_bp = st.number_input("Systolic BP (mmHg)",  min_value=60.0, max_value=250.0, value=None, placeholder="e.g. 128")
                with bp2:
                    dia_bp = st.number_input("Diastolic BP (mmHg)", min_value=30.0, max_value=150.0, value=None, placeholder="e.g. 82")

                st.markdown("**Blood Panel** *(enables diabetes risk assessment)*")
                lab1, lab2, lab3, lab4 = st.columns(4)
                with lab1:
                    glucose = st.number_input("Fasting glucose (mg/dL)", min_value=40.0, max_value=700.0, value=None, placeholder="e.g. 105")
                with lab2:
                    hba1c   = st.number_input("HbA1c (%)",               min_value=3.0,  max_value=20.0,  value=None, placeholder="e.g. 5.9")
                with lab3:
                    insulin = st.number_input("Fasting insulin (µU/mL)", min_value=0.1,  max_value=300.0, value=None, placeholder="e.g. 12")
                with lab4:
                    cholesterol = st.number_input("Total cholesterol (mg/dL)", min_value=50.0, max_value=600.0, value=None, placeholder="e.g. 210")

                run_m2 = st.form_submit_button("Run Clinical Assessment →", type="primary", use_container_width=True)

            if run_m2:
                m1p = st.session_state.get("m1_payload", {})
                full_payload = {
                    "lifestyle": m1p,
                    "clinical": {
                        k: v for k, v in {
                            "waist_cm": waist_cm, "hip_cm": hip_cm, "arm_cm": arm_cm,
                            "sys_bp": sys_bp, "dia_bp": dia_bp, "glucose": glucose,
                            "hba1c": hba1c, "insulin": insulin, "cholesterol": cholesterol,
                            "exact_bmi": st.session_state.get("exact_bmi"),
                        }.items() if v is not None
                    }
                }
                with st.spinner("Running Model 2 clinical assessment..."):
                    try:
                        resp = http_requests.post(f"{OBESITY_API}/predict/full", json=full_payload, timeout=15)
                        resp.raise_for_status()
                        st.session_state.full_result = resp.json()
                        fr = st.session_state.full_result
                        st.session_state.prediction_ctx = build_ctx(fr.get("model1", m1r), m1p) + (
                            f" Model 2 refined classification: "
                            f"{fr['final_classification']['predicted_class']} "
                            f"({fr['final_classification']['confidence_pct']}% confidence)."
                            if fr.get("model2_escalation", {}).get("triggered") else ""
                        ) + (
                            f" Diabetes risk: {fr['diabetes_risk']['risk_classification']}."
                            if fr.get("diabetes_risk") else ""
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Model 2 API error: {e}")

        # ── Model 2 result card ───────────────────────────────────────────────
        if full_r and full_r.get("model2_escalation", {}).get("triggered"):
            m2r   = full_r["model2_escalation"]["result"]
            m2cls = m2r["predicted_class"].replace("_", " ")
            m2cc  = CLASS_COLOR.get(m2r["predicted_class"], "#9b9b9b")
            m2tc  = TIER_COLOR.get(m2r["tier"], "#9b9b9b")
            m2conf = m2r["confidence_pct"]
            gmodel = m2r.get("gender_model", "pooled").title()
            n_inp  = m2r["confidence_components"]["inputs_provided"]
            compl  = m2r["confidence_components"]["c_completeness"]

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("## Model 2 — Clinical Refinement")

            st.markdown(f"""
            <div style="background:#0f1e33;border:1px solid #1e3a55;border-radius:8px;
                        padding:12px 18px;margin-bottom:14px;font-size:.78rem;
                        color:#60a5fa;line-height:1.6;">
                <strong>Why Model 2 ran:</strong> Model 1 showed a close call between the two
                Overweight classes (confidence {m1_conf:.1f}%). Model 2 was trained on
                {2289:,} NHANES 2021–2023 fasting participants using clinical measurements.
                Using the <strong>{gmodel}</strong> gender-stratified model.
                {n_inp} of 9 clinical inputs provided (completeness {compl*100:.0f}%).
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="result-card" style="border-left:4px solid {m2cc};border-top:2px solid #1e3a55;">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;">
                    <div>
                        <div style="font-size:.63rem;text-transform:uppercase;letter-spacing:.1em;color:#60a5fa;margin-bottom:4px;">
                            🔬 MODEL 2 — NHANES CLINICAL</div>
                        <div style="font-size:1.5rem;font-weight:700;color:{m2cc};letter-spacing:-.02em;">{m2cls}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:1.4rem;font-weight:700;color:{m2tc};">{m2conf:.1f}%</div>
                        <div style="font-size:.63rem;color:#555;margin-top:2px;">confidence</div>
                    </div>
                </div>
                <div style="margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;font-size:.7rem;color:#555;margin-bottom:4px;">
                        <span>OW_I probability</span><span>{m2r['probabilities']['OW_I']*100:.1f}%</span>
                    </div>
                    <div style="background:#1a1a1a;border-radius:4px;height:6px;overflow:hidden;border:1px solid #2f2f2f;">
                        <div style="width:{m2r['probabilities']['OW_I']*100:.1f}%;height:100%;background:#fbbf24;border-radius:4px;"></div>
                    </div>
                    <div style="display:flex;justify-content:space-between;font-size:.7rem;color:#555;margin:6px 0 4px 0;">
                        <span>OW_II probability</span><span>{m2r['probabilities']['OW_II']*100:.1f}%</span>
                    </div>
                    <div style="background:#1a1a1a;border-radius:4px;height:6px;overflow:hidden;border:1px solid #2f2f2f;">
                        <div style="width:{m2r['probabilities']['OW_II']*100:.1f}%;height:100%;background:#f97316;border-radius:4px;"></div>
                    </div>
                    <div style="font-size:.68rem;color:{m2tc};margin-top:4px;">{m2r['tier']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            cc = m2r["confidence_components"]
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:-4px 0 12px 0;">
                <div style="background:#1a1a1a;border:1px solid #2f2f2f;border-radius:6px;padding:10px;text-align:center;">
                    <div style="font-size:.58rem;text-transform:uppercase;letter-spacing:.08em;color:#555;margin-bottom:3px;">p · Model Probability</div>
                    <div style="font-size:1rem;font-weight:600;color:#ececec;">{cc['p_model_proba']:.3f}</div>
                </div>
                <div style="background:#1a1a1a;border:1px solid #2f2f2f;border-radius:6px;padding:10px;text-align:center;">
                    <div style="font-size:.58rem;text-transform:uppercase;letter-spacing:.08em;color:#555;margin-bottom:3px;">c · Input Completeness</div>
                    <div style="font-size:1rem;font-weight:600;color:#ececec;">{cc['c_completeness']:.3f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            missing = full_r.get("model2_escalation", {}).get("clinical_inputs_missing", [])
            if missing:
                st.markdown(
                    f'<div style="font-size:.72rem;color:#555;padding:4px 0 12px 0;">'
                    f'Missing inputs (would improve confidence): {", ".join(missing)}</div>',
                    unsafe_allow_html=True,
                )

        # ── Diabetes risk card ────────────────────────────────────────────────
        if full_r and full_r.get("diabetes_risk"):
            dr    = full_r["diabetes_risk"]
            risk  = dr["risk_classification"]
            score = dr["risk_score_pct"]

            if "Type 2" in risk or "Type 1" in risk:
                risk_col = "#ef4444"; risk_bg = "#220f0f"; risk_border = "#3d1a1a"
            elif "Prediabetes" in risk:
                risk_col = "#fbbf24"; risk_bg = "#1f1a0f"; risk_border = "#3d300f"
            else:
                risk_col = "#34d399"; risk_bg = "#0f2218"; risk_border = "#1a3d2b"

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("## Diabetes Risk Assessment")
            st.markdown(f"""
            <div style="background:{risk_bg};border:1px solid {risk_border};
                        border-radius:10px;padding:20px 24px;margin:8px 0;">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;">
                    <div>
                        <div style="font-size:.63rem;text-transform:uppercase;letter-spacing:.1em;color:#555;margin-bottom:4px;">
                            ADA 2024 Guidelines</div>
                        <div style="font-size:1.1rem;font-weight:700;color:{risk_col};line-height:1.3;max-width:520px;">{risk}</div>
                    </div>
                    <div style="text-align:right;flex-shrink:0;margin-left:16px;">
                        <div style="font-size:1.6rem;font-weight:700;color:{risk_col};">{score:.0f}%</div>
                        <div style="font-size:.63rem;color:#555;margin-top:2px;">risk score</div>
                    </div>
                </div>
                <div style="font-size:.8rem;color:#9b9b9b;line-height:1.6;margin-bottom:14px;">{dr['note']}</div>
            </div>
            """, unsafe_allow_html=True)

            if dr.get("driving_factors"):
                st.markdown('<div style="font-size:.75rem;font-weight:600;color:#9b9b9b;margin:12px 0 6px 0;">Driving factors</div>', unsafe_allow_html=True)
                for factor in dr["driving_factors"]:
                    dot_col = "#ef4444" if "diabetic" in factor.lower() or "resistance" in factor.lower() \
                              else "#fbbf24" if "prediabetes" in factor.lower() \
                              else "#34d399"
                    st.markdown(
                        f'<div style="font-size:.78rem;color:#bbb;padding:3px 0 3px 12px;'
                        f'border-left:2px solid {dot_col};margin:3px 0;">{factor}</div>',
                        unsafe_allow_html=True,
                    )

            vals = dr.get("values_used", {})
            if vals.get("homa_ir") is not None:
                hi     = vals["homa_ir"]
                hi_col = "#ef4444" if hi > 5 else "#fbbf24" if hi > 2.5 else "#34d399"
                hi_lbl = "Severe insulin resistance" if hi > 5 \
                         else "Insulin resistance" if hi > 2.5 else "Normal"
                st.markdown(f"""
                <div style="background:#1a1a1a;border:1px solid #2f2f2f;border-radius:6px;
                            padding:12px 16px;margin:10px 0;display:flex;align-items:center;gap:16px;">
                    <div>
                        <div style="font-size:.62rem;text-transform:uppercase;color:#555;margin-bottom:2px;">HOMA-IR</div>
                        <div style="font-size:1.2rem;font-weight:700;color:{hi_col};">{hi:.2f}</div>
                    </div>
                    <div style="font-size:.78rem;color:#9b9b9b;line-height:1.5;">
                        <strong style="color:{hi_col};">{hi_lbl}</strong><br>
                        Normal &lt;2.5 · Resistant &gt;2.5 · Severe &gt;5.0
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Next step nudge ───────────────────────────────────────────────────
        show_m2_hint = needs_m2 and full_r is None
        st.markdown("""
        <div style="background:#1a1a1a;border:1px solid #2f2f2f;border-radius:8px;
                    padding:16px 20px;margin:16px 0;">
            <div style="font-size:.68rem;color:#555;margin-bottom:5px;
                        text-transform:uppercase;letter-spacing:.08em;">Next step</div>
            <div style="font-size:.86rem;color:#9b9b9b;line-height:1.6;">
                Switch to the <strong style="color:#ececec;">Nutrition Plan</strong> tab —
                your result is pre-loaded as context.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if show_m2_hint:
            st.markdown('<div style="font-size:.82rem;color:#9b9b9b;padding:0 0 12px 0;">☝️ Enter clinical measurements above to run Model 2 first.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Nutrition Plan Chat
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:

    if not st.session_state.prediction_result and not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align:center;padding:72px 0;">
            <div style="font-size:2.5rem;margin-bottom:16px;">🏥</div>
            <div style="font-size:1.05rem;color:#ececec;font-weight:600;margin-bottom:8px;">
                Complete your assessment first</div>
            <div style="font-size:.82rem;color:#555;max-width:360px;margin:0 auto;line-height:1.6;">
                Head to the <strong style="color:#9b9b9b;">Assessment</strong> tab, fill in the form,
                and hit <strong style="color:#9b9b9b;">Run Assessment</strong>.
                Your result will be pre-loaded here as context.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        if not st.session_state.chat_history:
            r   = st.session_state.prediction_result
            cls = r["predicted_class"].replace("_", " ") if r else ""
            cc  = CLASS_COLOR.get(r["predicted_class"], "#9b9b9b") if r else "#9b9b9b"
            conf = r["confidence_pct"] if r else 0

            st.markdown(f"""
            <div style="text-align:center;padding:28px 0 16px 0;">
                <h1 style="font-size:1.8rem;font-weight:600;color:#ececec;margin:0 0 6px 0;">Nutrition Plan</h1>
                <p style="font-size:.75rem;color:#666;letter-spacing:.06em;text-transform:uppercase;margin:0 0 14px 0;">
                    RAG · Clinical Guidelines · {'Gateway' if st.session_state.cfg_use_gateway else 'Gemini'}</p>
                <div style="display:inline-block;background:#1a1a1a;border:1px solid #2f2f2f;
                            border-radius:8px;padding:10px 20px;">
                    <span style="font-size:.72rem;color:#555;margin-right:8px;">Result:</span>
                    <span style="font-size:.9rem;font-weight:600;color:{cc};">{cls}</span>
                    <span style="font-size:.72rem;color:#555;margin-left:8px;">· {conf:.1f}% confidence</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            suggestions = [
                "Generate a nutrition plan for my result",
                "What foods should I avoid?",
                "How much exercise do I need?",
                "Explain my classification",
            ]
            cols = st.columns(len(suggestions))
            for col, sug in zip(cols, suggestions):
                with col:
                    if st.button(sug, key=f"sug_{sug}", use_container_width=True):
                        st.session_state._queued = sug
                        st.rerun()

        for msg in st.session_state.chat_history:
            render_msg(msg)

        _queued = st.session_state.pop("_queued", None)
        prompt  = st.chat_input("Ask about nutrition, dietary plans, lifestyle changes...")
        active  = prompt or _queued

        # ── Chat interrupt button (shown while a response is generating) ───────
        if st.session_state.get("is_chatting", False):
            if st.button("⏹ Stop generating", key="interrupt_chat"):
                st.session_state.chat_interrupt_requested = True

        if active:
            ctx = st.session_state.prediction_ctx
            full_query = f"{ctx}\n\nUser question: {active}" if ctx and not st.session_state.chat_history else active

            st.session_state.chat_history.append({"role": "user", "content": active})
            _save_chat()
            render_msg({"role": "user", "content": active})

            answer, sources, recs = "", [], []
            st.session_state.is_chatting = True
            st.session_state.chat_interrupt_requested = False

            if st.session_state.indexer is None:
                r   = st.session_state.prediction_result
                cls = r["predicted_class"].replace("_", " ") if r else "your result"
                advice = CLASS_GENERIC_ADVICE.get(cls, "A balanced diet and regular physical activity are the foundation of any healthy lifestyle.")

                web_extra = ""
                if st.session_state.get("web_search_enabled", False):
                    with st.spinner("🌐 Searching the web..."):
                        web_snippet = _gemini_web_search(active, model_name=st.session_state.cfg_model if not st.session_state.cfg_use_gateway else "gemini-2.5-flash")
                    if web_snippet and not web_snippet.startswith("[Web search failed"):
                        st.markdown(
                            f'<div style="background:#0f1e33;border:1px solid #1e3a55;'
                            f'border-radius:6px;padding:10px 14px;font-size:.78rem;'
                            f'color:#60a5fa;margin-bottom:12px;line-height:1.6;">'
                            f'<span style="font-weight:600;display:block;margin-bottom:4px;">'
                            f'🌐 Web context</span>{web_snippet}</div>',
                            unsafe_allow_html=True,
                        )
                        web_extra = f"\n\nAdditional web context:\n{web_snippet}"

                answer = (
                    f"Based on your assessment result of **{cls}**:\n\n{advice}"
                    f"{web_extra}\n\n"
                    f"For fully evidence-backed recommendations with clinical citations, "
                    f"index relevant PDFs (dietary guidelines, nutrition studies) in the "
                    f"**Index PDFs** tab. The collection loads automatically once indexed."
                )
                st.markdown(answer)
                st.session_state.is_chatting = False
                st.session_state.chat_interrupt_requested = False
            else:
                # ── Optional web search ───────────────────────────────────────
                rag_query = full_query
                if st.session_state.get("web_search_enabled", False):
                    _ws_model = st.session_state.cfg_model if not st.session_state.cfg_use_gateway else "gemini-2.5-flash"
                    with st.spinner("🌐 Searching the web..."):
                        web_snippet = _gemini_web_search(active, model_name=_ws_model)
                    if web_snippet and not web_snippet.startswith("[Web search failed"):
                        st.markdown(
                            f'<div style="background:#0f1e33;border:1px solid #1e3a55;'
                            f'border-radius:6px;padding:10px 14px;font-size:.78rem;'
                            f'color:#60a5fa;margin-bottom:12px;line-height:1.6;">'
                            f'<span style="font-weight:600;display:block;margin-bottom:4px;">'
                            f'🌐 Web context</span>{web_snippet}</div>',
                            unsafe_allow_html=True,
                        )
                        rag_query = (
                            f"Web search context (use this to supplement your answer):\n"
                            f"{web_snippet}\n\n{full_query}"
                        )
                    elif web_snippet.startswith("[Web search failed"):
                        st.caption(web_snippet)

                # ── RAG query ─────────────────────────────────────────────────
                with st.spinner("Retrieving guidelines and generating plan..."):
                    try:
                        if st.session_state.chat_interrupt_requested:
                            answer = "_Response interrupted by user._"
                            st.info("Generation interrupted.")
                        else:
                            idx = st.session_state.indexer
                            wc  = st.session_state.cfg_word_count
                            wc_instruction = (
                                f"\n\n[RESPONSE LENGTH: Write approximately {wc} words. "
                                f"Be thorough but stay within this target.]"
                            )
                            final_rag_query = rag_query + wc_instruction

                            # ── Patch indexer for the active backend ──────────────
                            if st.session_state.cfg_use_gateway:
                                # Gateway mode — no Gemini key needed
                                if not st.session_state.cfg_gateway_key:
                                    st.error(
                                        "CMU AI Gateway is enabled but no key was provided. "
                                        "Enter your key in the sidebar, or uncheck 'Use CMU AI Gateway'."
                                    )
                                    st.stop()
                                try:
                                    _patch_indexer_for_gateway(
                                        idx,
                                        model=st.session_state.cfg_model,
                                        gateway_key=st.session_state.cfg_gateway_key,
                                        gateway_url=st.session_state.cfg_gateway_url,
                                        word_count=wc,
                                    )
                                except ValueError as ve:
                                    st.error(str(ve))
                                    st.stop()
                                st.markdown(
                                    f'<div style="font-size:.68rem;color:#a78bfa;padding-bottom:6px;">'
                                    f'🔮 Gateway · {st.session_state.cfg_model} · '
                                    f'Judge: Claude Sonnet · ~{wc} words</div>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                # Gemini mode — inject key from session state or env
                                _gkey = (
                                    st.session_state.cfg_gemini_api_key
                                    or os.environ.get("GEMINI_API_KEY", "")
                                    or os.environ.get("GOOGLE_API_KEY", "")
                                )
                                if not _gkey:
                                    st.error(
                                        "No Gemini API key found. "
                                        "Enter it in the **Index PDFs** tab or set GEMINI_API_KEY in your environment."
                                    )
                                    st.stop()
                                try:
                                    _patch_indexer_for_gemini(
                                        idx,
                                        model=st.session_state.cfg_model,
                                        gemini_key=_gkey,
                                        word_count=wc,
                                    )
                                except ValueError as ve:
                                    st.error(str(ve))
                                    st.stop()
                                st.markdown(
                                    f'<div style="font-size:.68rem;color:#555;padding-bottom:6px;">'
                                    f'✦ {st.session_state.cfg_model} · ~{wc} words</div>',
                                    unsafe_allow_html=True,
                                )

                            answer, sources = idx.query(
                                final_rag_query,
                                top_k=st.session_state.cfg_top_k,
                                model=st.session_state.cfg_model,
                                output_language=st.session_state.cfg_language,
                                max_iterations=st.session_state.cfg_iterations,
                            )
                            st.markdown(answer)
                            recs = _unique_papers(sources, st.session_state.file_url_map, k=2)
                            _render_recs(recs)
                            _render_sources(sources)
                    except Exception as e:
                        answer = f"Query failed: {e}"
                        st.error(answer)
                    finally:
                        st.session_state.is_chatting = False
                        st.session_state.chat_interrupt_requested = False

            st.session_state.chat_history.append({
                "role": "assistant", "content": answer,
                "sources": sources, "recs": recs,
            })
            _save_chat()
            st.rerun()

        if st.session_state.chat_history:
            if st.button("Clear chat", key="clear_chat"):
                st.session_state.chat_history = []
                _save_chat()
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Index PDFs
# ══════════════════════════════════════════════════════════════════════════════
with tab_index:
    st.markdown("""
    <div style="text-align:center;padding:36px 0 24px 0;">
        <h1 style="font-size:1.9rem;font-weight:600;color:#ececec;letter-spacing:-.03em;margin:0 0 6px 0;">
            Index Clinical PDFs</h1>
        <p style="font-size:.75rem;color:#666;letter-spacing:.06em;text-transform:uppercase;margin:0;">
            BGE · Milvus · LangGraph</p>
    </div>
    """, unsafe_allow_html=True)

    if not RAG_AVAILABLE:
        st.error("indexer.py not found in this folder. Place it alongside obesity_app_v2.py to enable indexing.")
    else:
        st.markdown("### Settings")

        em1, em2 = st.columns([1, 2])
        with em1:
            EMBED_OPTIONS = [
                "BGE (local · 1024-dim)",
                "Gemini embedding-001 (3072-dim)",
            ]
            embed_choice = st.selectbox(
                "Embedding model",
                EMBED_OPTIONS,
                index=EMBED_OPTIONS.index(st.session_state.cfg_embed_model)
                      if st.session_state.cfg_embed_model in EMBED_OPTIONS else 0,
                key="embed_model_sel",
                help="BGE runs locally — no API key, 1024 dimensions.\n"
                     "Gemini embedding-001 — requires GEMINI_API_KEY, 3072 dimensions.",
            )
            st.session_state.cfg_embed_model = embed_choice
        with em2:
            if "Gemini" in embed_choice:
                env_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
                if env_key:
                    st.markdown(
                        '<div style="background:#0f2218;border:1px solid #1a3d2b;border-radius:6px;'
                        'padding:8px 14px;font-size:.78rem;color:#34d399;margin-top:22px;">'
                        '✓ GEMINI_API_KEY found in environment</div>',
                        unsafe_allow_html=True,
                    )
                    st.session_state.cfg_gemini_api_key = env_key
                else:
                    typed_key = st.text_input(
                        "Gemini API key",
                        value=st.session_state.cfg_gemini_api_key,
                        type="password",
                        placeholder="AIza...",
                        key="gemini_key_input",
                        help="Required for Gemini embeddings and Gemini chat mode.",
                    )
                    st.session_state.cfg_gemini_api_key = typed_key
                    if typed_key:
                        # Inject immediately so the rest of the session can use it
                        os.environ["GEMINI_API_KEY"] = typed_key
                        os.environ["GOOGLE_API_KEY"] = typed_key
                        st.markdown(
                            '<div style="font-size:.72rem;color:#34d399;padding-top:4px;">✓ Key saved for this session</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div style="font-size:.72rem;color:#f87171;padding-top:4px;">⚠️ API key required</div>',
                            unsafe_allow_html=True,
                        )
            else:
                st.markdown(
                    '<div style="font-size:.75rem;color:#555;padding-top:26px;">'
                    'Runs locally · no API key · sentence-transformers</div>',
                    unsafe_allow_html=True,
                )

        # Also show a Gemini key input when using BGE embed but Gemini chat mode
        if "BGE" in embed_choice and not st.session_state.cfg_use_gateway:
            _env_gkey = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
            if not _env_gkey:
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("**Gemini Chat API Key** *(required for Nutrition Plan chat)*")
                _chat_key = st.text_input(
                    "Gemini API key for chat",
                    value=st.session_state.cfg_gemini_api_key,
                    type="password",
                    placeholder="AIza...",
                    key="gemini_chat_key_input",
                    help="Needed for generate/reflect/revise nodes in the RAG pipeline.",
                )
                if _chat_key:
                    st.session_state.cfg_gemini_api_key = _chat_key
                    os.environ["GEMINI_API_KEY"] = _chat_key
                    os.environ["GOOGLE_API_KEY"] = _chat_key
                    st.markdown(
                        '<div style="font-size:.72rem;color:#34d399;padding-top:4px;">✓ Gemini key saved for this session</div>',
                        unsafe_allow_html=True,
                    )

        use_gemini_embed = "Gemini" in embed_choice
        coll_name = "papers_rag_gemini" if use_gemini_embed else "papers_rag_interactive"
        st.markdown(
            f'<div style="font-size:.72rem;color:#9b9b9b;padding:6px 0 0 0;">'
            f'📦 Collection: <code>{coll_name}</code></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

        # ── Web search toggle ─────────────────────────────────────────────────
        col_ws, col_info = st.columns([1, 3])
        with col_ws:
            st.session_state.web_search_enabled = st.checkbox(
                "Enable web search in Nutrition Plan chat",
                value=st.session_state.web_search_enabled,
            )
        with col_info:
            if st.session_state.web_search_enabled:
                gemini_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
                if gemini_key:
                    st.markdown('<div style="background:#0f2218;border:1px solid #1a3d2b;border-radius:6px;'
                                'padding:8px 14px;font-size:.78rem;color:#34d399;margin-top:4px;">'
                                '✓ Web search active — Gemini API key found</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<div style="background:#220f0f;border:1px solid #3d1a1a;border-radius:6px;'
                                'padding:8px 14px;font-size:.78rem;color:#f87171;margin-top:4px;">'
                                '⚠️ GEMINI_API_KEY not set — web search will silently skip</div>',
                                unsafe_allow_html=True)
            else:
                st.markdown('<div style="font-size:.75rem;color:#555;padding-top:8px;">'
                            'Web search off — only indexed PDFs will be used</div>',
                            unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── File uploader ─────────────────────────────────────────────────────
        st.markdown("### Upload & Categorise")
        uploaded_files = st.file_uploader(
            "Drop one or more PDFs (dietary guidelines, nutrition studies, intervention research)",
            type=["pdf"],
            accept_multiple_files=True,
            key="idx_uploader",
        )

        if uploaded_files:
            st.markdown("**Add custom category**")
            cn1, cn2 = st.columns([3, 1])
            with cn1:
                new_type = st.text_input(
                    "new_cat", key="new_cat_input",
                    placeholder="e.g. Diabetes Research, Cardiology...",
                    label_visibility="collapsed",
                )
            with cn2:
                if st.button("Add", use_container_width=True, key="add_cat_btn"):
                    cleaned = new_type.strip()
                    all_current = st.session_state.custom_paper_types + PAPER_TYPES
                    if cleaned and cleaned not in all_current:
                        st.session_state.custom_paper_types.insert(0, cleaned)
                        st.rerun()

            if st.session_state.custom_paper_types:
                st.markdown(
                    " ".join(f'<span class="badge badge-type">{t}</span>'
                             for t in st.session_state.custom_paper_types),
                    unsafe_allow_html=True,
                )

            st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

            all_types   = st.session_state.custom_paper_types + PAPER_TYPES
            st.markdown("**Categorise files** — select a category, then click files to assign it")
            active_type = st.selectbox(
                "active_type", all_types, key="active_cat_sel",
                label_visibility="collapsed",
            )

            cols = st.columns(3)
            for i, uf in enumerate(uploaded_files):
                assignment = st.session_state.file_assignments.get(uf.name)
                bg     = "#1e1533" if assignment else "#2a2a2a"
                border = "#3a2f55" if assignment else "#444"
                tc     = "#a78bfa" if assignment else "#666"
                lbl    = assignment or "unassigned"
                with cols[i % 3]:
                    st.markdown(
                        f'<div style="background:{bg};border:1px solid {border};'
                        f'border-radius:8px;padding:10px 12px;margin-bottom:6px;">'
                        f'<div style="font-size:.78rem;color:#ececec;font-weight:500;'
                        f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'
                        f'margin-bottom:4px;" title="{uf.name}">{uf.name}</div>'
                        f'<span style="font-size:.62rem;background:#111;color:{tc};'
                        f'padding:1px 6px;border-radius:3px;border:1px solid {border};">'
                        f'{lbl}</span></div>',
                        unsafe_allow_html=True,
                    )
                    btn_lbl = f"✓ {active_type}" if assignment != active_type else "✕ Remove"
                    if st.button(btn_lbl, key=f"assign_{uf.name}", use_container_width=True):
                        if assignment == active_type:
                            del st.session_state.file_assignments[uf.name]
                        else:
                            st.session_state.file_assignments[uf.name] = active_type
                        st.rerun()

            assigned_count = len(st.session_state.file_assignments)
            unassigned     = [uf.name for uf in uploaded_files
                              if uf.name not in st.session_state.file_assignments]
            st.markdown(
                f'<div style="margin-top:8px;font-size:.78rem;color:#9b9b9b;">'
                f'{assigned_count}/{len(uploaded_files)} files assigned</div>',
                unsafe_allow_html=True,
            )
            if unassigned:
                st.markdown(
                    f'<div style="font-size:.72rem;color:#555;margin-top:2px;">'
                    f'Unassigned: {", ".join(unassigned)}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<hr>", unsafe_allow_html=True)

        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            append_mode = st.checkbox("Append to existing index", value=False)
        with oc2:
            show_chunks = st.checkbox("Preview first 5 chunks",   value=True)
        with oc3:
            st.caption(f"Index: {st.session_state.cfg_index_type} · "
                       f"Chunk: {st.session_state.cfg_chunk_size}w · "
                       f"Overlap: {st.session_state.cfg_chunk_overlap}w")

        ready_files = [uf for uf in (uploaded_files or [])
                       if uf.name in st.session_state.file_assignments]

        if st.button("Start Indexing", type="primary",
                     disabled=len(ready_files) == 0,
                     use_container_width=True, key="start_idx_btn"):

            st.session_state.interrupt_requested = False
            st.session_state.is_indexing         = True

            tmp_dir = Path("/tmp/ob_idx_uploads")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            saved = []
            try:
                for uf in ready_files:
                    dest = tmp_dir / uf.name
                    dest.write_bytes(uf.read())
                    saved.append((dest, st.session_state.file_assignments[uf.name]))

                status_box  = st.empty()
                bar         = st.progress(0, text="Starting...")
                log_box     = st.empty()
                logs: list  = []

                def log(msg):
                    logs.append(msg)
                    log_box.markdown(
                        '<div style="background:#2a2a2a;border:1px solid #333;'
                        'border-radius:8px;padding:12px 16px;font-size:.8rem;'
                        'color:#bbb;line-height:1.7;">'
                        + "".join(f"· {l}<br>" for l in logs[-10:])
                        + "</div>",
                        unsafe_allow_html=True,
                    )

                def tick(pct, label):
                    bar.progress(min(float(pct), 1.0),
                                 text=f"{label} — {int(min(pct,1.0)*100)}%")
                    status_box.markdown(
                        f'<div style="font-size:.82rem;color:#9b9b9b;padding:4px 0 8px 0;">'
                        f'{label}</div>',
                        unsafe_allow_html=True,
                    )

                tick(0.02, "Creating indexer")
                use_gemini_embed = "Gemini" in st.session_state.cfg_embed_model
                embed_dim        = 3072 if use_gemini_embed else 1024
                collection_name  = "papers_rag_gemini" if use_gemini_embed else "papers_rag_interactive"

                if use_gemini_embed:
                    gkey = (st.session_state.cfg_gemini_api_key
                            or os.environ.get("GEMINI_API_KEY", "")
                            or os.environ.get("GOOGLE_API_KEY", ""))
                    if not gkey:
                        st.error("Gemini API key required for Gemini embeddings. Enter it in the Settings section above.")
                        st.session_state.is_indexing = False
                        st.stop()
                    os.environ["GEMINI_API_KEY"] = gkey
                    os.environ["GOOGLE_API_KEY"] = gkey

                import indexer as _idx_mod
                _idx_mod.EMBED_DIM = embed_dim

                indexer_obj = PDFIndexer(
                    chunk_size=st.session_state.cfg_chunk_size,
                    chunk_overlap=st.session_state.cfg_chunk_overlap,
                    model=st.session_state.cfg_model,
                    index_type=st.session_state.cfg_index_type,
                    collection_name=collection_name,
                    drop_old_collection=(not append_mode),
                )
                log(f"Indexer ready [{st.session_state.cfg_index_type}] "
                    f"embed={'Gemini' if use_gemini_embed else 'BGE'} "
                    f"dim={embed_dim}")

                all_chunks = []
                for fi, (pdf_path, ptype) in enumerate(saved):
                    if st.session_state.interrupt_requested:
                        break
                    p0 = 0.05 + (fi / max(len(saved), 1)) * 0.25
                    p1 = 0.05 + ((fi+1) / max(len(saved), 1)) * 0.25
                    tick(p0, f"Extracting {pdf_path.name}")
                    chunks = indexer_obj.extract_and_chunk(str(pdf_path), paper_type=ptype)
                    all_chunks.extend(chunks)
                    tick(p1, f"{pdf_path.name} — {len(chunks)} chunks")
                    log(f"{pdf_path.name} [{ptype}] — {len(chunks)} chunks")

                total = len(all_chunks)
                tick(0.33, f"Extraction done — {total} chunks")

                if use_gemini_embed:
                    log("Embedding with Gemini embedding-001 (3072-dim)...")
                    from langchain_google_genai import GoogleGenerativeAIEmbeddings
                    gkey = (st.session_state.cfg_gemini_api_key
                            or os.environ.get("GEMINI_API_KEY", "")
                            or os.environ.get("GOOGLE_API_KEY", ""))
                    os.environ["GOOGLE_API_KEY"] = gkey
                    gem_embedder = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
                    BATCH     = 20
                    n_batches = math.ceil(total / BATCH)

                    for bi in range(0, total, BATCH):
                        if st.session_state.interrupt_requested:
                            break
                        bn    = bi // BATCH + 1
                        batch = all_chunks[bi: bi + BATCH]
                        texts = [c["text"] for c in batch]
                        embs  = gem_embedder.embed_documents(texts)
                        for j, emb in enumerate(embs):
                            all_chunks[bi + j]["embedding"] = emb
                        pct = 0.33 + (min(bi + BATCH, total) / max(total, 1)) * 0.42
                        tick(pct, f"Embedding batch {bn}/{n_batches} (Gemini 3072-dim)")
                        log(f"Gemini batch {bn}/{n_batches} done")
                else:
                    log("Embedding (BGE)...")
                    embedder  = indexer_obj._get_embedder()
                    BATCH     = 64
                    n_batches = math.ceil(total / BATCH)

                    for bi in range(0, total, BATCH):
                        if st.session_state.interrupt_requested:
                            break
                        bn    = bi // BATCH + 1
                        texts = ["Represent this passage for retrieval: " + all_chunks[bi+j]["text"]
                                 for j in range(min(BATCH, total-bi))]
                        embs  = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                        for j, emb in enumerate(embs):
                            all_chunks[bi+j]["embedding"] = emb.tolist()
                        pct = 0.33 + (min(bi+BATCH, total) / max(total,1)) * 0.42
                        tick(pct, f"Embedding batch {bn}/{n_batches}")
                        log(f"Batch {bn}/{n_batches} done")

                tick(0.80, f"Building {st.session_state.cfg_index_type} index...")
                log(f"Building {st.session_state.cfg_index_type} index...")

                if append_mode and st.session_state.indexer:
                    st.session_state.indexer.add_to_index(all_chunks)
                    st.session_state.indexer.set_output_language(st.session_state.cfg_language)
                else:
                    indexer_obj.build_index(all_chunks)
                    indexer_obj.set_output_language(st.session_state.cfg_language)
                    st.session_state.indexer = indexer_obj

                tick(1.0, "Indexing complete ✓")

                new_entries = [(p.name, pt) for p, pt in saved]
                if append_mode:
                    st.session_state.indexed_files.extend(new_entries)
                else:
                    st.session_state.indexed_files = new_entries

                st.session_state.indexer._indexed_files = st.session_state.indexed_files
                st.session_state.indexer._file_url_map  = st.session_state.file_url_map
                _save_indexed()

                st.session_state.index_stats = {
                    "total_chunks": total,
                    "total_files":  len(st.session_state.indexed_files),
                    "index_type":   st.session_state.cfg_index_type,
                    "embed_model":  "Gemini embedding-001" if use_gemini_embed else EMBED_MODEL,
                    "chunk_size":   st.session_state.cfg_chunk_size,
                }

                st.success(
                    f"**{len(saved)} file(s)** — **{total} chunks** — "
                    f"**{st.session_state.cfg_index_type}** index built ✓"
                )

                if show_chunks and all_chunks:
                    st.markdown("**Chunk preview (first 5)**")
                    for c in all_chunks[:5]:
                        st.markdown(
                            f'<div class="chunk-preview">'
                            f'<span class="badge">{c["source"]}</span>'
                            f'<span class="badge">p{c["page"]}</span>'
                            f'<span class="badge badge-type">{c["paper_type"]}</span>'
                            f'<span class="badge">#{c["chunk_id"]}</span><br><br>'
                            f'{c["text"][:480]}{"..." if len(c["text"])>480 else ""}</div>',
                            unsafe_allow_html=True,
                        )

            except Exception as e:
                if "Interrupted" in str(e):
                    st.warning("Indexing interrupted.")
                else:
                    st.error(str(e))
                    st.exception(e)
            finally:
                st.session_state.is_indexing = False

        if st.session_state.is_indexing:
            if st.button("⏹ Interrupt", key="interrupt_idx"):
                st.session_state.interrupt_requested = True

        # ── NPZ export ────────────────────────────────────────────────────────
        if st.session_state.indexer is not None and st.session_state.index_stats.get("total_chunks", 0) > 0:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("### Export vector store")
            st.caption(
                f"Download the current index ({st.session_state.index_stats['total_chunks']} chunks, "
                f"{st.session_state.index_stats.get('total_files',0)} files) as a compressed .npz file "
                f"for offline backup or re-import."
            )
            if st.button("⬇ Export to .npz", use_container_width=True, key="npz_export_btn"):
                try:
                    import tempfile, numpy as np
                    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
                        tmp_path = tmp.name
                    with st.spinner("Exporting index to .npz..."):
                        n_exported = st.session_state.indexer.export_to_npz(tmp_path)
                    with open(tmp_path, "rb") as fh:
                        npz_bytes = fh.read()
                    Path(tmp_path).unlink(missing_ok=True)
                    embed_label = st.session_state.index_stats.get("embed_model", "bge")
                    safe_label  = re.sub(r"[^a-zA-Z0-9_\-]", "_", embed_label)[:20]
                    fname = f"rag_index_{st.session_state.cfg_index_type}_{safe_label}.npz"
                    st.download_button(
                        label=f"💾 Download {fname} ({n_exported} chunks)",
                        data=npz_bytes,
                        file_name=fname,
                        mime="application/octet-stream",
                        use_container_width=True,
                        key="npz_download_btn",
                    )
                    st.success(f"Export ready — {n_exported} chunks ✓")
                except Exception as e:
                    st.error(f"Export failed: {e}")
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### Import existing vector store")
        st.caption("If you already exported an index from the RAG app, import it here directly.")

        npz_file = st.file_uploader("Upload .npz export", type=["npz"], key="npz_upload_idx")
        if npz_file:
            append_npz = st.checkbox("Append to existing index", value=False, key="npz_append_idx")
            if st.button("Import", type="primary", use_container_width=True, key="npz_import_btn"):
                try:
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
                        tmp.write(npz_file.read())
                        tmp_path = tmp.name

                    if st.session_state.indexer is None:
                        st.session_state.indexer = PDFIndexer(
                            chunk_size=st.session_state.cfg_chunk_size,
                            chunk_overlap=st.session_state.cfg_chunk_overlap,
                            model=st.session_state.cfg_model,
                            index_type=st.session_state.cfg_index_type,
                            collection_name="papers_rag_interactive",
                            drop_old_collection=(not append_npz),
                        )

                    with st.spinner("Importing..."):
                        stats = st.session_state.indexer.import_from_npz(tmp_path, append=append_npz)
                    Path(tmp_path).unlink(missing_ok=True)

                    imported_files   = stats.get("indexed_files", [])
                    imported_url_map = stats.get("file_url_map", {})
                    if imported_files:
                        st.session_state.indexed_files = imported_files
                        st.session_state.file_url_map  = imported_url_map
                        st.session_state.indexer._indexed_files = imported_files
                        st.session_state.indexer._file_url_map  = imported_url_map
                        _save_indexed()

                    st.session_state.index_stats = {
                        "total_chunks": stats["n_chunks"],
                        "total_files":  len(imported_files) or len(st.session_state.indexed_files),
                        "index_type":   stats.get("index_type", st.session_state.cfg_index_type),
                    }
                    st.success(f"Imported {stats['n_chunks']} chunks ✓")
                    st.rerun()
                except Exception as e:
                    st.error(f"Import failed: {e}")
