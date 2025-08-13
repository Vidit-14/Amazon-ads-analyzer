# pages/Voice_of_the_Customer.py

import re
import streamlit as st
import pandas as pd
import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from datetime import datetime, date, timedelta
if not st.session_state.get("logged_in", False):
    st.switch_page("app.py")

# Hide the top "app" header in the sidebar nav
st.markdown("""
<style>
/* In the sidebar, hide the first nav item, which is the main script label */
div[data-testid="stSidebarNav"] > div:first-child { 
    display: none; 
}
</style>
""", unsafe_allow_html=True)
# ── Page config must be first Streamlit call ─────────────────────────────────────
st.set_page_config(page_title="Voice of the Customer", layout="wide")

st.markdown("""
<style>
  h1 {
    background: -webkit-linear-gradient(#00BFFF, #1b7371);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 3px 6px rgba(0,0,0,0.08);
    font-weight: 800;
    font-size: 3.0rem;
    padding: .5rem 0 1rem;
  }
  .card {
    border-radius: 14px; padding: 18px 16px; text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,.06); border: 1px solid rgba(0,0,0,.06);
    background: #fff; height: 130px; display:flex; flex-direction:column; justify-content:center;
    margin-bottom: 8px;
  }
  .label { display:inline-block; padding:6px 14px; border-radius: 999px; color:#fff; font-weight:600; }
  .verypoor { background:#d63b16; }    /* red */
  .poor     { background:#ff9800; }    /* orange */
  .fair     { background:#f5c11a; }    /* yellow */
  .good     { background:#a6ce39; }    /* lime */
  .excellent{ background:#2e7d32; }    /* green */
  .count { font-size: 34px; font-weight:800; margin: 6px 0 2px; }

  /* Make Streamlit buttons full-width inside their column */
  div.stButton > button {
    width: 100%;
    border-radius: 10px;
    border: 1px solid rgba(0,0,0,.08);
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
    height: 38px;
    font-weight: 600;
  }

  .banner {
    padding:14px 16px; border-radius:12px; background:#fff8e1; border:1px solid #ffecb3; margin-bottom:16px;
  }
</style>
""", unsafe_allow_html=True)

st.title("Voice of the Customer")

# ── Config ───────────────────────────────────────────────────────────────────────
SHEET_NAME = "VOC_Status_Snapshots"  # worksheet/tab inside your "Dashboard_data" spreadsheet
REQUIRED_COLS = [
    'Snapshot Date', 'Product name', 'ASIN', 'SKU', 'Condition', 'Fulfilled by',
    'NCX rate', 'NCX orders', 'Total orders', 'Star rating', 'Return rate',
    'Top NCX reason', 'Last updated', 'Status', 'Return Badge Displayed'
]
STATUS_ORDER = ["Very poor", "Poor", "Fair", "Good", "Excellent"]
STATUS_TO_CLASS = {
    "Very poor": "verypoor", "Poor": "poor", "Fair": "fair",
    "Good": "good", "Excellent": "excellent"
}

# ── Helpers ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def connect_to_gsheet():
    try:
        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        return gc.open("Dashboard_data")  # change if you use a different spreadsheet name
    except Exception as e:
        st.error(f"Google Sheets connection failed: {e}")
        st.stop()

def ensure_sheet_with_columns(spreadsheet, sheet_name, required_cols):
    """Create sheet if missing and guarantee headers exist."""
    try:
        ws = spreadsheet.worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=sheet_name, rows="8000", cols="40")
        set_with_dataframe(ws, pd.DataFrame(columns=required_cols), resize=True)
        return ws

    try:
        df = get_as_dataframe(ws, evaluate_formulas=True).dropna(how="all")
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        df = pd.DataFrame(columns=required_cols)
    else:
        for c in required_cols:
            if c not in df.columns:
                df[c] = ""
        other = [c for c in df.columns if c not in required_cols]
        df = df[required_cols + other]
    set_with_dataframe(ws, df, include_index=False, include_column_header=True, resize=True)
    return ws

def clean_status(x: str) -> str:
    s = str(x)
    s = s.replace("\u00a0", " ")           # NBSP -> space
    s = s.replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    # canonicalize
    if s in {"very poor","very  poor"}: return "Very poor"
    if s == "poor": return "Poor"
    if s == "fair": return "Fair"
    if s == "good": return "Good"
    if s == "excellent": return "Excellent"
    return s.title()

def clean_asin(x) -> str:
    return str(x).strip().upper()

@st.cache_data(ttl=300)
def load_snapshots(_spreadsheet) -> pd.DataFrame:
    ws = ensure_sheet_with_columns(_spreadsheet, SHEET_NAME, REQUIRED_COLS)
    df = get_as_dataframe(ws, evaluate_formulas=True).dropna(how="all")
    if df.empty:
        return pd.DataFrame(columns=REQUIRED_COLS)

    df['Snapshot Date'] = pd.to_datetime(df['Snapshot Date'], errors='coerce').dt.tz_localize(None)
    df['ASIN']   = df['ASIN'].apply(clean_asin)
    df['Status'] = df['Status'].apply(clean_status)

    # one row per (Snapshot Date, ASIN) — keep "worst" status if duplicates slipped in
    rank = {s:i for i,s in enumerate(STATUS_ORDER)}  # 0=Very poor (worst) ... 4=Excellent (best)
    df['_rank'] = df['Status'].map(rank).fillna(999)
    df = df.sort_values(['Snapshot Date','ASIN','_rank']) \
           .drop_duplicates(subset=['Snapshot Date','ASIN'], keep='first') \
           .drop(columns=['_rank'])
    return df

def save_snapshot(spreadsheet, new_df: pd.DataFrame):
    """Overwrite rows for (Snapshot Date, ASIN) in existing data, then append new."""
    ws = ensure_sheet_with_columns(spreadsheet, SHEET_NAME, REQUIRED_COLS)
    try:
        existing = get_as_dataframe(ws, evaluate_formulas=True).dropna(how="all")
    except Exception:
        existing = pd.DataFrame(columns=REQUIRED_COLS)

    # Normalize types/keys
    for c in REQUIRED_COLS:
        if c not in existing.columns:
            existing[c] = ""
    existing['Snapshot Date'] = pd.to_datetime(existing['Snapshot Date'], errors='coerce').dt.tz_localize(None)
    existing['ASIN'] = existing['ASIN'].apply(clean_asin)

    new_df = new_df.copy()
    new_df['Snapshot Date'] = pd.to_datetime(new_df['Snapshot Date'], errors='coerce').dt.tz_localize(None)
    new_df['ASIN'] = new_df['ASIN'].apply(clean_asin)

    # Replace by key using index drop + append for precise overwrite
    if not existing.empty:
        existing_idxed = existing.set_index(['Snapshot Date','ASIN'])
        new_idxed = new_df.set_index(['Snapshot Date','ASIN'])
        existing_idxed = existing_idxed.drop(index=new_idxed.index, errors='ignore')
        combined = pd.concat([existing_idxed, new_idxed], axis=0).reset_index()
    else:
        combined = new_df

    set_with_dataframe(ws, combined[REQUIRED_COLS], include_index=False, include_column_header=True, resize=True)
    st.cache_data.clear()

def get_latest_and_previous(df: pd.DataFrame):
    if df.empty:
        return None, None, None, None
    dates = sorted(df['Snapshot Date'].dropna().unique())
    if not dates:
        return None, None, None, None
    latest = dates[-1]
    prev = dates[-2] if len(dates) >= 2 else None
    curr_df = df[df['Snapshot Date'] == latest].copy()
    prev_df = df[df['Snapshot Date'] == prev].copy() if prev is not None else pd.DataFrame()
    return latest, prev, curr_df, prev_df

def cards_row(curr_df: pd.DataFrame):
    counts = curr_df['Status'].value_counts().reindex(STATUS_ORDER, fill_value=0)
    cols = st.columns(5)
    for i, status in enumerate(STATUS_ORDER):
        with cols[i]:
            css = STATUS_TO_CLASS[status]
            st.markdown(f"""
            <div class="card">
              <span class="label {css}">{status}</span>
              <div class="count">{int(counts[status])}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Filter", key=f"filter_{status}", help=f"Show {status} listings"):
                st.session_state['voc_filter_status'] = status

def diff_banner(curr_df: pd.DataFrame, prev_df: pd.DataFrame, latest, prev):
    if prev_df is None or prev_df.empty:
        return
    # Status changes
    prev_s = prev_df[['ASIN','Status']].dropna()
    curr_s = curr_df[['ASIN','Status']].dropna()
    merged = prev_s.merge(curr_s, on='ASIN', how='inner', suffixes=('_prev','_curr'))
    changed = merged[merged['Status_prev'] != merged['Status_curr']].copy()

    # Direction of change
    rank = {s:i for i,s in enumerate(STATUS_ORDER)}
    def _dir(row):
        a, b = rank.get(row['Status_prev'], 999), rank.get(row['Status_curr'], 999)
        if a < b: return "Improved"
        if a > b: return "Worsened"
        return "Changed"
    if not changed.empty:
        changed['Direction'] = changed.apply(_dir, axis=1)

    # Missing from categories (yesterday → today)
    miss_by_cat = {}
    for s in STATUS_ORDER:
        y_set = set(prev_df[prev_df['Status']==s]['ASIN'].astype(str))
        t_set = set(curr_df[curr_df['Status']==s]['ASIN'].astype(str))
        miss_by_cat[s] = sorted(list(y_set - t_set))

    # Missing entirely (ASIN gone from file)
    missing_asins = sorted(list(set(prev_df['ASIN']) - set(curr_df['ASIN'])))

    total_drops = sum(len(v) for v in miss_by_cat.values())
    if len(changed) or total_drops or len(missing_asins):
        msg = f"""<div class="banner">
        <b>Updates since {pd.to_datetime(prev).date()} → {pd.to_datetime(latest).date()}:</b><br/>
        {len(changed)} ASIN(s) changed status. {total_drops} category drop(s). {len(missing_asins)} missing from today's file.
        </div>"""
        st.markdown(msg, unsafe_allow_html=True)
        with st.expander("View details"):
            if len(changed):
                st.markdown("**Status changes**")
                st.dataframe(changed[['ASIN','Status_prev','Status_curr','Direction']])
            lost_rows = []
            for s in STATUS_ORDER:
                for a in miss_by_cat[s]:
                    lost_rows.append({"Category (yesterday)": s, "ASIN": a})
            if lost_rows:
                st.markdown("**Moved out of categories (compared to yesterday)**")
                st.dataframe(pd.DataFrame(lost_rows))
            if missing_asins:
                st.markdown("**Missing from today's file**")
                st.write(", ".join(missing_asins))

# ── Upload dialog ────────────────────────────────────────────────────────────────
@st.dialog("Upload Voice of the Customer Report")
def upload_dialog(spreadsheet):
    st.write("Upload the latest VOC report. A snapshot will be saved for the selected date.")
    snap_date = st.date_input("Snapshot Date", value=date.today())
    file = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx","xls"])
    st.caption("Expected headers: " + ", ".join([c for c in REQUIRED_COLS if c!='Snapshot Date']))

    if st.button("Submit"):
        if not file:
            st.warning("Please select a file.")
            return
        # Read file
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        # Column name normalization (case-insensitive exact match to required)
        def match(col):
            c = str(col).strip()
            for req in REQUIRED_COLS:
                if req == 'Snapshot Date': 
                    continue
                if c.lower() == req.lower():
                    return req
            return c
        df.columns = [match(c) for c in df.columns]

        # Ensure required columns exist
        for c in REQUIRED_COLS:
            if c == 'Snapshot Date': 
                continue
            if c not in df.columns:
                df[c] = ""

        # Clean fields for accurate counts
        df['ASIN'] = df['ASIN'].apply(clean_asin)
        df['Status'] = df['Status'].apply(clean_status)

        # Within this uploaded file: keep 1 row per ASIN, preferring "worst" status
        rank = {s:i for i,s in enumerate(STATUS_ORDER)}  # 0=Very poor (worst)
        df['_rank'] = df['Status'].map(rank).fillna(999)
        df = df.sort_values('_rank').drop_duplicates(subset=['ASIN'], keep='first').drop(columns=['_rank'])

        # Add Snapshot Date
        df['Snapshot Date'] = pd.to_datetime(snap_date)

        # Order + save
        df = df[REQUIRED_COLS].copy()
        df['Snapshot Date'] = pd.to_datetime(df['Snapshot Date']).dt.tz_localize(None)

        # (Optional) quick sanity info
        raw_rows = (df['Status'] == 'Excellent').sum()
        uniq_asins = df.loc[df['Status']=='Excellent','ASIN'].nunique()
        st.info(f"Uploaded snapshot — Excellent rows: {raw_rows}, unique ASINs: {uniq_asins}")

        save_snapshot(spreadsheet, df)
        st.success("Snapshot saved.")
        st.rerun()

# ── Main ────────────────────────────────────────────────────────────────────────
spreadsheet = connect_to_gsheet()
ensure_sheet_with_columns(spreadsheet, SHEET_NAME, REQUIRED_COLS)

col_btn, _ = st.columns([1,3])
with col_btn:
    if st.button("⬆️ Upload Report"):
        upload_dialog(spreadsheet)

snapshots = load_snapshots(spreadsheet)
latest, prev, curr_df, prev_df = get_latest_and_previous(snapshots)

if curr_df is None or curr_df.empty:
    st.info("No snapshots yet. Upload your first Voice of the Customer report to begin.")
    st.stop()

# Alerts vs previous snapshot (latest vs previous date)
diff_banner(curr_df, prev_df, latest, prev)

# Status cards and filter buttons
st.subheader("CX Health breakdown of your listings")
cards_row(curr_df)

st.markdown("---")

# Filtered table (Filter buttons set session-state key)
filter_status = st.session_state.get("voc_filter_status")
if filter_status:
    st.markdown(f"**Filtered by Status:** {filter_status}  &nbsp;  ")
    if st.button("Clear filter"):
        st.session_state.pop("voc_filter_status", None)
        st.rerun()
    show_df = curr_df[curr_df['Status'] == filter_status].copy()
else:
    show_df = curr_df.copy()

# Nice ordering in table
cols_order = [
    'Product name','ASIN','SKU','Condition','Fulfilled by','NCX rate','NCX orders',
    'Total orders','Star rating','Return rate','Top NCX reason','Last updated',
    'Status','Return Badge Displayed'
]
cols_order = [c for c in cols_order if c in show_df.columns]
st.dataframe(show_df[cols_order], use_container_width=True)
