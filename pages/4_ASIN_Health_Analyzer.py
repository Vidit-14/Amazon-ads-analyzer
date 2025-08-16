import streamlit as st
import pandas as pd
import gspread
import numpy as np
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from datetime import datetime, timedelta, date
import altair as alt, math
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
st.markdown("""
<style>
    /* Fancy Title CSS */
    h1 {
        background: -webkit-linear-gradient(#00BFFF, #1b7371);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 4px 4px 6px rgba(0,0,0,0.1);
        font-weight: 800;
        font-size: 3.75rem;
        padding: 0.8rem 0rem 1rem; 
    }
</style>
""", unsafe_allow_html=True)

# --- Page Configuration ---
st.set_page_config(page_title="ASIN Health Dashboard", layout="wide")
st.title("ASIN Health Dashboard")

# --- Static Data: Checklists ---
CHECKLISTS = {
    "High Traffic / Low Conv": [
        "Quick sanity checks (Buy Box > 95%, In-stock 100%, Price vs peers)",
        "Relevance audit (Search term & listing keyword alignment)",
        "Primary image & title optimization (Clarity, benefits, keywords)",
        "First 3 bullets review (Objection-crushing, outcome-focused)",
        "A+ Content audit (Comparison charts, visual plans, FAQs)",
        "'Look Inside' / interior images review",
        "Social proof check (Reviews, endorsements)",
        "Offer tests (Coupons, price experiments)",
        "Variation hygiene (Merge duplicates, set best-seller as default)",
        "Ad strategy review (Shift budget from discovery to conversion)",
        "Category-specific must-haves check (e.g., 'Updated for 2025')"
    ],
    "Low Traffic / Low Conv": [
        "Keyword & relevance overhaul (Research and update all text)",
        "Fix core listing elements for conversion (Title, Image, Bullets, A+)",
        "Improve trust signals (Get more verified reviews, add editorial reviews)",
        "Run targeted ads to kickstart traffic (Sponsored Products - Exact Match)",
        "External traffic push (Social media, forums, Amazon Attribution)",
        "Promotional hooks (Launch with coupons or limited-time deals)",
        "Monitor & Iterate (Track changes in Traffic/Conversion flags)",
    ]
}

# --- Helper Functions ---

@st.cache_resource
def connect_to_gsheet():
    """Connects to Google Sheets and returns the spreadsheet object."""
    try:
        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        return gc.open("Dashboard_data")
    except Exception as e:
        st.error(f"Connection Error: {e}")
        st.stop()
        return None

def ensure_checklist_sheet_exists(spreadsheet):
    """Ensure 'ASIN_Checklist_Status' exists; create with headers if missing."""
    sheet_name = "ASIN_Checklist_Status"
    try:
        spreadsheet.worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        # Create a new sheet with headers
        ws = spreadsheet.add_worksheet(title=sheet_name, rows="1000", cols="10")
        header_df = pd.DataFrame(columns=['Month', 'ASIN', 'Checklist Item', 'Status'])
        set_with_dataframe(ws, header_df)

@st.cache_data(ttl=600)
def load_health_data(_spreadsheet):
    """Loads the raw ASIN Health data."""
    try:
        ws = _spreadsheet.worksheet("ASIN_Health_Data")
        df = get_as_dataframe(ws, evaluate_formulas=True).dropna(how='all')
        return df
    except (gspread.exceptions.WorksheetNotFound, ValueError):
        st.warning("'ASIN_Health_Data' worksheet not found or is empty.")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def load_checklist_status(_spreadsheet):
    required_cols = ['Month','ASIN','Checklist Item','Status']
    ensure_checklist_sheet_exists(_spreadsheet)
    try:
        ws = _spreadsheet.worksheet("ASIN_Checklist_Status")
        df = get_as_dataframe(ws, evaluate_formulas=True).dropna(how='all')
        if df.empty:
            return pd.DataFrame(columns=required_cols)

        for c in required_cols:
            if c not in df.columns:
                df[c] = "" if c != "Status" else False

        # Normalize types consistently
        df['Month'] = _norm_month_str(df['Month'])
        df['ASIN'] = df['ASIN'].astype(str)
        df['Checklist Item'] = df['Checklist Item'].astype(str)

        def to_bool(x):
            if isinstance(x, bool): return x
            return str(x).strip().lower() in ("true","t","1","yes","y")
        df['Status'] = df['Status'].apply(to_bool)

        return df[required_cols]
    except Exception:
        return pd.DataFrame(columns=required_cols)


def _norm_month_str(series):
    """Converts a pandas Series to 'YYYY-MM' string format, ignoring errors."""
    return pd.to_datetime(series, errors='coerce').dt.strftime('%Y-%m')

def update_checklist_status(spreadsheet, month, asin, all_items, checklist_type, updated_by=""):
    """Overwrite the (month, asin) block with the current form state."""
    try:
        ensure_checklist_sheet_exists(spreadsheet)
        ws = spreadsheet.worksheet("ASIN_Checklist_Status")
        status_df = load_checklist_status(spreadsheet).copy()

        # Ensure Month is normalized the same way we write it
        status_df['Month'] = _norm_month_str(status_df['Month'])

        # Build fresh block from current form
        new_rows = []
        for i, item_text in enumerate(all_items):
            key = f"check_{month}_{asin}_{i}"
            new_status = bool(st.session_state.get(key, False))
            new_rows.append({
                "Month": str(month),  # already 'YYYY-MM' from UI
                "ASIN": str(asin),
                "Checklist Item": str(item_text),
                "Status": new_status,
            })
        new_block_df = pd.DataFrame(new_rows)

        # Drop old block, append new block, and de-duplicate
        mask = (status_df["Month"] == str(month)) & (status_df["ASIN"] == str(asin))
        status_df = status_df.loc[~mask]
        status_df = pd.concat([status_df, new_block_df], ignore_index=True)
        status_df = status_df.drop_duplicates(subset=['Month','ASIN','Checklist Item'], keep='last')
        status_df['Status'] = status_df['Status'].astype(bool)

        # Write back and resize sheet to avoid stale rows
        set_with_dataframe(
            ws,
            status_df[['Month','ASIN','Checklist Item','Status']],
            include_index=False,
            include_column_header=True,
            resize=True
        )

        st.cache_data.clear()
        st.toast("Checklist saved (overwritten).", icon="✅")
        st.rerun()

    except Exception as e:
        st.error(f"Failed to update checklist status: {e}")


def calculate_metrics(df, target_usp):
    """Calculates all the derived metric columns."""
    # --- List of columns that this function will create ---
    calculated_cols = [
        'ASP ()', 'Traffic Flag', 'Conversion Flag', 'Buy Box Flag', 'Opportunity',
        'Potential Units Gain', 'Potential Revenue Gain ()', 'Priority Score'
    ]
    # Drop these columns if they already exist to prevent duplicates
    df = df.drop(columns=calculated_cols, errors='ignore')

    # Ensure numeric types for calculation, coercing errors
    numeric_cols = ['Ordered Product Sales', 'Units Ordered', 'Sessions - Total', 'Unit Session Percentage', 'Featured Offer Percentage']

    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0
        else:
            cleaned_col = df[col].astype(str).str.replace(r'[₹,% ,]', '', regex=True)
            df[col] = pd.to_numeric(cleaned_col, errors='coerce').fillna(0)

    df['ASP ()'] = np.divide(df['Ordered Product Sales'], df['Units Ordered'], out=np.zeros_like(df['Ordered Product Sales'], dtype=float), where=df['Units Ordered']!=0)
    df['Traffic Flag'] = np.where(df['Sessions - Total'] < 5000, "Low Traffic", "OK")
    df['Conversion Flag'] = np.where(df['Unit Session Percentage'] < target_usp, "Low Conversion", "OK")
    df['Buy Box Flag'] = np.where(df['Featured Offer Percentage'] < 0.9, "Buy Box Loss", "OK")
    # Use consistent Opportunity label
    df['Opportunity'] = np.where((df['Sessions - Total'] >= 5000) & (df['Unit Session Percentage'] < target_usp), "High Traffic / Low Conv", "")

    potential_gain = (target_usp - df['Unit Session Percentage']) * df['Sessions - Total']
    df['Potential Units Gain'] = potential_gain.clip(lower=0)
    df['Potential Revenue Gain ()'] = df['Potential Units Gain'] * df['ASP ()']

    if_factor = np.where(df['Sessions - Total'] < 5000, 0.8, 1)
    df['Priority Score'] = df['Potential Revenue Gain ()'] * df['Featured Offer Percentage'] * if_factor

    return df.sort_values(by='Priority Score', ascending=False)

# --- Upload Dialog ---
@st.dialog("Upload ASIN Health Report")
def upload_dialog(spreadsheet):
    """Handles the monthly data upload process."""
    st.write("Upload the raw data file. The script will add the selected month and update the backend sheet.")

    # Replace st.date_input with st.selectbox for month selection
    month_options = []
    current_date = datetime.now()
    for i in range(24):  # last 2 years
        year = current_date.year - (i // 12)
        month = current_date.month - (i % 12)
        if month <= 0:
            year -= 1
            month += 12
        month_options.append(datetime(year, month, 1).strftime('%Y-%m'))

    selected_month = st.selectbox(
        "Select the Month and Year for this data",
        options=month_options
    )

    uploaded_file = st.file_uploader("Upload Raw Data File", type=['csv', 'xlsx'])

    # --- ALWAYS show whether last month's data is available + path below uploader ---
    try:
        ws_check = spreadsheet.worksheet("ASIN_Health_Data")
        existing_df_check = get_as_dataframe(ws_check).dropna(how='all')
    except Exception:
        existing_df_check = pd.DataFrame()

    today_dt = date.today()
    last_month_dt = (today_dt.replace(day=1) - timedelta(days=1))
    last_month_key = last_month_dt.strftime("%Y-%m")
    last_month_label = last_month_dt.strftime("%B %Y")

    has_last_month = (
        not existing_df_check.empty
        and 'Month' in existing_df_check.columns
        and (last_month_key in existing_df_check['Month'].astype(str).values)
    )
    if has_last_month:
        st.success(f"Last month's data ({last_month_label}) is already available.")
    else:
        st.info(f"Last month's data for **{last_month_label}** is not available yet. Please upload it.")

    seller_url = "https://sellercentral.amazon.in/"
    st.markdown(
        f"**Path:** [Seller Central]({seller_url}) → Reports → Business Reports → By ASIN →  \n"
        f"Detail Page Sales and Traffic By Child Item→ Date: **{last_month_label}**"
    )

    if st.button("Submit"):
        if uploaded_file and selected_month:
            with st.spinner("Processing and updating sheets..."):
                try:
                    if uploaded_file.name.endswith('.csv'):
                        new_df = pd.read_csv(uploaded_file)
                    else:
                        new_df = pd.read_excel(uploaded_file)

                    new_df['Month'] = selected_month

                    ws = spreadsheet.worksheet("ASIN_Health_Data")
                    existing_df = load_health_data(spreadsheet)

                    # Remove any old data for the selected month before appending
                    if not existing_df.empty:
                        existing_df = existing_df[existing_df['Month'] != selected_month]

                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    set_with_dataframe(ws, combined_df)

                    st.cache_data.clear()
                    st.success("Data updated successfully!")
                    st.rerun()

                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please select a month and upload a file.")


# --- View Rendering Functions ---

def render_main_view(health_df, checklist_df, spreadsheet):
    """Renders the main table view with filters."""
    if st.button("⬆️ Upload Monthly Report"):
        upload_dialog(spreadsheet)

    if health_df.empty:
        st.info("No data available. Please upload a report to begin.")
        return pd.DataFrame()

    # Top Level Filters
    col1, col2 = st.columns([1, 2])
    with col1:
        month_options = sorted(health_df['Month'].astype(str).unique(), reverse=True)
        default_months = [month_options[0]] if month_options else []
        # ---- NEW: multiselect for months ----
        selected_months = st.multiselect("Select month(s) to analyze", options=month_options, default=default_months, key="selected_health_month_filter")
    with col2:
        target_usp = st.number_input(
            "Target Unit Session Percentage (e.g., 0.08 for 8%)",
            min_value=0.0, max_value=1.0, value=0.08, step=0.01, format="%.4f"
        )
        st.session_state['target_usp'] = target_usp  # store for checklist view

    st.markdown("---")

    # Data Processing for Selected Months (aggregate across months)
    if not selected_months:
        st.info("Please select at least one month.")
        return pd.DataFrame()

    month_df = health_df[health_df['Month'].astype(str).isin(selected_months)].copy()
    if month_df.empty:
        st.info(f"No data found for the selected months.")
        return pd.DataFrame()

    # Determine ASIN column and ensure Title column exists
    asin_col_name = '(Child) ASIN' if '(Child) ASIN' in month_df.columns else ('ASIN' if 'ASIN' in month_df.columns else None)
    if not asin_col_name:
        st.error("Fatal Error: Could not find a valid ASIN column in the data.")
        st.stop()
    if 'Title' not in month_df.columns:
        month_df['Title'] = ""

    # Coerce raw numeric fields
    def to_num(s):
        return pd.to_numeric(s.astype(str).str.replace(r'[₹,% ,]', '', regex=True), errors='coerce').fillna(0)

    for c in ['Ordered Product Sales', 'Units Ordered', 'Sessions - Total', 'Unit Session Percentage', 'Featured Offer Percentage']:
        if c not in month_df.columns:
            month_df[c] = 0
        month_df[c] = to_num(month_df[c])

    # Aggregate across selected months per ASIN (sum counts, sessions-weighted percentages)
    def _agg_group(g):
        sessions = g['Sessions - Total'].sum()
        out = {
            'Ordered Product Sales': g['Ordered Product Sales'].sum(),
            'Units Ordered': g['Units Ordered'].sum(),
            'Sessions - Total': sessions,
            'Title': g['Title'].iloc[0],
        }
        out['Unit Session Percentage'] = (g['Unit Session Percentage'].mul(g['Sessions - Total']).sum() / sessions) if sessions > 0 else 0
        out['Featured Offer Percentage'] = (g['Featured Offer Percentage'].mul(g['Sessions - Total']).sum() / sessions) if sessions > 0 else 0
        return pd.Series(out)

    aggregated_df = month_df.groupby(asin_col_name, dropna=False).apply(_agg_group).reset_index().rename(columns={asin_col_name: 'ASIN'})

    # Recalculate metrics on the aggregated data
    calculated_df = calculate_metrics(aggregated_df.copy(), target_usp)

    # Filters
    st.markdown("##### Filter by ASIN and Title")
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        asin_options = sorted([str(asin) for asin in calculated_df['ASIN'].unique()])
        selected_asins = st.multiselect("Filter by ASIN", options=asin_options)
    with f_col2:
        title_options = sorted(list(calculated_df['Title'].unique())) if 'Title' in calculated_df.columns else []
        selected_titles = st.multiselect("Filter by Title", options=title_options)

    filtered_df = calculated_df.copy()
    if selected_asins:
        filtered_df = filtered_df[filtered_df['ASIN'].astype(str).isin(selected_asins)]
    if selected_titles and 'Title' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Title'].isin(selected_titles)]

    # --- NEW: Aggregated KPIs for the current selection ---
    tot_sales = filtered_df['Ordered Product Sales'].sum() if 'Ordered Product Sales' in filtered_df.columns else 0.0
    tot_units = filtered_df['Units Ordered'].sum() if 'Units Ordered' in filtered_df.columns else 0.0
    tot_sessions = filtered_df['Sessions - Total'].sum() if 'Sessions - Total' in filtered_df.columns else 0.0
    w_usp = (filtered_df['Unit Session Percentage'].mul(filtered_df['Sessions - Total']).sum() / tot_sessions) if tot_sessions > 0 else 0.0
    w_bb = (filtered_df['Featured Offer Percentage'].mul(filtered_df['Sessions - Total']).sum() / tot_sessions) if tot_sessions > 0 else 0.0
    asp = (tot_sales / tot_units) if tot_units > 0 else 0.0

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1: st.metric("Total Sales", f"₹{tot_sales:,.2f}")
    with m2: st.metric("Units Ordered", f"{int(tot_units):,}")
    with m3: st.metric("Sessions", f"{int(tot_sessions):,}")
    with m4: st.metric("Unit Session % (weighted)", f"{w_usp:.3f}")
    with m5: st.metric("Buy Box % (weighted)", f"{w_bb:.3f}")

    # Build display df
    raw_cols = [c for c in ['Ordered Product Sales','Units Ordered','Sessions - Total','Unit Session Percentage','Featured Offer Percentage'] if c in filtered_df.columns]
    calculated_cols = [
        'ASP ()', 'Traffic Flag', 'Conversion Flag', 'Buy Box Flag', 'Opportunity',
        'Potential Units Gain', 'Potential Revenue Gain ()', 'Priority Score'
    ]
    final_column_order = ['ASIN', 'Title'] + raw_cols + calculated_cols

    display_df = filtered_df.copy()
    existing_ordered_cols = [col for col in final_column_order if col in display_df.columns]
    display_df = display_df[existing_ordered_cols]

    num_cols = display_df.select_dtypes(include=[np.number]).columns
    display_df[num_cols] = display_df[num_cols].round(2)
    # --- Styling for flag columns (subtle red/green backgrounds) ---
    flag_cols = [c for c in ['Traffic Flag', 'Conversion Flag', 'Buy Box Flag'] if c in display_df.columns]

    def _flag_bg(val):
        if isinstance(val, str):
            v = val.lower()
            if v == "ok":
                return "background-color: rgba(0, 200, 0, 0.12);"
            if ("low" in v) or ("loss" in v):
                return "background-color: rgba(255, 0, 0, 0.12);"
        return ""

    styled = display_df.style.applymap(_flag_bg, subset=flag_cols)

    # Render with selection enabled (Streamlit supports Styler in st.dataframe)
    st.dataframe(
        styled,
        column_config={
            "ASIN": st.column_config.TextColumn(label="ASIN", width="medium"),
            "Title": st.column_config.TextColumn(label="Title", width="large"),
            "Potential Revenue Gain ()": st.column_config.NumberColumn(format="₹%.2f"),
            "ASP ()": st.column_config.NumberColumn(format="₹%.2f"),
        },
        on_select="rerun",
        selection_mode="single-row",
        hide_index=True,
        key="health_table_selection"
    )

    return display_df


def render_checklist_view(health_df, checklist_df, spreadsheet):
    """Renders the detailed checklist and charts for a selected ASIN."""
    asin = st.session_state.selected_health_asin
    month = st.session_state.selected_health_month  # this is now the most recent of the selected months

    if st.button("← Back to Main Dashboard"):
        st.session_state.selected_health_asin = None
        st.session_state.selected_health_month = None
        st.rerun()

    # Dynamically find the correct ASIN column name
    asin_col_name = '(Child) ASIN' if '(Child) ASIN' in health_df.columns else 'ASIN'
    asin_data_row = health_df[(health_df[asin_col_name] == asin) & (health_df['Month'] == month)]

    if asin_data_row.empty:
        st.error("Could not find data for the selected ASIN and month.")
        st.stop()

    asin_data = asin_data_row.iloc[0]

    checklist_type = "Low Traffic / Low Conv"
    if 'Opportunity' in asin_data.index and asin_data['Opportunity'] == "High Traffic / Low Conv":
        checklist_type = "High Traffic / Low Conv"

    st.header(f"Checklist for {asin}")
    st.subheader(f"Category: {checklist_type}")

    # Ensure checklist sheet exists and load a fresh copy (cache cleared on writes)
    ensure_checklist_sheet_exists(spreadsheet)
    checklist_df = load_checklist_status(spreadsheet)

    # If there are no rows for this (month, asin), create default rows so checkboxes show
    checklist_items = CHECKLISTS[checklist_type]

    current_status_df = checklist_df[
        (checklist_df['Month'] == str(month)) & (checklist_df['ASIN'] == str(asin))
    ]

    # Wrap the checklist in a form
    with st.form(key="checklist_form"):
        for i, item in enumerate(checklist_items):
            is_checked = False
            if not current_status_df.empty:
                row = current_status_df[current_status_df['Checklist Item'] == item]
                if not row.empty:
                    is_checked = bool(row['Status'].iloc[0])
            st.checkbox(item, value=is_checked, key=f"check_{month}_{asin}_{i}")

        if st.form_submit_button("Save Changes"):
            update_checklist_status(spreadsheet, month, asin, checklist_items, checklist_type)

    # --- Historical Charts (stacked, fixed scales) ---
    st.markdown("---")
    st.subheader("Historical Performance")

    history_df = health_df[health_df[asin_col_name] == asin].copy()
    history_df["Month"] = pd.to_datetime(history_df["Month"], errors="coerce")
    history_df = history_df.sort_values("Month")

    # Coerce to numeric
    if "Units Ordered" in history_df.columns:
        history_df["Units Ordered"] = pd.to_numeric(history_df["Units Ordered"], errors="coerce")

    # Strip currency and commas -> float
    if "Ordered Product Sales" in history_df.columns:
        history_df["OPS_num"] = (
            history_df["Ordered Product Sales"]
            .astype(str)
            .str.replace(r"[₹,]", "", regex=True)
            .astype(float)
        )

    # Domains (so the axis is correct and starts at 0)
    units_max = float(history_df.get("Units Ordered", pd.Series([0])).max())
    sales_max = float(history_df.get("OPS_num", pd.Series([0.0])).max())
    units_top = max(1, math.ceil(units_max * 1.15))
    sales_top = max(1, math.ceil(sales_max * 1.15))

    # 1) Units Ordered
    st.markdown("##### Monthly Units Ordered")
    if "Units Ordered" in history_df.columns and history_df["Units Ordered"].notna().any():
        units_chart = (
            alt.Chart(history_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("Month:T", title="Month"),
                y=alt.Y(
                    "Units Ordered:Q",
                    title="Units Ordered",
                    scale=alt.Scale(domain=[0, units_top]),
                ),
                tooltip=[
                    alt.Tooltip("Month:T"),
                    alt.Tooltip("Units Ordered:Q", format=".0f"),
                ],
            )
            .properties(height=360)   # expanded
        )
        st.altair_chart(units_chart, use_container_width=True)
    else:
        st.info("No 'Units Ordered' data found.")

    
# --- Main App Logic ---

spreadsheet = connect_to_gsheet()
health_df = load_health_data(spreadsheet)
checklist_df = load_checklist_status(spreadsheet)

# Router to display the correct view
if 'selected_health_asin' in st.session_state and st.session_state.selected_health_asin:
    render_checklist_view(health_df, checklist_df, spreadsheet)
else:
    displayed_health_df = render_main_view(health_df, checklist_df, spreadsheet)

    # Robust selection handling (works whether selection is stored as object or dict)
    selection_state = st.session_state.get("health_table_selection")
    selected_row_index = None
    if selection_state:
        # Try attribute-style access (Streamlit sometimes uses an object)
        try:
            sel = getattr(selection_state, "selection", None)
            if sel and getattr(sel, "rows", None):
                selected_row_index = sel.rows[0]
        except Exception:
            pass
        # Try dict-style access
        if selected_row_index is None:
            try:
                sel = selection_state.get("selection", {})
                rows = sel.get("rows", []) if isinstance(sel, dict) else []
                if rows:
                    selected_row_index = rows[0]
            except Exception:
                pass

    if selected_row_index is not None and not displayed_health_df.empty and selected_row_index < len(displayed_health_df):
        selected_asin = displayed_health_df.iloc[selected_row_index]['ASIN']

        st.markdown("---")

        if st.button(f"🔎 View Checklist for ASIN: {selected_asin}"):
            st.session_state.selected_health_asin = selected_asin
            # --- NEW: use the most recent month from the multiselect for the checklist ---
            sel_months = st.session_state.get("selected_health_month_filter", [])
            if isinstance(sel_months, list) and sel_months:
                st.session_state.selected_health_month = max(sel_months)  # 'YYYY-MM' sorts correctly
            else:
                # fallback: try to pick the latest from data
                possible_months = sorted(health_df['Month'].astype(str).unique())
                st.session_state.selected_health_month = possible_months[-1] if possible_months else None
            st.rerun()
