# pages/1_Return_Fraud_Analysis.py

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
# --- Page Configuration (must be first Streamlit call) ---
st.set_page_config(page_title="Return Fraud Analysis", layout="wide")

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

st.title("Return Fraud Analysis")

# --- Helper Functions ---
def connect_to_gsheet():
    """Connects to Google Sheets and returns the spreadsheet object."""
    try:
        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        return gc.open("Dashboard_data")
    except Exception as e:
        st.error(f"Connection Error: {e}")
        st.stop()

def ensure_sheet_with_columns(spreadsheet, sheet_name, required_cols):
    """Create the sheet if missing and ensure required headers exist."""
    try:
        ws = spreadsheet.worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=sheet_name, rows="2000", cols="20")
        empty = pd.DataFrame(columns=required_cols)
        set_with_dataframe(ws, empty, include_index=False, include_column_header=True, resize=True)
        return ws

    try:
        df = get_as_dataframe(ws, evaluate_formulas=True).dropna(how='all')
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        df = pd.DataFrame(columns=required_cols)
    else:
        for c in required_cols:
            if c not in df.columns:
                df[c] = ""
        # optional: keep required columns first
        other = [c for c in df.columns if c not in required_cols]
        df = df[required_cols + other]

    set_with_dataframe(ws, df, include_index=False, include_column_header=True, resize=True)
    return ws

def update_raw_sheet(worksheet, new_df, unique_column):
    """Appends new data to a raw data sheet and de-duplicates it."""
    try:
        existing_df = get_as_dataframe(worksheet, evaluate_formulas=True).dropna(how='all')
    except (gspread.exceptions.WorksheetNotFound, ValueError):
        existing_df = pd.DataFrame()

    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.columns = combined_df.columns.astype(str)
    final_df = combined_df.drop_duplicates(subset=[unique_column], keep='last')
    worksheet.clear()
    set_with_dataframe(worksheet, final_df, include_index=False, include_column_header=True, resize=True)

def get_date_range_from_sheets(spreadsheet):
    """
    Returns (start_date, end_date) by trying, in order:
    1) Fulfilled-Sales-Data['Date']
    2) Fulfilled-Sales-Data['Payments Date']
    3) FBA-Returns-Data['Date'] (derived from 'return-date')
    Falls back to last 30 days if none present.
    """
    # Try sales sheet
    try:
        sales_ws = spreadsheet.worksheet("Fulfilled-Sales-Data")
        sales_df = get_as_dataframe(sales_ws).dropna(how='all')
    except Exception:
        sales_df = pd.DataFrame()

    date_series = pd.Series(dtype="datetime64[ns]")

    if not sales_df.empty:
        if 'Date' in sales_df.columns:
            date_series = pd.to_datetime(sales_df['Date'], errors='coerce')
        elif 'Payments Date' in sales_df.columns:
            date_series = pd.to_datetime(sales_df['Payments Date'], errors='coerce')

    # Fallback: returns sheet
    if date_series.dropna().empty:
        try:
            returns_ws = spreadsheet.worksheet("FBA-Returns-Data")
            returns_df = get_as_dataframe(returns_ws).dropna(how='all')
            if 'Date' in returns_df.columns:
                date_series = pd.to_datetime(returns_df['Date'], errors='coerce')
        except Exception:
            pass

    date_series = date_series.dropna()
    if date_series.empty:
        start_date = (datetime.utcnow() - timedelta(days=30)).date()
        end_date = datetime.utcnow().date()
    else:
        start_date = date_series.dt.tz_localize(None).dt.date.min()
        end_date = date_series.dt.tz_localize(None).dt.date.max()

    return start_date, end_date

def generate_and_clean_data(spreadsheet):
    """Single source of truth for loading, cleaning, and joining data."""
    try:
        sales_ws = spreadsheet.worksheet("Fulfilled-Sales-Data")
        returns_ws = spreadsheet.worksheet("FBA-Returns-Data")

        sales_df = get_as_dataframe(sales_ws).dropna(how='all')
        returns_df = get_as_dataframe(returns_ws).dropna(how='all')
        
        if sales_df.empty or returns_df.empty:
            return pd.DataFrame()

        # Merge keys as strings
        sales_df['Order Id'] = sales_df['Order Id'].astype(str)
        returns_df['Order Id'] = returns_df['Order Id'].astype(str)

        # Use the returns 'Date' (populated from 'return-date')
        returns_df['Date'] = pd.to_datetime(returns_df['Date'], errors='coerce')
        returns_df.dropna(subset=['Date'], inplace=True)
        returns_df['Date'] = returns_df['Date'].dt.tz_localize(None)

        # We don't need Sales 'Date' in the combined sheet (retain only in sales sheet)
        sales_df = sales_df.drop(columns=['Date'], errors='ignore')

        # Join: Date comes from returns_df
        joined_df = pd.merge(returns_df, sales_df, on='Order Id', how='left')
        return joined_df

    except (gspread.exceptions.WorksheetNotFound, ValueError):
        return pd.DataFrame()


# --- Main App Logic ---
spreadsheet = connect_to_gsheet()

# Ensure required sheets/columns exist to avoid KeyErrors
RETURNS_REQUIRED = ['Order Id','ASIN','detailed-disposition','reason','customer-comments','return-date','Date']
SALES_REQUIRED   = ['Order Id','Date','Buyer Email','Title','Shipping Postal Code']

ensure_sheet_with_columns(spreadsheet, "FBA-Returns-Data", RETURNS_REQUIRED)
ensure_sheet_with_columns(spreadsheet, "Fulfilled-Sales-Data", SALES_REQUIRED)

# Initialize session state
if 'df' not in st.session_state:
    with st.spinner("Loading and processing data..."):
        st.session_state.df = generate_and_clean_data(spreadsheet)
        if not st.session_state.df.empty:
            set_with_dataframe(spreadsheet.worksheet("Returns-Sales-Comb-Data"), st.session_state.df)

# --- Upload Dialog ---
@st.dialog("Upload Amazon Reports")
def upload_dialog():
    # show current date range (robust across missing columns)
    start_date, end_date = get_date_range_from_sheets(spreadsheet)
    st.write("New data will be added to raw data sheets, and the combined report will be regenerated.")
    st.write(f"Current data date range: {start_date.strftime('%d-%b-%Y')} to {end_date.strftime('%d-%b-%Y')}")
    st.markdown("---")
    
    # FBA Returns Uploader with Instructions
    returns_file = st.file_uploader("Upload FBA Returns Report", type=['txt', 'csv'])
    st.markdown("Path: [Seller Central](https://sellercentral.amazon.in) → Reports → Fulfillment → Customer Concessions → FBA customer returns")
    st.markdown("---")

    # Fulfilled Sales Uploader with Instructions
    sales_file = st.file_uploader("Upload Amazon Fulfilled Sales Report", type=['txt', 'csv'])
    st.markdown("Path: [Seller Central](https://sellercentral.amazon.in) → Reports → Fulfillment → Sales → Amazon fulfilled shipments")
    st.markdown("---")

    if st.button("Submit"):
        if returns_file and sales_file:
            with st.spinner("Processing and updating sheets..."):
                # 1) Update Raw Returns Sheet (write Date from 'return-date')
                returns_cols = ['order-id', 'asin', 'detailed-disposition', 'reason', 'customer-comments', 'return-date']
                new_returns_df = pd.read_csv(returns_file, sep=',', usecols=returns_cols)
                new_returns_df.rename(columns={'order-id': 'Order Id', 'asin': 'ASIN'}, inplace=True)
                new_returns_df['Date'] = (
                    pd.to_datetime(new_returns_df['return-date'], errors='coerce')
                      .dt.tz_localize(None)
                )
                update_raw_sheet(spreadsheet.worksheet("FBA-Returns-Data"), new_returns_df, 'Order Id')
                
                # 2) Update Raw Sales Sheet (keep Date there for widgets; combined uses returns Date)
                sales_cols = ['Amazon Order Id', 'Payments Date', 'Buyer Email', 'Title', 'Shipping Postal Code']
                new_sales_df = pd.read_csv(sales_file, sep=',', usecols=sales_cols)
                new_sales_df.rename(columns={'Amazon Order Id': 'Order Id', 'Payments Date': 'Date'}, inplace=True)
                update_raw_sheet(spreadsheet.worksheet("Fulfilled-Sales-Data"), new_sales_df, 'Order Id')

                # 3) Regenerate combined data from the updated raw sheets
                st.session_state.df = generate_and_clean_data(spreadsheet)
                
                # 4) Save the newly generated combined data
                if not st.session_state.df.empty:
                    set_with_dataframe(spreadsheet.worksheet("Returns-Sales-Comb-Data"), st.session_state.df)
                
                st.success("Data updated!")
                st.rerun()
        else:
            st.warning("Please upload both files before submitting.")

if st.button("⬆️ Upload New Reports"):
    upload_dialog()

st.markdown("---")

# --- Display All Content: Metrics and Insights ---
if 'df' not in st.session_state or st.session_state.df.empty:
    st.info("No data available. Please upload reports to begin analysis.")
else:
    df = st.session_state.df.copy()
    # Convert to datetime for filtering
    df['Date'] = pd.to_datetime(df['Date'])

    # Robust date range from sheets (uses sales Date, or Payments Date, or returns Date)
    start_date, end_date = get_date_range_from_sheets(spreadsheet)

    excluded_reasons = ['UNDELIVERABLE_REFUSED', 'UNDELIVERABLE_UNKNOWN']
    # Use .str.upper() for case-insensitive matching
    filtered_df_for_reason = df[~df['reason'].str.upper().isin(excluded_reasons)]
    top_return_reason = filtered_df_for_reason['reason'].mode()[0] if not filtered_df_for_reason.empty else "N/A"

    # Display Metrics
    # Initialize date range in session state if not already set
    if 'selected_date_range' not in st.session_state:
        st.session_state.selected_date_range = (start_date, end_date)

    col_date_input, col_clear = st.columns([4, 1])

    with col_date_input:
        selected_date_range = st.date_input(
            "Filter by Date Range",
            value=st.session_state.selected_date_range,
            min_value=start_date,
            max_value=end_date,
            key="date_filter"
        )

        # ✅ Update session state if user changes date
        if selected_date_range != st.session_state.selected_date_range:
            st.session_state.selected_date_range = selected_date_range

        # Filter df based on selected date range
        start_sel, end_sel = st.session_state.selected_date_range
        start_sel = pd.to_datetime(start_sel)
        end_sel = pd.to_datetime(end_sel)
        df = df[(df['Date'] >= start_sel) & (df['Date'] <= end_sel)]

    with col_clear:
        st.markdown("<div style='padding-top: 2.2rem;'></div>", unsafe_allow_html=True)
        if st.button("Clear Date Filter"):
            st.session_state.selected_date_range = (start_date, end_date)
            st.rerun()

    # Display selected range next to widget
    col_range_info = st.columns([1])[0]
    with col_range_info:
        st.markdown(f"<div style='padding-top: 2rem;'><b>Sales Data Range:</b> {start_sel.strftime('%d-%b-%Y')} to {end_sel.strftime('%d-%b-%Y')}</div>",unsafe_allow_html=True)

    # If user selected a valid range (it's always a tuple)
    total_returns = len(df.drop_duplicates(subset=['Order Id']))
    # Top 3 most returned ASINs
    top_asins_series = df['ASIN'].value_counts().head(3)
    top_asins_df = pd.DataFrame({'ASIN': top_asins_series.index, 'Return Count': top_asins_series.values})

    # Get the titles
    if 'Title' in df.columns:
        asin_titles = df.groupby('ASIN')['Title'].first()
        top_asins_df['Title'] = top_asins_df['ASIN'].map(asin_titles).fillna("No title")

    # Truncate long titles
    top_asins_df['Title'] = top_asins_df['Title'].apply(lambda x: (x[:45] + '...') if len(x) > 45 else x)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- Row 1: Top 3 Most Returned ASINs ---
    st.markdown("Top 3 Most Returned ASINs")

    asin_cols = st.columns(3)
    for col, (_, row) in zip(asin_cols, top_asins_df.iterrows()):
        with col:
            st.markdown(f"**`{row['ASIN']}`**")
            st.markdown(f"**{row['Return Count']}** returns")
            st.caption(row['Title'])

    st.markdown("---")
    def get_top_asin_for_reason(df_local, reason_code):
        filtered = df_local[df_local['reason'].str.upper() == reason_code]
        if filtered.empty:
            return ("N/A", 0)
        top_asin = filtered['ASIN'].value_counts().idxmax()
        count = filtered['ASIN'].value_counts().max()
        return top_asin, count

    # Calculate top ASINs for each reason
    top_defective_asin, defective_count = get_top_asin_for_reason(df, "DEFECTIVE")
    top_not_desc_asin, not_desc_count = get_top_asin_for_reason(df, "NOT_AS_DESCRIBED")

    # --- Row 2: Supporting Metrics ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📦 Total Unique Returns", f"{total_returns:,}")

    with col2:
        st.metric("🛠️ Most DEFECTIVE Returns", f"{top_defective_asin} ({defective_count})")

    with col3:
        st.metric("📦 Most NOT_AS_DESCRIBED", f"{top_not_desc_asin} ({not_desc_count})")

    # --- Display Detailed Insights by Default ---
    st.markdown("---")
    st.subheader("Detailed Analysis")

    with st.expander("📍 Geographic Return Clusters"):
        df['Date'] = pd.to_datetime(df['Date'])  # Ensure it's datetime for comparison

        suspicious_reasons = ['DEFECTIVE', 'NOT_AS_DESCRIBED', 'WRONG_ITEM_SENT',
                              'MISSING_PARTS', 'UNDELIVERABLE_UNKNOWN', 'SWITCHEROO',
                              'DAMAGED_BY_CARRIER', 'NO_REASON_GIVEN']
        
        suspicious_returns = df[df['reason'].isin(suspicious_reasons)].copy()

        # Count number of returns per pincode
        return_counts = df['Shipping Postal Code'].value_counts()

        # Filter pincodes with 2 or more returns
        frequent_pincodes = return_counts[return_counts >= 2].index.tolist()

        # Filter original DataFrame
        flagged_clusters = df[df['Shipping Postal Code'].isin(frequent_pincodes)].copy()

        # Sort for easier reading
        flagged_clusters.sort_values(by=['Shipping Postal Code', 'Date'], inplace=True)

        if not flagged_clusters.empty:
            st.markdown("Returns from pincodes with 2 or more returns:")
            
            df2 = flagged_clusters.copy()
            df2['Date'] = df2['Date'].dt.date
            df2['Shipping Postal Code'] = df2['Shipping Postal Code'].astype(str).str.split('.').str[0]  # Ensure consistency for filtering

            # Create dropdown filters inline with headers
            colf1, colf2, colf3, colf4, colf5 = st.columns(5)

            with colf1:
                st.markdown("**Date**")
                date_filter = st.selectbox(" ", options=["All"] + sorted(df2['Date'].unique().tolist()), label_visibility="collapsed")

            with colf2:
                st.markdown("**Pincode**")
                pincode_filter = st.selectbox("  ", options=["All"] + sorted(df2['Shipping Postal Code'].unique().tolist()), label_visibility="collapsed")

            with colf3:
                st.markdown("**Order ID**")
                order_filter = st.selectbox("   ", options=["All"] + sorted(df2['Order Id'].unique().tolist()), label_visibility="collapsed")

            with colf4:
                st.markdown("**ASIN**")
                asin_filter = st.selectbox("    ", options=["All"] + sorted(df2['ASIN'].astype(str).unique().tolist()), label_visibility="collapsed")

            with colf5:
                st.markdown("**Reason**")
                reason_filter = st.selectbox("     ", options=["All"] + sorted(df2['reason'].unique().tolist()), label_visibility="collapsed")

            # Apply filters
            filtered_df = df2.copy()
            if date_filter != "All":
                filtered_df = filtered_df[filtered_df['Date'] == date_filter]
            if pincode_filter != "All":
                filtered_df = filtered_df[filtered_df['Shipping Postal Code'] == pincode_filter]
            if order_filter != "All":
                filtered_df = filtered_df[filtered_df['Order Id'] == order_filter]
            if asin_filter != "All":
                filtered_df = filtered_df[filtered_df['ASIN'].astype(str) == asin_filter]
            if reason_filter != "All":
                filtered_df = filtered_df[filtered_df['reason'] == reason_filter]

            # ✅ Show count of filtered rows
            st.markdown(f"**Total Rows:** {len(filtered_df)}")

            # Show filtered table
            st.dataframe(filtered_df[['Date', 'Shipping Postal Code', 'Order Id', 'ASIN', 'reason']], use_container_width=True)
        else:
            st.info("No pincodes with 2 or more returns found.")

    with st.expander("📈 ASIN Return Rate Spikes"):
        df['ASIN'] = df['ASIN'].astype(str)  # Ensure ASINs are strings
        df_dated = df.set_index(pd.to_datetime(df['Date']))
        monthly_returns = df_dated.groupby('ASIN').resample('ME')['Order Id'].nunique()
        
        spikes = monthly_returns[monthly_returns > 5].reset_index()
        spikes.columns = ['ASIN', 'Month', 'Return Count']
        spikes['Month'] = spikes['Month'].dt.strftime('%Y-%B')
        st.write("ASINs with more than 5 returns in a single month:")
        st.dataframe(spikes)

    with st.expander("🤔 Suspicious Return Reasons"):
        suspicious_df2 = df[
            (df['reason'] != 'UNDELIVERABLE_REFUSED') & 
            (df['detailed-disposition'] == 'SELLABLE')
        ]
        st.dataframe(suspicious_df2[['Date', 'Order Id', 'ASIN', 'Title', 'reason', 'customer-comments']])

    # --- Data Preview ---
    st.markdown("---")
    st.subheader("Combined Data Preview")
    preview_df = df.copy()
    preview_df['Date'] = pd.to_datetime(preview_df['Date']).dt.strftime('%Y-%m-%d')
    st.dataframe(preview_df)
