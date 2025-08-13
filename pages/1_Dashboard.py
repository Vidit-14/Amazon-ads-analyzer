# pages/Dashboard.py
import streamlit as st
import pandas as pd
import gspread
from datetime import date, timedelta, datetime
import numpy as np
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
# --- Custom CSS and Page Configuration ---
# Page config is inherited from app.py, but we can add to it.
# Here we inject CSS for the fancy title.
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


# --- Session State Initialization ---
if 'start_date' not in st.session_state:
    st.session_state.start_date = pd.Timestamp.now().normalize() - timedelta(days=29)
if 'end_date' not in st.session_state:
    st.session_state.end_date = pd.Timestamp.now().normalize()
if 'country' not in st.session_state:
    st.session_state.country = "India"
if 'selected_asin' not in st.session_state: 
    st.session_state.selected_asin = None
if 'selected_campaign' not in st.session_state: 
    st.session_state.selected_campaign = None


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

@st.cache_data(ttl=600)
def load_sales_data(_spreadsheet, sheet_name):
    """Loads and cleans the Amazon Orders report and returns Date, SKU, Total Sales, Total Units."""
    try:
        ws = _spreadsheet.worksheet(sheet_name)
        df = pd.DataFrame(ws.get_all_records())

        # Expecting columns from Orders report:
        # purchase-date, sku, asin, quantity, currency, item-price, item-tax,
        # shipping-price, shipping-tax, gift-wrap-price, gift-wrap-tax,
        # item-promotion-discount, ship-promotion-discount
        # Normalize column names just in case of stray spaces/case
        df.columns = [str(c).strip() for c in df.columns]

        # Date
        if 'purchase-date' in df.columns:
            df['Date'] = pd.to_datetime(df['purchase-date'], errors='coerce').dt.tz_localize(None)
            df.dropna(subset=['Date'], inplace=True)

        # SKU
        if 'sku' in df.columns:
            df['SKU'] = df['sku'].astype(str)

        # Quantity -> Total Units
        if 'quantity' in df.columns:
            df['Total Units'] = pd.to_numeric(
                df['quantity'].astype(str).str.replace(r'[^0-9\.-]', '', regex=True),
                errors='coerce'
            ).fillna(0)
        else:
            df['Total Units'] = 0

        if 'asin' in df.columns:
            df['ASIN'] = df['asin'].astype(str)

        # Helper to coerce numeric money fields
        def num(col):
            if col in df.columns:
                return pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9\.-]', '', regex=True), errors='coerce').fillna(0)
            return 0

        item_price              = num('item-price')
        item_tax                = num('item-tax')
        shipping_price          = num('shipping-price')
        shipping_tax            = num('shipping-tax')
        gift_wrap_price         = num('gift-wrap-price')
        gift_wrap_tax           = num('gift-wrap-tax')
        item_promo_discount     = num('item-promotion-discount')
        ship_promo_discount     = num('ship-promotion-discount')

        # Compute Total Sales per line:
        # Sum of positive components minus discounts (discounts are positive numbers to subtract)
        df['Total Sales'] = (
            item_price + item_tax + shipping_price + shipping_tax + gift_wrap_price + gift_wrap_tax
            - item_promo_discount - ship_promo_discount
        )

        return df[['Date', 'SKU', 'Total Sales', 'Total Units', 'ASIN']]

    except gspread.exceptions.WorksheetNotFound:
        st.warning(f"'{sheet_name}' worksheet not found. Orders-based sales data will not be available.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to load orders data from '{sheet_name}': {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def load_data(_spreadsheet, sheet_name, date_col, date_format):
    """Loads and cleans data from a specified worksheet."""
    try:
        worksheet = _spreadsheet.worksheet(sheet_name)
        df = pd.DataFrame(worksheet.get_all_records())
        
        if 'Advertised ASIN' in df.columns:
            df['Advertised ASIN'] = df['Advertised ASIN'].astype(str)
        if 'Targeting' in df.columns:
            df['Targeting'] = df['Targeting'].astype(str)

        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors='coerce')
            df.dropna(subset=[date_col], inplace=True)

        sales_col_name = next((col for col in df.columns if '14 day total sales' in col.lower()), None)
        
        numeric_cols = ['Spend', '14 Day Total Orders (#)', 'Clicks', 'Impressions']
        if sales_col_name:
            numeric_cols.append(sales_col_name)

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(r'[$,₹]', '', regex=True),
                    errors='coerce'
                ).fillna(0)
        
        if sales_col_name:
            df.rename(columns={sales_col_name: 'Sales'}, inplace=True)
            
        return df
    except gspread.exceptions.WorksheetNotFound:
        st.error(f"Error: The '{sheet_name}' worksheet was not found. Please ensure it exists in your Google Sheet.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to load data from '{sheet_name}': {e}")
        return pd.DataFrame()

def log_change_to_sheet(_spreadsheet, asin, campaign, comment):
    """Logs a change to the ASIN Selections sheet in the correct column order."""
    try:
        worksheet = _spreadsheet.worksheet("ASIN Selections (log changes)")
        log_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        expiry_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        row_to_append = [log_date, expiry_date, asin, campaign, comment, ""]
        worksheet.append_row(row_to_append)
        st.toast("Change logged successfully!", icon="✅")
    except Exception as e:
        st.error(f"Failed to log change: {e}")

@st.cache_data(ttl=60)
def get_active_logged_pairs(_spreadsheet):
    """Gets a set of (ASIN, Campaign) tuples that have been logged recently."""
    try:
        worksheet = _spreadsheet.worksheet("ASIN Selections (log changes)")
        log_df = pd.DataFrame(worksheet.get_all_records())
        if not log_df.empty and 'Timestamp' in log_df.columns and 'ASIN' in log_df.columns and 'Campaign' in log_df.columns:
            log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'], format='mixed', dayfirst=False)
            seven_days_ago = datetime.now() - timedelta(days=7)
            recent_logs = log_df[log_df['Timestamp'] >= seven_days_ago]
            return set(zip(recent_logs['ASIN'].astype(str), recent_logs['Campaign'].astype(str)))
    except Exception as e:
        st.warning(f"Could not load change logs: {e}")
    return set()

def style_pivot_table(row, active_pairs):
    """Applies row highlighting for logged pairs and cell coloring for spend/orders."""
    row_asin = str(row.name[0])
    row_campaign = str(row.name[1])
    styles = pd.Series('', index=row.index)
    
    if (row_asin, row_campaign) in active_pairs:
        styles[:] = 'background-color: #ffcdd2'
    
    for date_col in row.index.get_level_values(0).unique():
        spend_col = (date_col, 'Spend')
        orders_col = (date_col, 'Orders')
        
        if spend_col in row.index and orders_col in row.index:
            spend = row[spend_col]
            orders = row[orders_col]
            
            if spend > 0 and orders == 0:
                styles[spend_col] = 'background-color: #720e9e; color: white;'
            elif orders > 0:
                ratio = spend / (orders + 1e-9)
                if ratio > 15: styles[spend_col] = 'background-color: #ff0040; color: white;'
                elif ratio > 7.5: styles[spend_col] = 'background-color: #00BFFF; color: white;'
                else: styles[spend_col] = 'background-color: #1DB954; color: white;'
    return styles

# --- Dialog Windows ---

@st.dialog("Log a Change")
def log_change_dialog(spreadsheet, asin, campaign):
    st.write(f"**ASIN:** {asin}")
    st.write(f"**Campaign:** {campaign}")
    comment = st.text_area("Enter comment or change description:")
    if st.button("Submit Log") and comment:
        log_change_to_sheet(spreadsheet, asin, campaign, comment)
        st.rerun()

@st.dialog("Recent Change Logs (Last 7 Days)")
def show_change_logs_dialog(spreadsheet):
    # --- MODIFIED: CSS to make the dialog wider ---
    st.markdown("""
    <style>
        .stModal > div {
            max-width: 80vw !important;
        }
    </style>
    """, unsafe_allow_html=True)
    try:
        log_ws = spreadsheet.worksheet("ASIN Selections (log changes)")
        log_df = pd.DataFrame(log_ws.get_all_records())
        if not log_df.empty:
            log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'], format='mixed', dayfirst=False)
            seven_days_ago = datetime.now() - timedelta(days=7)
            recent_logs = log_df[log_df['Timestamp'] >= seven_days_ago].sort_values(by="Timestamp", ascending=False)
            st.dataframe(recent_logs[['Timestamp', 'ASIN', 'Campaign', 'Comment']], hide_index=True, use_container_width=True)
        else:
            st.info("No logs found.")
    except Exception as e:
        st.error(f"Could not load change logs: {e}")

@st.dialog("Upload Reports")
def upload_dialog(spreadsheet, country_name, country_code):
    """Dialog to handle file uploads and append data to Google Sheets."""
    st.info(f"You are uploading reports for: **{country_name}**")
    
    # Main Ad Reports
    uploaded_adv = st.file_uploader("Upload Advertised Product Report (.xlsx)", type="xlsx", key="adv_uploader")
    uploaded_tgt = st.file_uploader("Upload Targeting Report (.xlsx)", type="xlsx", key="tgt_uploader")

    # --- NEW: Conditional uploader for India Sales Report ---
    uploaded_sales = None
    if country_name == "India":
        st.markdown("---")
        uploaded_sales = st.file_uploader("Upload Amazon Fulfilled Sales Report (for Organic Sales)", type=['xlsx', 'csv'])

    if st.button("Process and Append Data"):
        if uploaded_adv and uploaded_tgt:
            try:
                with st.spinner("Processing files and updating sheets..."):
                    # Process Ad Reports (existing logic)
                    df_adv = pd.read_excel(uploaded_adv)
                    df_tgt = pd.read_excel(uploaded_tgt)
                    for df in [df_adv, df_tgt]:
                        df.replace([np.nan, np.inf, -np.inf], None, inplace=True)
                        for col in df.columns:
                            if pd.api.types.is_datetime64_any_dtype(df[col]):
                                df[col] = df[col].dt.strftime('%b %d, %Y')
                    ws_adv = spreadsheet.worksheet(f"Advertised_product_{country_code}")
                    ws_tgt = spreadsheet.worksheet(f"Targeting_{country_code}")
                    ws_adv.append_rows(df_adv.values.tolist(), value_input_option='USER_ENTERED')
                    ws_tgt.append_rows(df_tgt.values.tolist(), value_input_option='USER_ENTERED')
                    
                    # --- NEW: Process Sales Report if uploaded for India ---
                    if uploaded_sales and country_name == "India":
                        if uploaded_sales.name.endswith('.csv'):
                            df_sales = pd.read_csv(uploaded_sales)
                        else:
                            df_sales = pd.read_excel(uploaded_sales)
                        
                        # Append data to the sales sheet (no de-duplication needed for sales)
                        ws_sales = spreadsheet.worksheet("Fulfilled_Sales_IND")
                        ws_sales.append_rows(df_sales.values.tolist(), value_input_option='USER_ENTERED')

                st.success("✅ Data appended successfully! The dashboard will now refresh.")
                st.cache_data.clear()
                st.rerun()

            except Exception as e:
                st.error(f"An error occurred during the upload process: {e}")
        else:
            st.warning("⚠️ Please upload at least the two advertising reports before processing.")

# --- Views / Pages ---

def render_main_dashboard(adv_df, spreadsheet, currency_symbol):
    # --- Filters (ASIN, Campaign, Date) ---
    st.markdown("##### Filters")
    col1, col2, col3 = st.columns([2, 2, 3])

    with col1:
        asin_options = [''] + sorted(list(adv_df['Advertised ASIN'].unique()))
        selected_asins = st.multiselect("Filter by ASIN", options=asin_options)

    with col2:
        campaign_options = [''] + sorted(list(adv_df['Campaign Name'].unique()))
        selected_campaigns = st.multiselect("Filter by Campaign", options=campaign_options)

    with col3:
        selected_date_range = st.date_input(
            "Select Date Range",
            value=(st.session_state.start_date, st.session_state.end_date),
            key="main_date_range"
        )
        if len(selected_date_range) == 2:
            st.session_state.start_date, st.session_state.end_date = pd.to_datetime(selected_date_range)

    # Start from the (already date-filtered at top level) adv_df and then apply ASIN/Campaign filters
    filtered_df = adv_df.copy()
    if selected_asins:
        filtered_df = filtered_df[filtered_df['Advertised ASIN'].isin(selected_asins)]
    if selected_campaigns:
        filtered_df = filtered_df[filtered_df['Campaign Name'].isin(selected_campaigns)]

    # --- KPIs based on the *filtered* data ---
    total_spend = filtered_df['Spend'].sum() if not filtered_df.empty else 0.0
    total_orders = filtered_df['Orders'].sum() if not filtered_df.empty else 0.0

    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.metric("Total Spend (Filtered Selection)", f"{currency_symbol}{total_spend:,.2f}")
    with mcol2:
        st.metric("Total Orders (Filtered Selection)", f"{int(total_orders):,}")
    st.markdown("---")

    # --- Pivot / table view (unchanged except uses filtered_df) ---
    pivot_df = pd.DataFrame()
    if not filtered_df.empty and 'Sales' in filtered_df.columns:
        acos_summary = filtered_df.groupby(['Advertised ASIN', 'Campaign Name']).agg(
            TotalSpend=('Spend', 'sum'),
            TotalSales=('Sales', 'sum')
        ).reset_index()
        acos_summary['ACOS'] = 100 * acos_summary['TotalSpend'] / (acos_summary['TotalSales'] + 1e-9)
        sorted_by_acos = acos_summary.sort_values(by='ACOS', ascending=False)

        pivot_df = filtered_df.pivot_table(
            index=['Advertised ASIN', 'Campaign Name'],
            columns='Date',
            values=['Spend', 'Orders'],
            aggfunc='sum'
        ).fillna(0)

        sorted_index = pd.MultiIndex.from_frame(sorted_by_acos[['Advertised ASIN', 'Campaign Name']])
        pivot_df = pivot_df.reindex(sorted_index).dropna(how='all')
        pivot_df = pivot_df.reorder_levels([1, 0], axis=1).sort_index(axis=1, ascending=False)

    if not pivot_df.empty:
        st.info("To see details or log a change, click on a row in the table below.")

        legend_html = """
        <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 15px; margin-bottom: 10px; font-size: 14px;">
            <strong>Legend:</strong>
            <div style="display: flex; align-items: center;">
                <span style="height: 15px; width: 15px; background-color: #720e9e; border-radius: 3px; margin-right: 5px;"></span>
                <span style="color: black;">Spend > 0 and Orders = 0</span>
            </div>
            <div style="display: flex; align-items: center;">
                <span style="height: 15px; width: 15px; background-color: #ff0040; border-radius: 3px; margin-right: 5px;"></span>
                <span style="color: black;">Spends/Orders > 15</span>
            </div>
            <div style="display: flex; align-items: center;">
                <span style="height: 15px; width: 15px; background-color: #00BFFF; border-radius: 3px; margin-right: 5px;"></span>
                <span style="color: black;">7.5 < Spends/Orders < 15</span>
            </div>
            <div style="display: flex; align-items: center;">
                <span style="height: 15px; width: 15px; background-color: #1DB954; border-radius: 3px; margin-right: 5px;"></span>
                <span style="color: black;">Spends/Orders < 7.5</span>
            </div>
            <div style="display: flex; align-items: center;">
                <span style="height: 15px; width: 30px; background-color: #ffcdd2; border-radius: 3px; margin-right: 5px;"></span>
                <span style="color: black;">Row Logged</span>
            </div>
        </div>
        """
        st.markdown(legend_html, unsafe_allow_html=True)

        active_pairs = get_active_logged_pairs(spreadsheet)
        styled_pivot = pivot_df.style.apply(style_pivot_table, active_pairs=active_pairs, axis=1).format("{:,.2f}")
        st.dataframe(
            styled_pivot,
            on_select="rerun",
            selection_mode="single-row",
            key="main_table_selection",
            use_container_width=True
        )
        return pivot_df
    else:
        st.info("No data matches the current filters for the selected date range.")
        return pd.DataFrame()


def render_asin_details(adv_df, spreadsheet):
    asin = st.session_state.selected_asin
    st.header(f"ASIN Details: {asin}")

    col1, col2 = st.columns([1, 3], vertical_alignment="center")
    with col1:
        if st.button("← Back to Main Dashboard"):
            st.session_state.selected_asin = None
            st.rerun()
    with col2:
        selected_date_range = st.date_input(
            "Select Date Range",
            value=(st.session_state.start_date, st.session_state.end_date),
            key="asin_date_range",
            label_visibility="collapsed"
        )
        if len(selected_date_range) == 2:
            st.session_state.start_date, st.session_state.end_date = pd.to_datetime(selected_date_range)
    
    # --- Initial Data Filtering by ASIN and Date ---
    asin_df_all = adv_df[adv_df['Advertised ASIN'] == asin].copy()
    mask = (asin_df_all['Date'] >= st.session_state.start_date) & (asin_df_all['Date'] <= st.session_state.end_date)
    asin_df = asin_df_all.loc[mask].copy()

    # --- NEW: Organic vs. Ad Sales Insights (FOR INDIA ONLY) ---
    if st.session_state.country == "India":
        st.markdown("---")
        st.subheader("Sales Insights (Ad vs. Organic)")

        # Load the total sales data from Orders report
        sales_df = load_sales_data(spreadsheet, "Orders_IND")
        
        if not sales_df.empty:
            # Match by ASIN only
            asin_value = asin  # Already the selected ASIN
            sales_mask = (sales_df['Date'] >= st.session_state.start_date) & (sales_df['Date'] <= st.session_state.end_date)
            sku_sales_df = sales_df[(sales_df['ASIN'] == asin_value) & sales_mask]

            if asin_value:
                # 1. Calculate Ad Sales & Units
                total_ad_sales = asin_df['Sales'].sum()
                total_ad_units = asin_df['14 Day Total Orders (#)'].sum()

                # 2. Calculate Total Sales & Units
                total_sales = sku_sales_df['Total Sales'].sum()
                total_units = sku_sales_df['Total Units'].sum()
                
                # 3. Calculate Organic Sales & Ad Dependency
                organic_sales = total_sales - total_ad_sales
                ad_dependency = (total_ad_units / total_units * 100) if total_units > 0 else 0
                
                # Display Metrics
                mcol1, mcol2, mcol3 = st.columns(3)
                mcol1.metric("Total Ad Sales", f"₹{total_ad_sales:,.2f}")
                mcol2.metric("Total Organic Sales", f"₹{organic_sales:,.2f}")
                mcol3.metric("Ad Dependency", f"{ad_dependency:.1f}%")

            else:
                st.info("ASIN not found in the advertising report, cannot calculate organic sales.")
        else:
            st.info("Orders report not found or empty. Upload the report to see these insights.")

    # --- Campaign Filter and Existing Logic ---
    st.markdown("---")
    campaign_options = sorted(list(asin_df['Campaign Name'].unique()))
    selected_campaigns = st.multiselect("Filter by Campaign", options=campaign_options)

    if selected_campaigns:
        asin_df = asin_df[asin_df['Campaign Name'].isin(selected_campaigns)]

    if asin_df.empty:
        st.info("No advertising data available for the current filter selection.")
        st.stop()

    asin_df.rename(columns={'14 Day Total Orders (#)': 'Orders'}, inplace=True)
    agg_map = {'Clicks': 'sum', 'Impressions': 'sum', 'Spend': 'sum', 'Sales': 'sum', 'Orders': 'sum'}

    def recalculate_metrics(df):
        df['ACOS'] = np.where(df['Sales'] > 0, (df['Spend'] / df['Sales']) * 100, 0)
        df['ROAS'] = np.where(df['Spend'] > 0, df['Sales'] / df['Spend'], 0)
        df['CTR %'] = np.where(df['Impressions'] > 0, (df['Clicks'] / df['Impressions']) * 100, 0)
        df['CVR %'] = np.where(df['Clicks'] > 0, (df['Orders'] / df['Clicks']) * 100, 0)
        return df

    # ... The rest of the function (tabs, charts, etc.) remains unchanged ...
    tab_daily, tab_weekly, tab_monthly = st.tabs(["Daily", "Weekly", "Monthly"])
    with tab_daily:
        st.subheader("Daily Performance by Campaign")
        daily_df = asin_df.groupby(['Date', 'Campaign Name']).agg(agg_map).reset_index()
        daily_df = recalculate_metrics(daily_df).sort_values(by='Date', ascending=False)
        st.dataframe(daily_df, column_config={"Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD")}, hide_index=True, on_select="rerun", selection_mode="single-row", key="daily_asin_select")
        if st.button("Generate Daily Chart", key="chart_daily"):
            chart_df = daily_df.groupby('Date').agg(agg_map).sort_index()
            chart_df = recalculate_metrics(chart_df)
            st.write("ACOS Trend (%)"); st.line_chart(chart_df, y=['ACOS'], color="#FF5733")
            st.write("Clicks Trend"); st.line_chart(chart_df, y=['Clicks'], color="#33A7FF")
            st.write("Impressions Trend"); st.line_chart(chart_df, y=['Impressions'], color="#28A745")
    with tab_weekly:
        st.subheader("Weekly Aggregated Performance by Campaign")
        weekly_df = asin_df.copy()
        weekly_df['Period'] = weekly_df['Date'].dt.to_period('W-SUN').apply(lambda r: r.start_time).dt.date
        weekly_agg_df = weekly_df.groupby(['Campaign Name', 'Period']).agg(agg_map).reset_index()
        weekly_agg_df = recalculate_metrics(weekly_agg_df).sort_values(by='Period', ascending=False)
        st.dataframe(weekly_agg_df, column_config={"Period": st.column_config.DateColumn("Week Of", format="D MMM YYYY")}, hide_index=True, on_select="rerun", selection_mode="single-row", key="weekly_asin_select")
        if st.button("Generate Weekly Chart", key="chart_weekly"):
            chart_df = weekly_agg_df.groupby('Period').agg(agg_map).sort_index()
            chart_df = recalculate_metrics(chart_df)
            st.write("ACOS Trend (%)"); st.line_chart(chart_df, y=['ACOS'], color="#FF5733")
            st.write("Clicks Trend"); st.line_chart(chart_df, y=['Clicks'], color="#33A7FF")
            st.write("Impressions Trend"); st.line_chart(chart_df, y=['Impressions'], color="#28A745")
    with tab_monthly:
        st.subheader("Monthly Aggregated Performance by Campaign")
        monthly_df = asin_df.groupby(['Campaign Name', pd.Grouper(key='Date', freq='MS')]).agg(agg_map).reset_index()
        monthly_df = recalculate_metrics(monthly_df).sort_values(by='Date', ascending=False)
        monthly_df.rename(columns={'Date': 'Period'}, inplace=True)
        st.dataframe(monthly_df, column_config={"Period": st.column_config.DateColumn("Month", format="MMMM YYYY")}, hide_index=True, on_select="rerun", selection_mode="single-row", key="monthly_asin_select")
        if st.button("Generate Monthly Chart", key="chart_monthly"):
            chart_df = monthly_df.groupby('Period').agg(agg_map).sort_index()
            chart_df = recalculate_metrics(chart_df)
            st.write("ACOS Trend (%)"); st.line_chart(chart_df, y=['ACOS'], color="#FF5733")
            st.write("Clicks Trend"); st.line_chart(chart_df, y=['Clicks'], color="#33A7FF")
            st.write("Impressions Trend"); st.line_chart(chart_df, y=['Impressions'], color="#28A745")
    selected_campaign = None; selected_df = None; selection_state = None
    if "daily_asin_select" in st.session_state and st.session_state.daily_asin_select.selection.rows: selected_df, selection_state = daily_df, st.session_state.daily_asin_select
    elif "weekly_asin_select" in st.session_state and st.session_state.weekly_asin_select.selection.rows: selected_df, selection_state = weekly_agg_df, st.session_state.weekly_asin_select
    elif "monthly_asin_select" in st.session_state and st.session_state.monthly_asin_select.selection.rows: selected_df, selection_state = monthly_df, st.session_state.monthly_asin_select
    if selected_df is not None and selection_state is not None:
        selected_row_index = selection_state.selection.rows[0]
        if selected_row_index < len(selected_df): selected_campaign = selected_df.iloc[selected_row_index]['Campaign Name']
    if selected_campaign:
        st.markdown("---"); st.write(f"**You selected:** Campaign `{selected_campaign}`")
        if st.button(f"View Details for Campaign: {selected_campaign}", key="asin_to_campaign_nav_button"): st.session_state.selected_campaign, st.session_state.selected_asin = selected_campaign, None; st.rerun()
    st.markdown("---"); st.subheader("Log a Change for this ASIN")
    with st.form("log_form"):
        comment = st.text_input("Enter comment or change description:")
        campaign_to_log = st.selectbox("Select associated campaign (optional)", options=asin_df_all['Campaign Name'].unique())
        if st.form_submit_button("Log Change") and comment: log_change_to_sheet(spreadsheet, asin, campaign_to_log, comment)

def render_campaign_details(targeting_df, spreadsheet):
    campaign = st.session_state.selected_campaign
    st.header(f"Campaign Details: {campaign}")
    
    # --- Date Filter Row ---
    col1, col2 = st.columns([1, 3], vertical_alignment="center")
    with col1:
        if st.button("← Back to Main Dashboard"): st.session_state.selected_campaign = None; st.rerun()
    with col2:
        selected_date_range = st.date_input("Select Date Range", value=(st.session_state.start_date, st.session_state.end_date), key="campaign_date_range", label_visibility="collapsed")
        if len(selected_date_range) == 2: st.session_state.start_date, st.session_state.end_date = pd.to_datetime(selected_date_range)
    
    # --- Initial Data Filtering by Campaign and Date ---
    campaign_df_all = targeting_df[targeting_df['Campaign Name'] == campaign].copy()
    mask = (campaign_df_all['Date'] >= st.session_state.start_date) & (campaign_df_all['Date'] <= st.session_state.end_date)
    campaign_df = campaign_df_all.loc[mask].copy()

    
        
    campaign_df.rename(columns={'14 Day Total Orders (#)': 'Orders'}, inplace=True)
    agg_map = {'Clicks': 'sum', 'Impressions': 'sum', 'Spend': 'sum', 'Sales': 'sum', 'Orders': 'sum'}
    
    def recalculate_metrics(df):
        df['ACOS'] = np.where(df['Sales'] > 0, (df['Spend'] / df['Sales']) * 100, 0)
        df['ROAS'] = np.where(df['Spend'] > 0, df['Sales'] / df['Spend'], 0)
        df['CTR %'] = np.where(df['Impressions'] > 0, (df['Clicks'] / df['Impressions']) * 100, 0)
        df['CVR %'] = np.where(df['Clicks'] > 0, (df['Orders'] / df['Clicks']) * 100, 0)
        return df
        
    st.subheader("Overall Campaign Performance")
    chart_period = st.radio("Select Chart Period", ["Daily", "Weekly", "Monthly"], horizontal=True, key="campaign_chart_period")
    campaign_agg_df = pd.DataFrame()
    if chart_period == "Daily": campaign_agg_df = campaign_df.groupby('Date').agg(agg_map)
    elif chart_period == "Weekly": campaign_agg_df = campaign_df.resample('W-SUN', on='Date').agg(agg_map)
    elif chart_period == "Monthly": campaign_agg_df = campaign_df.resample('MS', on='Date').agg(agg_map)
    if not campaign_agg_df.empty:
        campaign_agg_df = recalculate_metrics(campaign_agg_df)
        st.write("ACOS Trend (%)"); st.line_chart(campaign_agg_df, y=['ACOS'], color="#FF5733")
        st.write("Clicks Trend"); st.line_chart(campaign_agg_df, y=['Clicks'], color="#33A7FF")
        st.write("Impressions Trend"); st.line_chart(campaign_agg_df, y=['Impressions'], color="#28A745")
        
    st.markdown("---"); st.subheader("Performance by Keyword")
    grouping_cols = ['Targeting', 'Match Type']
    existing_grouping_cols = [col for col in grouping_cols if col in campaign_df.columns]

    # --- NEW: Targeting and Match Type Filters ---
    st.markdown("---")
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        target_options = sorted(list(campaign_df['Targeting'].unique()))
        selected_targets = st.multiselect("Filter by Targeting", options=target_options)
    with f_col2:
        match_type_options = sorted(list(campaign_df['Match Type'].unique()))
        selected_match_types = st.multiselect("Filter by Match Type", options=match_type_options)

    if selected_targets:
        campaign_df = campaign_df[campaign_df['Targeting'].isin(selected_targets)]
    if selected_match_types:
        campaign_df = campaign_df[campaign_df['Match Type'].isin(selected_match_types)]
    # --- END NEW FILTERS ---

    if campaign_df.empty: 
        st.info("No data available for the current filter selection.")
        st.stop()

    tab_daily, tab_weekly, tab_monthly = st.tabs(["Daily", "Weekly", "Monthly"])
    with tab_daily:
        daily_df = campaign_df.groupby(['Date'] + existing_grouping_cols).agg(agg_map).reset_index()
        daily_df = recalculate_metrics(daily_df).sort_values(by='Date', ascending=False)
        st.dataframe(daily_df, column_config={"Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD")}, hide_index=True)
        st.markdown("---"); st.subheader("Chart for a specific Keyword (Daily)")
        keywords = sorted(list(daily_df['Targeting'].unique())) if 'Targeting' in daily_df.columns else []
        selected_keyword = st.selectbox("Select a keyword", options=keywords, key="daily_keyword_select")
        if selected_keyword:
            keyword_df = daily_df[daily_df['Targeting'] == selected_keyword].set_index('Date')
            st.write("ACOS Trend (%)"); st.line_chart(keyword_df, y=['ACOS'], color="#FF5733")
            st.write("Clicks Trend"); st.line_chart(keyword_df, y=['Clicks'], color="#33A7FF")
            st.write("Impressions Trend"); st.line_chart(keyword_df, y=['Impressions'], color="#28A745")
    with tab_weekly:
        weekly_df = campaign_df.copy()
        weekly_df['Period'] = weekly_df['Date'].dt.to_period('W-SUN').apply(lambda r: r.start_time).dt.date
        weekly_agg_df = weekly_df.groupby(['Period'] + existing_grouping_cols).agg(agg_map).reset_index()
        weekly_agg_df = recalculate_metrics(weekly_agg_df).sort_values(by='Period', ascending=False)
        st.dataframe(weekly_agg_df, column_config={"Period": st.column_config.DateColumn("Week Of", format="D MMM YYYY")}, hide_index=True)
        st.markdown("---"); st.subheader("Chart for a specific Keyword (Weekly)")
        keywords = sorted(list(weekly_agg_df['Targeting'].unique())) if 'Targeting' in weekly_agg_df.columns else []
        selected_keyword = st.selectbox("Select a keyword", options=keywords, key="weekly_keyword_select")
        if selected_keyword:
            keyword_df = weekly_agg_df[weekly_agg_df['Targeting'] == selected_keyword].set_index('Period')
            st.write("ACOS Trend (%)"); st.line_chart(keyword_df, y=['ACOS'], color="#FF5733")
            st.write("Clicks Trend"); st.line_chart(keyword_df, y=['Clicks'], color="#33A7FF")
            st.write("Impressions Trend"); st.line_chart(keyword_df, y=['Impressions'], color="#28A745")
    with tab_monthly:
        monthly_df = campaign_df.groupby([pd.Grouper(key='Date', freq='MS')] + existing_grouping_cols).agg(agg_map).reset_index()
        monthly_df = recalculate_metrics(monthly_df).sort_values(by='Date', ascending=False)
        monthly_df.rename(columns={'Date': 'Period'}, inplace=True)
        st.dataframe(monthly_df, column_config={"Period": st.column_config.DateColumn("Month", format="MMMM YYYY")}, hide_index=True)
        st.markdown("---"); st.subheader("Chart for a specific Keyword (Monthly)")
        keywords = sorted(list(monthly_df['Targeting'].unique())) if 'Targeting' in monthly_df.columns else []
        selected_keyword = st.selectbox("Select a keyword", options=keywords, key="monthly_keyword_select")
        if selected_keyword:
            keyword_df = monthly_df[monthly_df['Targeting'] == selected_keyword].set_index('Period')
            st.write("ACOS Trend (%)"); st.line_chart(keyword_df, y=['ACOS'], color="#FF5733")
            st.write("Clicks Trend"); st.line_chart(keyword_df, y=['Clicks'], color="#33A7FF")
            st.write("Impressions Trend"); st.line_chart(keyword_df, y=['Impressions'], color="#28A745")

# --- Main Application Flow ---

def on_country_change():
    """Reset page navigation to avoid state conflicts when changing country."""
    st.session_state.selected_asin = None
    st.session_state.selected_campaign = None
    for key in ['main_table_selection', 'daily_asin_select', 'weekly_asin_select', 'monthly_asin_select']:
        if key in st.session_state: st.session_state[key]['selection']['rows'] = []

title_col, selector_col = st.columns([3, 1])
with title_col:
    st.title("ASIN Performance Dashboard")
with selector_col:
    st.selectbox("Select Country", ("India", "USA"), key="country", on_change=on_country_change)

COUNTRY_SETTINGS = {"India": {"code": "IND", "currency": "₹"}, "USA": {"code": "USA", "currency": "$"}}
current_country = st.session_state.country
country_code = COUNTRY_SETTINGS[current_country]["code"]
currency_symbol = COUNTRY_SETTINGS[current_country]["currency"]

spreadsheet = connect_to_gsheet()
adv_df_raw = load_data(spreadsheet, f"Advertised_product_{country_code}", date_col='Date', date_format='%b %d, %Y')
targeting_df_raw = load_data(spreadsheet, f"Targeting_{country_code}", date_col='Date', date_format='%b %d, %Y')

# Filter data by date range once, at the top level
date_mask = (adv_df_raw['Date'] >= st.session_state.start_date) & (adv_df_raw['Date'] <= st.session_state.end_date)
adv_df = adv_df_raw.loc[date_mask].copy()
adv_df['Date'] = adv_df['Date'].dt.date
adv_df.rename(columns={'14 Day Total Orders (#)': 'Orders'}, inplace=True)

# --- Router to display the correct page ---
if st.session_state.selected_asin:
    render_asin_details(adv_df_raw, spreadsheet)
elif st.session_state.selected_campaign:
    render_campaign_details(targeting_df_raw, spreadsheet)
else:
    log_col, upload_col, _ = st.columns([1.5, 1.5, 5])
    with log_col:
        if st.button("View Recent Change Logs"): show_change_logs_dialog(spreadsheet)
    with upload_col:
        if st.button("Upload Reports"): upload_dialog(spreadsheet, current_country, country_code)
    displayed_df = render_main_dashboard(adv_df, spreadsheet, currency_symbol)
    if "main_table_selection" in st.session_state and st.session_state.main_table_selection.selection.rows:
        selected_row_index = st.session_state.main_table_selection.selection.rows[0]
        if not displayed_df.empty and selected_row_index < len(displayed_df):
            selected_asin, selected_campaign = displayed_df.index[selected_row_index]
            st.markdown("---"); st.write(f"**You selected:** ASIN `{selected_asin}` in Campaign `{selected_campaign}`")
            b_col1, b_col2, b_col3 = st.columns(3)
            with b_col1:
                if st.button(f"View Details for ASIN: {selected_asin}"): st.session_state.selected_asin = selected_asin; st.rerun()
            with b_col2:
                if st.button(f"View Details for Campaign: {selected_campaign}"): st.session_state.selected_campaign = selected_campaign; st.rerun()
            with b_col3:
                if st.button("Log a Change"): log_change_dialog(spreadsheet, selected_asin, selected_campaign)
