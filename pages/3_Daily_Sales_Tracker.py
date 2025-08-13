# pages/2_Daily_Sales_Tracker.py

import streamlit as st
import pandas as pd
import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from datetime import date, timedelta, datetime
import calendar
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
st.set_page_config(page_title="Daily Sales Tracker", layout="wide")
st.title("Daily Sales Tracker")
st.markdown("---")

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

def get_or_create_worksheet(spreadsheet, sheet_name):
    """Gets a worksheet by name, creating it if it doesn't exist."""
    try:
        return spreadsheet.worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        return spreadsheet.add_worksheet(title=sheet_name, rows="100", cols="50")

def style_comparison_table(df, avg_col_name):
    """Applies color styling and number formatting to the final table."""
    date_cols = [col for col in df.columns if isinstance(col, date)]
    
    def style_row(row):
        avg = row[avg_col_name]
        styles = pd.Series('', index=row.index)
        for col in date_cols:
            val = row[col]
            if pd.isna(val) or val == 0:
                styles[col] = 'color: #999999' # Grey for no sales
            elif val > avg:
                styles[col] = 'background-color: #c8e6c9' # Green
            else: # Less than or equal to avg
                styles[col] = 'background-color: #ffcdd2' # Red
        return styles
    
    # Create format dictionary for all columns
    formatters = {avg_col_name: "{:.2f}"}
    for col in date_cols:
        formatters[col] = "{:.0f}" # Format daily units as integers

    return df.style.apply(style_row, axis=1).format(formatters)

# --- Main App Logic ---
spreadsheet = connect_to_gsheet()
today = date.today()

# Get or create the necessary worksheets
monthly_reports_ws = get_or_create_worksheet(spreadsheet, "Business_Reports_Monthly")
daily_reports_ws = get_or_create_worksheet(spreadsheet, "Business_Reports_Daily")
top_asins_avg_ws = get_or_create_worksheet(spreadsheet, "Top_ASINs_Monthly_Avg")

# --- Automatic Trigger for Monthly Upload ---
if today.day == 1:
    last_month_date = today.replace(day=1) - timedelta(days=1)
    last_month_str = last_month_date.strftime("%Y-%m")
    try:
        monthly_data = get_as_dataframe(monthly_reports_ws).dropna(how='all')
        if monthly_data.empty or last_month_str not in monthly_data['Month'].values:
            st.warning(f"Reminder: Please upload the full Business Report for {last_month_date.strftime('%B %Y')} to set this month's baseline.", icon="⚠️")
    except (KeyError, gspread.exceptions.GSpreadException):
         st.warning(f"Reminder: Please upload the full Business Report for {last_month_date.strftime('%B %Y')} to set this month's baseline.", icon="⚠️")


# --- Upload Section ---
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.header("Upload Monthly Report")
        month_options = pd.date_range(end=today, periods=12, freq='MS').strftime("%Y-%m-%B").tolist()
        selected_month_str = st.selectbox("Select Month for Upload", options=month_options)
        
        monthly_report_file = st.file_uploader("Upload Full Month Business Report", type=['csv'], key="monthly_uploader")

        if st.button("Submit Monthly Report"):
            if monthly_report_file:
                with st.spinner("Processing monthly data..."):
                    month_date = datetime.strptime(selected_month_str, "%Y-%m-%B")
                    month_key = month_date.strftime("%Y-%m")
                    _, days_in_month = calendar.monthrange(month_date.year, month_date.month)

                    # Read and tag the new monthly report
                    df_new_month = pd.read_csv(monthly_report_file)
                    df_new_month['Month'] = month_key
                    
                    # Append new data to the main monthly sheet, replacing if month exists
                    existing_monthly_df = get_as_dataframe(monthly_reports_ws).dropna(how='all')
                    if not existing_monthly_df.empty and 'Month' in existing_monthly_df.columns:
                        existing_monthly_df = existing_monthly_df[existing_monthly_df['Month'] != month_key]
                    updated_monthly_df = pd.concat([existing_monthly_df, df_new_month], ignore_index=True)
                    set_with_dataframe(monthly_reports_ws, updated_monthly_df)

                    # Calculate and save the monthly average for top 30 ASINs
                    df_new_month.rename(columns={'(Child) ASIN': 'ASIN'}, inplace=True)
                    top_30 = df_new_month.nlargest(30, 'Units Ordered').copy()
                    top_30['Avg Daily Units'] = top_30['Units Ordered'] / days_in_month
                    top_30['Month'] = month_key
                    
                    # Update the averages sheet
                    existing_avg_df = get_as_dataframe(top_asins_avg_ws).dropna(how='all')
                    if not existing_avg_df.empty and 'Month' in existing_avg_df.columns:
                        existing_avg_df = existing_avg_df[existing_avg_df['Month'] != month_key]
                    updated_avg_df = pd.concat([existing_avg_df, top_30[['Month', 'ASIN', 'Title', 'Avg Daily Units']]], ignore_index=True)
                    set_with_dataframe(top_asins_avg_ws, updated_avg_df)

                    st.success(f"Baseline for {month_date.strftime('%B %Y')} processed successfully!")
            else:
                st.error("Please upload a file.")

with col2:
    with st.container(border=True):
        st.header("Upload Daily Report")
        selected_date = st.date_input("Select Date for Upload", value=today - timedelta(days=1))
        
        daily_report_file = st.file_uploader("Upload Single Day Business Report", type=['csv'], key="daily_uploader")

        if st.button("Submit Daily Report"):
            if daily_report_file:
                with st.spinner("Processing daily data..."):
                    df_daily = pd.read_csv(daily_report_file)
                    df_daily['Date'] = selected_date.strftime("%Y-%m-%d")
                    
                    # Append daily data to the sheet, replacing if date exists
                    existing_daily_df = get_as_dataframe(daily_reports_ws).dropna(how='all')
                    if not existing_daily_df.empty and 'Date' in existing_daily_df.columns:
                         existing_daily_df = existing_daily_df[existing_daily_df['Date'] != selected_date.strftime("%Y-%m-%d")]
                    updated_daily_df = pd.concat([existing_daily_df, df_daily], ignore_index=True)
                    set_with_dataframe(daily_reports_ws, updated_daily_df)
                    st.success(f"Report for {selected_date.strftime('%d-%b-%Y')} saved!")
            else:
                st.error("Please upload a file.")


st.markdown("---")

# --- Display Section: Daily Order Tracking ---
st.header("Daily Order Tracking")

display_month_options = pd.date_range(end=today, periods=12, freq='MS').strftime("%B %Y").tolist()
display_month_str = st.selectbox("Select Month to View", options=display_month_options)

if display_month_str:
    with st.spinner("Generating comparison table..."):
        selected_month_dt = datetime.strptime(display_month_str, "%B %Y")
        prev_month_dt = selected_month_dt.replace(day=1) - timedelta(days=1)
        avg_month_str = prev_month_dt.strftime("%Y-%m")

        all_avg_df = get_as_dataframe(top_asins_avg_ws).dropna(how='all')
        if all_avg_df.empty or avg_month_str not in all_avg_df['Month'].values:
            st.warning(f"No average data found for {prev_month_dt.strftime('%B %Y')}. Please upload that month's full report first.")
        else:
            avg_df = all_avg_df[all_avg_df['Month'] == avg_month_str].copy()
            avg_df.rename(columns={'Avg Daily Units': 'Monthly Avg'}, inplace=True)
            
            all_daily_df = get_as_dataframe(daily_reports_ws).dropna(how='all')
            all_daily_df['Date'] = pd.to_datetime(all_daily_df['Date']).dt.date
            
            daily_for_month_df = all_daily_df[
                (pd.to_datetime(all_daily_df['Date']).dt.year == selected_month_dt.year) &
                (pd.to_datetime(all_daily_df['Date']).dt.month == selected_month_dt.month)
            ].copy()
            
            if daily_for_month_df.empty:
                st.info(f"No daily reports have been uploaded for {display_month_str} yet.")
            else:
                daily_for_month_df.rename(columns={'(Child) ASIN': 'ASIN'}, inplace=True)
                
                pivoted_daily_df = daily_for_month_df.pivot_table(
                    index='ASIN',
                    columns='Date',
                    values='Units Ordered',
                    aggfunc='sum'
                ).fillna(0)
                
                final_table = pd.merge(
                    avg_df[['ASIN', 'Title', 'Monthly Avg']],
                    pivoted_daily_df,
                    on='ASIN',
                    how='left'
                ).fillna(0)
                
                st.dataframe(style_comparison_table(final_table, 'Monthly Avg'), use_container_width=True)