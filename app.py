import streamlit as st
from collections.abc import Mapping

# Hide the top "app" header in the sidebar nav
st.markdown("""
<style>
/* In the sidebar, hide the first nav item, which is the main script label */
div[data-testid="stSidebarNav"] > div:first-child { 
    display: none; 
}
</style>
""", unsafe_allow_html=True)
# App-wide config
st.set_page_config(page_title="ASIN Performance Dashboard", layout="wide")

# --- Helpers ---
def check_credentials(username: str, password: str) -> bool:

    try:
        # Try nested table first: [login.users]
        users = st.secrets["login"]["users"]
    except Exception:
        try:
            # Fallback if someone used a flat key like "login.users"
            users = st.secrets["login.users"]
        except Exception:
            return False

    # Coerce to a plain dict (Streamlit returns a mapping-like object)
    if isinstance(users, Mapping):
        users = dict(users)
    else:
        # Last-ditch: try to parse if it's a string (shouldn't happen,
        # but avoids false negatives)
        try:
            import json
            users = json.loads(str(users))
            if not isinstance(users, dict):
                return False
        except Exception:
            return False

    username = username.strip()
    password = password.strip()

    stored_pw = users.get(username)
    return stored_pw is not None and str(stored_pw) == str(password)

def render_login():
    # Optional: hide sidebar/nav while on login
    st.markdown(
        """
        <style>
            /* Hide the page selector sidebar during login for a cleaner look */
            section[data-testid="stSidebar"] { display: none; }
            div[data-testid="stToolbar"] { display: none; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🔐 Please sign in")

    with st.form("login_form", clear_on_submit=False):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

    if submit:
        if check_credentials(u.strip(), p):
            st.session_state.logged_in = True
            st.session_state.username = u.strip()
            st.success("Login successful. Redirecting…")
            st.switch_page("pages/1_Dashboard.py")
        else:
            st.error("Invalid username or password")

# --- Main router ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# If already logged in, jump straight to Dashboard
if st.session_state.logged_in:
    st.switch_page("pages/1_Dashboard.py")
else:
    render_login()
