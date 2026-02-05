"""
================================================================================
CSS STYLING FOR E91 QKD GUI
================================================================================

Professional dark theme styling for the Streamlit interface.

Author: Tyler Barr
Version: 7.0.0 Modular
Date: 2025

================================================================================
"""

# ============================================================================
# PROFESSIONAL CSS STYLING
# ============================================================================

PROFESSIONAL_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    body, .main, .stApp {
        font-family: 'Inter', sans-serif;
    }
    .main { background: linear-gradient(135deg, #0a0e1a 0%, #1a1f2e 100%); }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; background: #1a1f2e; border-radius: 12px; padding: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px; padding: 12px 24px; font-weight: 600; color: #94a3b8;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #4a9eff 0%, #7c3aed 100%); color: #fff;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        white-space: nowrap;
    }

    /* Inputs */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 6px; border: 2px solid #2d3548; background: #1a1f2e; color: #fff;
    }
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #4a9eff; box-shadow: 0 0 0 3px rgba(74, 158, 255, 0.1);
    }

    /* Selectboxes */
    .stSelectbox > div > div {
        border-radius: 6px;
        border: 2px solid #2d3548;
        background: #1a1f2e;
        color: #fff;
    }

    /* Radio buttons */
    .stRadio > div {
        flex-direction: row;
        gap: 16px;
    }

    /* Tooltips */
    [data-testid="stTooltipIcon"] {
        color: #6b7280;
        font-size: 16px;
        margin-left: 6px;
    }
    [data-testid="stTooltipIcon"]:hover { color: #4a9eff; }

    /* Alerts */
    .stSuccess { background: rgba(16, 185, 129, 0.1); border-left: 4px solid #10b981; border-radius: 8px; }
    .stError { background: rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444; border-radius: 8px; }
    .stWarning { background: rgba(251, 191, 36, 0.1); border-left: 4px solid #fbbf24; border-radius: 8px; }
    .stInfo { background: rgba(74, 158, 255, 0.1); border-left: 4px solid #4a9eff; border-radius: 8px; }

    /* Expanders */
    .streamlit-expanderHeader {
        border-radius: 8px;
        background: #1a1f2e;
        border: 2px solid #2d3548;
    }

    /* Fix dropdown arrow icons */
    details summary::-webkit-details-marker {
        display: none;
    }

    /* Responsive text */
    h1, h2, h3 {
        word-wrap: break-word;
    }
</style>
"""

# Minimal safe CSS (no dropdown hacks) used by the app.
BASE_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif !important; }
    .main { background: linear-gradient(135deg, #0a0e1a 0%, #1a1f2e 100%); }
</style>
"""


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'PROFESSIONAL_CSS',
    'BASE_CSS',
]
