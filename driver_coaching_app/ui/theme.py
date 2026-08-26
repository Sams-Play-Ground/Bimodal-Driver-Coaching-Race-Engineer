"""Central color palette + QSS stylesheet for the whole app.

The wireframe uses plain white cards with black outlines and three accent
colors (green = go / enter / save, orange = process / in-progress,
red = not-ready / coming soon). We mirror that directly instead of
inventing a different visual language.
"""

COLORS = {
    "bg": "#FFFFFF",
    "surface": "#FFFFFF",
    "border": "#1A1A1A",
    "text": "#1A1A1A",
    "text_muted": "#5B5B5B",
    "green": "#2FA84F",
    "green_bg": "#EAF7EE",
    "orange": "#F2994A",
    "orange_bg": "#FDF1E4",
    "red": "#E85D4C",
    "red_bg": "#FDECEA",
    "sidebar_bg": "#F7F7F8",
    "hover": "#F0F0F0",
}

FONT_FAMILY = "Segoe UI, -apple-system, Arial, sans-serif"

APP_STYLESHEET = f"""
QWidget {{
    background-color: {COLORS['bg']};
    color: {COLORS['text']};
    font-family: {FONT_FAMILY};
    font-size: 14px;
}}

QMainWindow {{
    background-color: {COLORS['bg']};
}}

#PageScrollArea {{
    border: none;
}}

/* ---- Top header bar ---- */
#TopBar {{
    background-color: {COLORS['bg']};
    border-bottom: 1px solid {COLORS['border']};
}}
#HamburgerButton {{
    border: none;
    font-size: 22px;
    padding: 4px 10px;
    background: transparent;
}}
#HamburgerButton:hover {{
    background-color: {COLORS['hover']};
    border-radius: 6px;
}}
#BackButton {{
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    padding: 5px 12px;
    margin-left: 6px;
    background: transparent;
}}
#BackButton:hover {{
    background-color: {COLORS['hover']};
}}
#AppTitle {{
    font-size: 16px;
    font-weight: 700;
}}

/* ---- Sidebar ---- */
#Sidebar {{
    background-color: {COLORS['sidebar_bg']};
    border-right: 1px solid {COLORS['border']};
}}
#SidebarNavButton {{
    text-align: left;
    padding: 12px 16px;
    border: none;
    border-radius: 8px;
    font-size: 14px;
    background: transparent;
}}
#SidebarNavButton:hover {{
    background-color: {COLORS['hover']};
}}
#SidebarNavButton[active="true"] {{
    background-color: #EDEDED;
    font-weight: 700;
}}

/* ---- Generic card ---- */
.Card, QFrame#Card {{
    background-color: {COLORS['surface']};
    border: 1.5px solid {COLORS['border']};
    border-radius: 14px;
}}

/* ---- Buttons ---- */
QPushButton {{
    border-radius: 8px;
    padding: 10px 18px;
    font-weight: 600;
    border: 1.5px solid {COLORS['border']};
    background-color: {COLORS['bg']};
}}
QPushButton:hover {{
    background-color: {COLORS['hover']};
}}
QPushButton:disabled {{
    color: #A0A0A0;
    border-color: #D0D0D0;
}}

QPushButton#GreenButton {{
    background-color: {COLORS['green_bg']};
    border-color: {COLORS['green']};
    color: {COLORS['green']};
}}
QPushButton#GreenButton:hover {{ background-color: #DBF2E1; }}

QPushButton#OrangeButton {{
    background-color: {COLORS['orange_bg']};
    border-color: {COLORS['orange']};
    color: {COLORS['orange']};
}}
QPushButton#OrangeButton:hover {{ background-color: #FBE5CC; }}

QPushButton#RedButton {{
    background-color: {COLORS['red_bg']};
    border-color: {COLORS['red']};
    color: {COLORS['red']};
}}
QPushButton#RedButton:hover {{ background-color: #FBDAD6; }}

QPushButton#GhostButton {{
    background-color: transparent;
    border: 1.5px solid {COLORS['border']};
}}

/* ---- Inputs ---- */
QLineEdit, QDoubleSpinBox, QSpinBox {{
    border: 1.5px solid {COLORS['border']};
    border-radius: 8px;
    padding: 8px 10px;
    background-color: {COLORS['bg']};
}}

/* ---- Progress bar ---- */
QProgressBar {{
    border: 1.5px solid {COLORS['border']};
    border-radius: 10px;
    text-align: center;
    height: 22px;
    background-color: #FFFFFF;
}}
QProgressBar::chunk {{
    background-color: #2D6CDF;
    border-radius: 8px;
}}

/* ---- Lists / tables ---- */
QListWidget, QTableWidget {{
    border: 1.5px solid {COLORS['border']};
    border-radius: 10px;
}}

QScrollArea {{
    border: none;
}}
"""
