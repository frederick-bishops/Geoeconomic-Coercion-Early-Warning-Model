"""
Geoeconomic Coercion Early-Warning Model
=========================================
Institutional analytical environment for bilateral coercion risk assessment.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_layer import COUNTRIES, load_all_data
from risk_engine import (
    DEFAULT_WEIGHTS,
    compute_composite_risk,
    compute_sector_risk_flags,
    generate_all_scenarios,
    generate_scenario_narrative,
)

st.set_page_config(
    page_title="Geoeconomic Coercion Early-Warning Model",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

SCENARIO_TYPE_LABELS = {
    "trade_coercion": "Trade Coercion",
    "supply_disruption": "Supply Disruption",
    "sanctions_escalation": "Sanctions Escalation",
    "route_vulnerability": "Route Vulnerability",
}

RISK_COLORS = {
    "Critical": "#BF4A3A",
    "High": "#A86A36",
    "Elevated": "#BC9C42",
    "Moderate": "#2C7A7B",
    "Low": "#4F7A5C",
}

SEVERITY_ORDER = ["Low", "Moderate", "Elevated", "High", "Critical"]

DARK_TOKENS = {
    "app_bg": "#0E151A",
    "panel_bg": "#151E24",
    "sidebar_bg": "#121A20",
    "border": "#2A3943",
    "text_primary": "#E4ECEF",
    "text_secondary": "#AFBDC4",
    "text_muted": "#8D9CA5",
    "accent": "#6BA7B8",
    "positive": "#6BAA7A",
    "negative": "#BF4A3A",
    "warning": "#BC9C42",
    "chart_bg": "#151E24",
    "grid": "#33454F",
    "legend": "#C2CED3",
}

LIGHT_TOKENS = {
    "app_bg": "#F6F8FA",
    "panel_bg": "#FFFFFF",
    "sidebar_bg": "#EEF3F6",
    "border": "#D4DEE5",
    "text_primary": "#14222D",
    "text_secondary": "#395062",
    "text_muted": "#5A6D7B",
    "accent": "#2C7A7B",
    "positive": "#4F7A5C",
    "negative": "#BF4A3A",
    "warning": "#A8862C",
    "chart_bg": "#FFFFFF",
    "grid": "#D6E0E7",
    "legend": "#2D4353",
}


@st.cache_data(ttl=3600)
def get_data(seed: int = 42):
    return load_all_data(seed)


@st.cache_data(ttl=3600)
def get_risk_scores(_data, weights_tuple):
    weights = dict(weights_tuple)
    risk_df = compute_composite_risk(_data, weights)
    trade_totals = _data["trade"][["reporter", "partner", "total_trade_b"]].drop_duplicates()
    return risk_df.merge(trade_totals, on=["reporter", "partner"], how="left")


@st.cache_data(ttl=3600)
def get_sector_flags(_data, _risk_df):
    return compute_sector_risk_flags(_data, _risk_df)


def scenario_type_label(scenario_type: str) -> str:
    return SCENARIO_TYPE_LABELS.get(scenario_type, scenario_type.replace("_", " ").title())


def get_theme_tokens() -> dict:
    base = st.get_option("theme.base")
    return DARK_TOKENS if base == "dark" else LIGHT_TOKENS


def apply_styling(tokens: dict) -> None:
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            .stApp {{ background: {tokens['app_bg']}; color: {tokens['text_primary']}; font-family: 'Inter', sans-serif; }}
            section[data-testid="stSidebar"] {{ background: {tokens['sidebar_bg']}; border-right: 1px solid {tokens['border']}; }}
            /* --- Dark mode header/chrome fix --- */
            header[data-testid="stHeader"] {{
                background: {tokens['app_bg']} !important;
                color: {tokens['text_primary']} !important;
            }}
            header[data-testid="stHeader"] button,
            header[data-testid="stHeader"] a,
            header[data-testid="stHeader"] svg {{
                color: {tokens['text_secondary']} !important;
                fill: {tokens['text_secondary']} !important;
                opacity: 0.85;
            }}
            header[data-testid="stHeader"] button:hover,
            header[data-testid="stHeader"] a:hover {{
                color: {tokens['text_primary']} !important;
                opacity: 1;
            }}
            /* --- Dark mode sidebar contrast fix --- */
            section[data-testid="stSidebar"] label,
            section[data-testid="stSidebar"] .stSelectbox label,
            section[data-testid="stSidebar"] .stSlider label {{
                color: {tokens['text_primary']} !important;
            }}
            section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {{
                background: {tokens['panel_bg']} !important;
                border-color: {tokens['border']} !important;
            }}
            section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] * {{
                color: {tokens['text_primary']} !important;
            }}
            section[data-testid="stSidebar"] h3 {{
                color: {tokens['text_primary']} !important;
            }}
            section[data-testid="stSidebar"] .stMarkdown p,
            section[data-testid="stSidebar"] .stMarkdown span {{
                color: {tokens['text_secondary']} !important;
            }}
            section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {{
                color: {tokens['text_primary']} !important;
            }}
            /* --- Tab visibility fix for dark mode --- */
            .stTabs [data-baseweb="tab-list"] button {{
                color: {tokens['text_secondary']} !important;
                font-weight: 500;
            }}
            .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
                color: {tokens['text_primary']} !important;
                font-weight: 600;
                border-bottom-color: {tokens['accent']} !important;
            }}
            .stTabs [data-baseweb="tab-list"] button:hover {{
                color: {tokens['text_primary']} !important;
            }}
            .stTabs [data-baseweb="tab-list"] {{
                border-bottom-color: {tokens['border']} !important;
            }}
            .app-title {{font-size: 1.95rem; font-weight: 650; margin-bottom: 0.25rem;}}
            .app-subtitle {{font-size: 1rem; color: {tokens['text_secondary']}; margin-bottom: 1rem;}}
            .section-title {{font-size: 1.2rem; font-weight: 600; margin-top:0.2rem; margin-bottom:0.6rem;}}
            .memo-block {{background:{tokens['panel_bg']}; border:1px solid {tokens['border']}; border-radius:10px; padding:1rem;}}
            .metric-card {{
                background: {tokens['panel_bg']};
                border: 1px solid {tokens['border']};
                border-left: 4px solid {tokens['accent']};
                border-radius: 10px;
                padding: 1rem;
                min-height: 128px;
            }}
            .metric-card.stack-structural {{ border-left-color: #2C7A7B; }}
            .metric-card.stack-escalation {{ border-left-color: #A8862C; }}
            .metric-card.stack-impact {{ border-left-color: #BF4A3A; }}
            .metric-label {{font-size:0.78rem; text-transform:uppercase; letter-spacing:0.05em; color:{tokens['text_muted']}; white-space:normal;}}
            .metric-value {{font-size:1.7rem; font-weight:700; line-height:1.1; margin:0.2rem 0;}}
            .metric-sub {{font-size:0.82rem; color:{tokens['text_secondary']}; white-space:normal;}}
            .risk-badge {{display:inline-block; border-radius:999px; padding:0.15rem 0.65rem; font-size:0.75rem; font-weight:600;}}
            .scenario-card {{background:{tokens['panel_bg']}; border:1px solid {tokens['border']}; border-left:4px solid {tokens['accent']}; border-radius:10px; padding:0.9rem 1rem; margin-bottom:0.45rem;}}
            .insignia {{font-size:0.73rem; letter-spacing:0.12em; color:{tokens['text_muted']}; text-transform:uppercase;}}
            .data-window {{font-size:0.78rem; color:{tokens['text_secondary']}; margin-bottom:0.5rem;}}
            .sidebar-footer {{margin-top: 2rem; padding-top: 0.75rem; border-top: 1px solid {tokens['border']};}}
            .flag-critical {{background:#BF4A3A22; color:#BF4A3A; border:1px solid #BF4A3A66;}}
            .flag-high {{background:#A8862C22; color:#A8862C; border:1px solid #A8862C66;}}
            .flag-moderate {{background:#2C7A7B22; color:#2C7A7B; border:1px solid #2C7A7B66;}}
            .flag-low {{background:#4F7A5C22; color:#4F7A5C; border:1px solid #4F7A5C66;}}
            .bottom-space {{height: 3.2rem;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_plotly_style(fig: go.Figure, tokens: dict, *, height=360, left=80, right=40, top=40, bottom=60):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=tokens["chart_bg"],
        plot_bgcolor=tokens["chart_bg"],
        font=dict(family="Inter", color=tokens["text_primary"]),
        legend=dict(font=dict(color=tokens["legend"]), orientation="h", y=1.08, x=0),
        margin=dict(t=top, b=bottom, l=left, r=right),
        height=height,
    )
    fig.update_xaxes(gridcolor=tokens["grid"], automargin=True)
    fig.update_yaxes(gridcolor=tokens["grid"], automargin=True)
    return fig


def render_metric_card(title: str, value: str, subtitle: str, variant: str = ""):
    extra = f"stack-{variant}" if variant else ""
    st.markdown(
        f'<div class="metric-card {extra}"><div class="metric-label">{title}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-sub">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


def risk_badge(tier: str):
    color = RISK_COLORS.get(tier, "#6BA7B8")
    text_color = "#101418" if tier == "Elevated" else "#FFFFFF"
    return f'<span class="risk-badge" style="background:{color}; color:{text_color};">{tier}</span>'


def compute_output_stack(row: pd.Series):
    structural = (row["trade_dep_score"] * 0.40 + row["commodity_score"] * 0.30 + row["sector_score"] * 0.30) * 100
    escalation = (row["diplomatic_score"] * 0.45 + row["alliance_risk"] * 0.35 + row["route_score"] * 0.20) * 100
    impact = (row["sector_score"] * 0.40 + row["route_score"] * 0.30 + row["trade_dep_score"] * 0.30) * 100
    return {
        "Structural Vulnerability": structural,
        "Escalation Likelihood": escalation,
        "Impact Severity": impact,
    }


def suggested_action(row: pd.Series, scenario: dict, stack: dict) -> str:
    top_dimension = max(stack, key=stack.get)
    if row["risk_tier"] in ["Critical", "High"]:
        return (
            f"Prioritize stress-test and contingency planning for {top_dimension.lower()}, "
            "monitor diplomatic tone shifts weekly, and compare exposure concentration against peer-country alternatives."
        )
    if row["risk_tier"] == "Elevated":
        return (
            f"Investigate driver sensitivity around {top_dimension.lower()}, review scenario assumptions quarterly, "
            "and monitor route or sector chokepoint developments."
        )
    return "Maintain routine monitoring and compare exposures over time for early directional changes."


def init_default_state():
    keys_map = {
        "w_trade": "trade_dependence",
        "w_commodity": "commodity_concentration",
        "w_diplo": "diplomatic_intensity",
        "w_sector": "sector_exposure",
        "w_route": "route_exposure",
        "w_alliance": "alliance_dynamics",
    }
    for state_key, weight_key in keys_map.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = DEFAULT_WEIGHTS[weight_key]

    if "pair_src" not in st.session_state:
        st.session_state.pair_src = "United States"
    if "pair_dst" not in st.session_state:
        st.session_state.pair_dst = "China"


init_default_state()
tokens = get_theme_tokens()
apply_styling(tokens)

with st.sidebar:
    focus_options = {v["name"]: k for k, v in COUNTRIES.items()}
    st.markdown("### Focus Country")
    focus_name = st.selectbox("Focus Country", list(focus_options.keys()), index=list(focus_options.keys()).index("United States"), label_visibility="collapsed")
    focus_code = focus_options[focus_name]

    st.markdown("### Model Parameters")

    w_trade = st.slider("Trade Dependence", 0.0, 1.0, st.session_state.w_trade, 0.02, key="w_trade")
    w_commodity = st.slider("Commodity Concentration", 0.0, 1.0, st.session_state.w_commodity, 0.02, key="w_commodity")
    w_diplo = st.slider("Diplomatic Intensity", 0.0, 1.0, st.session_state.w_diplo, 0.02, key="w_diplo")
    w_sector = st.slider("Sector Exposure", 0.0, 1.0, st.session_state.w_sector, 0.02, key="w_sector")
    w_route = st.slider("Route Exposure", 0.0, 1.0, st.session_state.w_route, 0.02, key="w_route")
    w_alliance = st.slider("Alliance Dynamics", 0.0, 1.0, st.session_state.w_alliance, 0.02, key="w_alliance")

    if st.button("Reset Weights", use_container_width=True):
        st.session_state.w_trade = DEFAULT_WEIGHTS["trade_dependence"]
        st.session_state.w_commodity = DEFAULT_WEIGHTS["commodity_concentration"]
        st.session_state.w_diplo = DEFAULT_WEIGHTS["diplomatic_intensity"]
        st.session_state.w_sector = DEFAULT_WEIGHTS["sector_exposure"]
        st.session_state.w_route = DEFAULT_WEIGHTS["route_exposure"]
        st.session_state.w_alliance = DEFAULT_WEIGHTS["alliance_dynamics"]
        st.rerun()

    weights = {
        "trade_dependence": w_trade,
        "commodity_concentration": w_commodity,
        "diplomatic_intensity": w_diplo,
        "sector_exposure": w_sector,
        "route_exposure": w_route,
        "alliance_dynamics": w_alliance,
    }

    st.session_state["_last_weight_tuple"] = tuple(sorted(weights.items()))

    st.markdown('<div class="sidebar-footer">', unsafe_allow_html=True)
    st.markdown('<div class="insignia">Designing Decision Systems</div>', unsafe_allow_html=True)
    st.markdown('<div class="data-window">Data window: 2019–2024</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


data = get_data(seed=42)
weights_tuple = tuple(sorted(weights.items()))
risk_df = get_risk_scores(data, weights_tuple)
sector_flags = get_sector_flags(data, risk_df)

st.markdown('<div class="app-title">Geoeconomic Coercion Early-Warning Model</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Decision-support environment for assessing structural exposure, escalation conditions, and likely coercive impact across bilateral trade relationships.</div>',
    unsafe_allow_html=True,
)

tab_summary, tab_drivers, tab_scenarios, tab_actions = st.tabs(
    ["Summary", "Drivers", "Scenarios", "Actions"]
)

with tab_summary:
    st.markdown('<div class="section-title">Analytical Summary</div>', unsafe_allow_html=True)

    focus_pairs = risk_df[risk_df["reporter"] == focus_code]
    if len(focus_pairs) == 0:
        focus_row = risk_df.loc[risk_df["risk_score_100"].idxmax()]
    else:
        focus_row = focus_pairs.loc[focus_pairs["risk_score_100"].idxmax()]

    stack = compute_output_stack(focus_row)
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        render_metric_card("Structural Vulnerability", f"{stack['Structural Vulnerability']:.1f}", "Exposure concentration and dependency profile", "structural")
    with c2:
        render_metric_card("Escalation Likelihood", f"{stack['Escalation Likelihood']:.1f}", "Diplomatic and alignment pressure conditions", "escalation")
    with c3:
        render_metric_card("Impact Severity", f"{stack['Impact Severity']:.1f}", "Likely consequence if pressure materializes", "impact")
    with c4:
        render_metric_card("Overall Composite (Secondary)", f"{focus_row['risk_score_100']:.1f}", f"{risk_badge(str(focus_row['risk_tier']))}", "")

    st.markdown(
        "**Causal chain:** structural exposure → escalation conditions → likely transmission path → likely consequence"
    )

    st.markdown("#### Exposure Profile")
    left_col, right_col = st.columns([3, 2])
    with left_col:
        fig_hist = px.histogram(
            risk_df,
            x="risk_score_100",
            nbins=35,
            color="risk_tier",
            category_orders={"risk_tier": SEVERITY_ORDER[::-1]},
            color_discrete_map=RISK_COLORS,
            labels={"risk_score_100": "Composite Risk Score", "risk_tier": "Risk Tier"},
        )
        apply_plotly_style(fig_hist, tokens, height=360, left=65, right=30, bottom=70)
        st.plotly_chart(fig_hist, use_container_width=True)
    with right_col:
        tier_counts = (
            risk_df["risk_tier"].value_counts().reindex(["Critical", "High", "Elevated", "Moderate", "Low"]).fillna(0)
            .reset_index()
        )
        tier_counts.columns = ["Risk Tier", "Count"]
        fig_pie = px.pie(
            tier_counts,
            values="Count",
            names="Risk Tier",
            color="Risk Tier",
            color_discrete_map=RISK_COLORS,
            hole=0.45,
        )
        fig_pie.update_traces(textposition="outside", textinfo="label+value")
        apply_plotly_style(fig_pie, tokens, height=360, left=45, right=45, bottom=40)
        fig_pie.update_layout(showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    top_pairs = risk_df.nlargest(15, "risk_score_100")[["reporter_name", "partner_name", "risk_score_100", "risk_tier", "total_trade_b"]].copy()
    top_pairs.columns = ["Reporter", "Partner", "Risk Score", "Risk Tier", "Total Trade (B USD)"]
    top_pairs["Risk Score"] = top_pairs["Risk Score"].map(lambda x: f"{x:.1f}")
    top_pairs["Total Trade (B USD)"] = top_pairs["Total Trade (B USD)"].map(lambda x: f"{x:.1f}")
    st.dataframe(
        top_pairs,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Reporter": st.column_config.TextColumn(width="medium"),
            "Partner": st.column_config.TextColumn(width="medium"),
            "Risk Score": st.column_config.TextColumn(width="small"),
            "Risk Tier": st.column_config.TextColumn(width="small"),
            "Total Trade (B USD)": st.column_config.TextColumn(width="small"),
        },
    )

    st.markdown("#### Bilateral Risk Heatmap")
    heatmap_mode = st.radio("Display Mode", ["Major Powers + Focus", "All Countries", "Custom Selection"], horizontal=True)
    if heatmap_mode == "Major Powers + Focus":
        selected_codes = list(dict.fromkeys(["USA", "CHN", "RUS", "DEU", "GBR", "FRA", "JPN", "IND", "KOR", "AUS", "TWN", focus_code]))
    elif heatmap_mode == "All Countries":
        selected_codes = list(COUNTRIES.keys())
    else:
        country_options = {v["name"]: k for k, v in COUNTRIES.items()}
        selected_names = st.multiselect(
            "Select countries",
            list(country_options.keys()),
            default=["United States", "China", "Germany", "Japan", "India", "Taiwan", "South Korea"],
        )
        selected_codes = [country_options[n] for n in selected_names]

    if len(selected_codes) >= 2:
        heatmap_data = risk_df[(risk_df["reporter"].isin(selected_codes)) & (risk_df["partner"].isin(selected_codes))].copy()
        pivot = heatmap_data.pivot_table(index="reporter_name", columns="partner_name", values="risk_score_100", aggfunc="first")
        matrix = pivot.to_numpy(dtype=float)
        text_matrix = np.round(matrix, 1).astype(object)
        n = min(text_matrix.shape)
        for i in range(n):
            text_matrix[i, i] = "—"

        fig_heat = px.imshow(
            matrix,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            aspect="auto",
            color_continuous_scale=[[0, "#E7EFF3"], [0.5, "#B59D4A"], [1.0, "#BF4A3A"]],
            labels=dict(color="Risk Score"),
        )
        apply_plotly_style(
            fig_heat,
            tokens,
            height=max(420, len(selected_codes) * 34),
            left=140,
            right=50,
            bottom=115,
            top=40,
        )
        fig_heat.update_xaxes(tickangle=45)
        fig_heat.update_traces(text=text_matrix, texttemplate="%{text}", textfont=dict(size=9))
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Select at least 2 countries to generate the heatmap.")

with tab_drivers:
    st.markdown('<div class="section-title">Key Drivers</div>', unsafe_allow_html=True)
    name_to_code = {v["name"]: k for k, v in COUNTRIES.items()}
    csrc, cdst = st.columns(2)
    with csrc:
        src_name = st.selectbox("Reporter (vulnerable party)", list(name_to_code.keys()), key="pair_src")
    with cdst:
        dst_options = [n for n in name_to_code.keys() if n != src_name]
        default_dst = "China" if src_name != "China" else "United States"
        if st.session_state.get("pair_dst") not in dst_options:
            st.session_state.pair_dst = default_dst
        dst_name = st.selectbox("Partner (potential coercer)", dst_options, key="pair_dst")

    row = risk_df[(risk_df["reporter"] == name_to_code[src_name]) & (risk_df["partner"] == name_to_code[dst_name])].iloc[0]

    st.markdown(
        f"**Driver summary:** {src_name} → {dst_name} is assessed at {row['risk_score_100']:.1f} ({row['risk_tier']}). "
        "Component-level decomposition below explains relative contribution and transmission risks."
    )

    cols = st.columns(6)
    dims = [
        ("Trade", row["trade_dep_score"]),
        ("Commodity", row["commodity_score"]),
        ("Diplomatic", row["diplomatic_score"]),
        ("Sector", row["sector_score"]),
        ("Route", row["route_score"]),
        ("Alliance", row["alliance_risk"]),
    ]
    for col, (label, val) in zip(cols, dims):
        with col:
            render_metric_card(label, f"{val:.2f}", "Dimension score")

    left, right = st.columns([1.1, 1])
    with left:
        categories = [d[0] for d in dims]
        values = [d[1] for d in dims]
        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                line=dict(color="#2C7A7B", width=2),
                fillcolor="rgba(44,122,123,0.20)",
                name=f"{src_name} → {dst_name}",
            )
        )
        fig_radar.update_layout(polar=dict(radialaxis=dict(range=[0, 1], visible=True)))
        apply_plotly_style(fig_radar, tokens, height=430, left=90, right=90, bottom=60)
        st.plotly_chart(fig_radar, use_container_width=True)
    with right:
        st.markdown("**Risk Dimension Breakdown for Top Partner**")
        st.markdown(f"- Trade dependence: **{row['trade_dependence']:.2%}**")
        st.markdown(f"- Commodity concentration (HHI): **{row['commodity_hhi']:.3f}**")
        st.markdown(f"- Net diplomatic tone: **{row['net_diplomatic_tone']:.2f}**")
        st.markdown(f"- Alliance type: **{row['alliance_type']}**")
        st.markdown(f"- Route exposure score: **{row['route_score']:.2f}**")
        st.markdown(f"- Critical sectors exposed: **{int(row['n_critical_sectors'])}**")

    st.markdown("#### Sector Risk Flags")
    sector_country = st.selectbox("Country", [v["name"] for v in COUNTRIES.values()], index=[v["name"] for v in COUNTRIES.values()].index(src_name))
    sector_code = name_to_code[sector_country]
    cflags = sector_flags[sector_flags["country"] == sector_code].copy().sort_values(
        by="flag_level", key=lambda s: s.map({"Critical": 0, "High": 1, "Moderate": 2, "Low": 3})
    )
    badge_map = {
        "Critical": "<span class='risk-badge flag-critical'>Critical</span>",
        "High": "<span class='risk-badge flag-high'>High</span>",
        "Moderate": "<span class='risk-badge flag-moderate'>Moderate</span>",
        "Low": "<span class='risk-badge flag-low'>Low</span>",
    }
    for _, r in cflags.head(12).iterrows():
        st.markdown(
            f"<div class='scenario-card'><strong>{r['sector_label']}</strong> · {badge_map.get(r['flag_level'], r['flag_level'])}<br/>"
            f"Import dependence: {r['import_dependence']:.1%} · Dominant supplier: {r['supplier_name']}<br/>"
            f"<span style='color:{tokens['text_secondary']};'>{r['reason']}</span></div>",
            unsafe_allow_html=True,
        )

    table_flags = cflags[["sector_label", "flag_level", "import_dependence", "supplier_name", "reason"]].copy()
    table_flags.columns = ["Sector", "Flag", "Import Dependence", "Dominant Supplier", "Assessment"]
    table_flags["Import Dependence"] = table_flags["Import Dependence"].map(lambda x: f"{x:.1%}")
    st.dataframe(
        table_flags,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Sector": st.column_config.TextColumn(width="medium"),
            "Flag": st.column_config.TextColumn(width="small"),
            "Import Dependence": st.column_config.TextColumn(width="small"),
            "Dominant Supplier": st.column_config.TextColumn(width="medium"),
            "Assessment": st.column_config.TextColumn(width="large"),
        },
    )

with tab_scenarios:
    st.markdown('<div class="section-title">Scenario Comparison</div>', unsafe_allow_html=True)
    n_scenarios = st.slider("Number of scenarios", 5, 30, 15, key="n_scenarios")
    scenarios = generate_all_scenarios(risk_df, top_n=n_scenarios)

    scenario_types = [k for k in SCENARIO_TYPE_LABELS if any(s["scenario_type"] == k for s in scenarios)]
    type_filter = st.multiselect(
        "Filter by scenario type",
        scenario_types,
        default=scenario_types,
        format_func=scenario_type_label,
        key="scenario_filter",
        help="Clear all selections to view all scenario types.",
    )
    active_types = type_filter or scenario_types
    filtered = [s for s in scenarios if s["scenario_type"] in active_types]

    for sc in filtered:
        color = RISK_COLORS.get(sc["risk_tier"], "#6BA7B8")
        st.markdown(
            f"<div class='scenario-card' style='border-left-color:{color};'><strong>{sc['reporter_name']} → {sc['partner_name']}</strong> "
            f"{risk_badge(sc['risk_tier'])}<br/>"
            f"<span title='{scenario_type_label(sc['scenario_type'])}' style='font-size:0.8rem; color:{tokens['text_secondary']};'>"
            f"{scenario_type_label(sc['scenario_type'])} · {sc['trigger']}</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(sc["narrative"])
    if not filtered:
        st.info("No scenarios match the selected filters.")

    st.markdown("#### Counterfactual Stress-Test (within existing model dimensions)")
    stress_pair = st.selectbox(
        "Pair for counterfactual comparison",
        [f"{r['reporter_name']} → {r['partner_name']}" for _, r in risk_df.nlargest(20, "risk_score_100").iterrows()],
    )
    stress_dim = st.selectbox("Driver to stress", ["trade_dep_score", "commodity_score", "diplomatic_score", "sector_score", "route_score", "alliance_risk"], format_func=lambda x: x.replace("_", " ").title())
    stress_delta = st.slider("Counterfactual adjustment", -0.25, 0.25, 0.10, 0.01)

    reporter_name, partner_name = [x.strip() for x in stress_pair.split("→")]
    base_row = risk_df[(risk_df["reporter_name"] == reporter_name) & (risk_df["partner_name"] == partner_name)].iloc[0].copy()
    cf_row = base_row.copy()
    cf_row[stress_dim] = float(np.clip(cf_row[stress_dim] + stress_delta, 0.0, 1.0))
    base_stack = compute_output_stack(base_row)
    cf_stack = compute_output_stack(cf_row)

    compare_df = pd.DataFrame(
        {
            "Layer": list(base_stack.keys()),
            "Baseline": list(base_stack.values()),
            "Counterfactual": list(cf_stack.values()),
        }
    )
    fig_compare = px.bar(compare_df.melt(id_vars="Layer", var_name="Case", value_name="Score"), x="Layer", y="Score", color="Case", barmode="group")
    apply_plotly_style(fig_compare, tokens, height=340, left=70, right=30, bottom=60)
    st.plotly_chart(fig_compare, use_container_width=True)

with tab_actions:
    st.markdown('<div class="section-title">Policy Implications</div>', unsafe_allow_html=True)
    highest_pair = risk_df.loc[risk_df["risk_score_100"].idxmax()]
    scenario = generate_scenario_narrative(highest_pair, "sanctions_escalation")
    stack_high = compute_output_stack(highest_pair)

    main_drivers = sorted(
        [
            ("Trade Dependence", highest_pair["trade_dep_score"]),
            ("Commodity Concentration", highest_pair["commodity_score"]),
            ("Diplomatic Intensity", highest_pair["diplomatic_score"]),
            ("Sector Exposure", highest_pair["sector_score"]),
            ("Route Exposure", highest_pair["route_score"]),
            ("Alliance Dynamics", highest_pair["alliance_risk"]),
        ],
        key=lambda x: x[1],
        reverse=True,
    )[:3]

    memo = f"""
**Central analytical finding**
{highest_pair['reporter_name']} → {highest_pair['partner_name']} is currently assessed at **{highest_pair['risk_score_100']:.1f} ({highest_pair['risk_tier']})**.

**Main drivers**
- {main_drivers[0][0]} ({main_drivers[0][1]:.2f})
- {main_drivers[1][0]} ({main_drivers[1][1]:.2f})
- {main_drivers[2][0]} ({main_drivers[2][1]:.2f})

**Most plausible scenario implication**
{scenario_type_label(scenario['scenario_type'])}: {scenario['trigger']}.

**Recommended analytic action**
{suggested_action(highest_pair, scenario, stack_high)}
"""
    st.markdown("<div class='memo-block'>", unsafe_allow_html=True)
    st.markdown(memo)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### Decision Rationale")
    rationale_df = pd.DataFrame(
        {
            "Analytical Layer": list(stack_high.keys()),
            "Score": [round(v, 1) for v in stack_high.values()],
            "Interpretation": [
                "Higher values indicate deeper embedded bilateral exposure.",
                "Higher values indicate stronger near-term escalation conditions.",
                "Higher values indicate larger system impact if coercion occurs.",
            ],
        }
    )
    st.dataframe(rationale_df, use_container_width=True, hide_index=True)

st.markdown("<div class='bottom-space'></div>", unsafe_allow_html=True)
st.markdown(
    f"<div style='text-align:center; font-size:0.75rem; color:{tokens['text_muted']}; padding-bottom:2.2rem;'>"
    "Geoeconomic Coercion Early-Warning Model · Structured early-warning analysis for policy researchers and strategic assessment teams."
    "</div>",
    unsafe_allow_html=True,
)
