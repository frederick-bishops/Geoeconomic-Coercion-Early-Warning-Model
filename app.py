"""
Geoeconomic Coercion Early-Warning Model
=========================================
A forecasting system for trade-related coercion, identifying when bilateral
trade relationships drift toward weaponization, informal retaliation,
export restrictions, or supply-chain pressure.

System: US / EU / China / Russia competition, with exposed mid- and smaller-state nodes.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

from data_layer import COUNTRIES, STRATEGIC_SECTORS, CHOKEPOINTS, load_all_data
from risk_engine import (
    compute_composite_risk,
    compute_sector_risk_flags,
    generate_all_scenarios,
    generate_scenario_narrative,
    DEFAULT_WEIGHTS,
)

# ─────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Geoeconomic Coercion Early-Warning Model",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# Brand colors
# ─────────────────────────────────────────────────────────────────────
COLORS = {
    "bg": "#FCFAF6",
    "text": "#13343B",
    "teal": "#20808D",
    "dark_teal": "#115058",
    "light_teal": "#D6F5FA",
    "rust": "#A84B2F",
    "mauve": "#944454",
    "gold": "#FFC553",
    "olive": "#848456",
    "paper": "#F3F3EE",
    "beige": "#E5E3D4",
    "navy": "#091717",
}

RISK_COLORS = {
    "Critical": "#A84B2F",
    "High": "#944454",
    "Elevated": "#FFC553",
    "Moderate": "#20808D",
    "Low": "#848456",
}

# Chart color sequence
CHART_COLORS = ["#20808D", "#A84B2F", "#1B474D", "#BCE2E7", "#944454", "#FFC553", "#848456", "#6E522B"]

# ─────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #13343B;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 1rem;
        color: #2E565D;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    
    .metric-card {
        background: #F3F3EE;
        border-radius: 8px;
        padding: 1.2rem;
        border-left: 4px solid #20808D;
    }
    .metric-card h3 {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #2E565D;
        margin: 0 0 0.4rem 0;
        font-weight: 600;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #13343B;
        line-height: 1.1;
    }
    .metric-card .delta {
        font-size: 0.8rem;
        color: #2E565D;
        margin-top: 0.2rem;
    }
    
    .risk-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .risk-critical { background: #A84B2F; color: white; }
    .risk-high { background: #944454; color: white; }
    .risk-elevated { background: #FFC553; color: #13343B; }
    .risk-moderate { background: #20808D; color: white; }
    .risk-low { background: #848456; color: white; }
    
    .scenario-card {
        background: #F3F3EE;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #A84B2F;
    }
    .scenario-card h4 {
        margin: 0 0 0.5rem 0;
        color: #13343B;
        font-weight: 600;
    }
    .scenario-card p {
        color: #2E565D;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    .sidebar-info {
        background: #F3F3EE;
        border-radius: 6px;
        padding: 0.8rem;
        font-size: 0.8rem;
        color: #2E565D;
        line-height: 1.5;
    }
    
    div[data-testid="stMetricValue"] {
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Data loading with caching
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_data(seed: int = 42):
    return load_all_data(seed)


@st.cache_data(ttl=3600)
def get_risk_scores(_data, weights_tuple, seed=42):
    weights = dict(weights_tuple)
    risk_df = compute_composite_risk(_data, weights)
    # Merge in total_trade_b for display
    trade_totals = _data["trade"][["reporter", "partner", "total_trade_b"]].drop_duplicates()
    risk_df = risk_df.merge(trade_totals, on=["reporter", "partner"], how="left")
    return risk_df


@st.cache_data(ttl=3600)
def get_sector_flags(_data, _risk_df):
    return compute_sector_risk_flags(_data, _risk_df)


# ─────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Model Parameters")
    
    st.markdown("**Dimension Weights**")
    st.markdown(
        '<div class="sidebar-info">Adjust the relative weight of each risk dimension. '
        'Weights are normalized to sum to 1.0.</div>',
        unsafe_allow_html=True,
    )
    
    w_trade = st.slider("Trade Dependence", 0.0, 1.0, DEFAULT_WEIGHTS["trade_dependence"], 0.02, key="w_trade")
    w_commodity = st.slider("Commodity Concentration", 0.0, 1.0, DEFAULT_WEIGHTS["commodity_concentration"], 0.02, key="w_commodity")
    w_diplo = st.slider("Diplomatic Intensity", 0.0, 1.0, DEFAULT_WEIGHTS["diplomatic_intensity"], 0.02, key="w_diplo")
    w_sector = st.slider("Sector Exposure", 0.0, 1.0, DEFAULT_WEIGHTS["sector_exposure"], 0.02, key="w_sector")
    w_route = st.slider("Route Exposure", 0.0, 1.0, DEFAULT_WEIGHTS["route_exposure"], 0.02, key="w_route")
    w_alliance = st.slider("Alliance Dynamics", 0.0, 1.0, DEFAULT_WEIGHTS["alliance_dynamics"], 0.02, key="w_alliance")
    
    weights = {
        "trade_dependence": w_trade,
        "commodity_concentration": w_commodity,
        "diplomatic_intensity": w_diplo,
        "sector_exposure": w_sector,
        "route_exposure": w_route,
        "alliance_dynamics": w_alliance,
    }
    
    st.divider()
    
    st.markdown("### Focus Country")
    focus_options = {v["name"]: k for k, v in COUNTRIES.items()}
    focus_name = st.selectbox(
        "Select a country to analyze",
        list(focus_options.keys()),
        index=list(focus_options.keys()).index("Ghana"),
    )
    focus_code = focus_options[focus_name]
    
    st.divider()
    
    st.markdown("### About This Model")
    st.markdown(
        '<div class="sidebar-info">'
        '<strong>Geoeconomic Coercion Early-Warning Model</strong><br><br>'
        'A forecasting system for trade-related coercion across the '
        'US / EU / China / Russia competitive system.<br><br>'
        '<strong>Six dimensions:</strong><br>'
        '• Trade dependence (asymmetric exposure)<br>'
        '• Commodity concentration (HHI)<br>'
        '• Diplomatic event intensity<br>'
        '• Strategic sector chokepoints<br>'
        '• Maritime route exposure<br>'
        '• Alliance / bloc dynamics<br><br>'
        '<strong>Data:</strong> Calibrated synthetic data modeled on '
        'real-world trade volumes, GDELT event patterns, and geopolitical alignments. '
        'Designed for integration with UN Comtrade, GDELT, and World Bank APIs.'
        '</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────
# Load data and compute scores
# ─────────────────────────────────────────────────────────────────────
data = get_data(seed=42)
weights_tuple = tuple(sorted(weights.items()))
risk_df = get_risk_scores(data, weights_tuple, seed=42)
sector_flags = get_sector_flags(data, risk_df)

# ─────────────────────────────────────────────────────────────────────
# Tab navigation
# ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">Geoeconomic Coercion Early-Warning Model</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Forecasting trade-related coercion across the US / EU / China / Russia system'
    '</div>',
    unsafe_allow_html=True,
)

tab_overview, tab_heatmap, tab_pair, tab_sector, tab_scenario, tab_data = st.tabs([
    "Overview Dashboard",
    "Risk Heatmap",
    "Pair-Level Analysis",
    "Sector Risk Flags",
    "Scenario Explorer",
    "Data Export",
])


# ═════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW DASHBOARD
# ═════════════════════════════════════════════════════════════════════
with tab_overview:
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    n_critical = len(risk_df[risk_df["risk_tier"] == "Critical"])
    n_high = len(risk_df[risk_df["risk_tier"] == "High"])
    avg_risk = risk_df["risk_score_100"].mean()
    max_risk_pair = risk_df.loc[risk_df["risk_score_100"].idxmax()]
    
    with col1:
        st.markdown(
            f'<div class="metric-card"><h3>Critical Risk Pairs</h3>'
            f'<div class="value">{n_critical}</div>'
            f'<div class="delta">of {len(risk_df)} total pairs</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card"><h3>High Risk Pairs</h3>'
            f'<div class="value">{n_high}</div>'
            f'<div class="delta">requiring active monitoring</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="metric-card"><h3>System Avg Risk</h3>'
            f'<div class="value">{avg_risk:.1f}</div>'
            f'<div class="delta">out of 100</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="metric-card"><h3>Highest Risk Pair</h3>'
            f'<div class="value">{max_risk_pair["risk_score_100"]:.1f}</div>'
            f'<div class="delta">{max_risk_pair["reporter_name"]} → {max_risk_pair["partner_name"]}</div></div>',
            unsafe_allow_html=True,
        )
    
    st.markdown("---")
    
    # Risk distribution
    left_col, right_col = st.columns([3, 2])
    
    with left_col:
        st.markdown("#### Risk Score Distribution")
        fig_hist = px.histogram(
            risk_df,
            x="risk_score_100",
            nbins=40,
            color="risk_tier",
            color_discrete_map=RISK_COLORS,
            labels={"risk_score_100": "Composite Risk Score", "risk_tier": "Risk Tier"},
            category_orders={"risk_tier": ["Low", "Moderate", "Elevated", "High", "Critical"]},
        )
        fig_hist.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter", color=COLORS["text"]),
            bargap=0.05,
            height=380,
            margin=dict(t=20, b=40, l=50, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with right_col:
        st.markdown("#### Risk Tier Breakdown")
        tier_counts = risk_df["risk_tier"].value_counts().reindex(
            ["Critical", "High", "Elevated", "Moderate", "Low"]
        ).fillna(0).reset_index()
        tier_counts.columns = ["Risk Tier", "Count"]
        
        fig_pie = px.pie(
            tier_counts,
            values="Count",
            names="Risk Tier",
            color="Risk Tier",
            color_discrete_map=RISK_COLORS,
            hole=0.4,
        )
        fig_pie.update_traces(textposition="outside", textinfo="label+value")
        fig_pie.update_layout(
            showlegend=False,
            height=380,
            margin=dict(t=20, b=20, l=20, r=20),
            font=dict(family="Inter", color=COLORS["text"]),
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Top risk pairs table
    st.markdown("#### Top 15 Highest-Risk Bilateral Pairs")
    top_pairs = risk_df.nlargest(15, "risk_score_100")[
        ["reporter_name", "partner_name", "risk_score_100", "risk_tier",
         "trade_dep_score", "commodity_score", "diplomatic_score",
         "sector_score", "alliance_risk"]
    ].copy()
    top_pairs.columns = [
        "Reporter", "Partner", "Risk Score", "Risk Tier",
        "Trade Dep.", "Commodity", "Diplomatic", "Sector", "Alliance"
    ]
    for col in ["Trade Dep.", "Commodity", "Diplomatic", "Sector", "Alliance"]:
        top_pairs[col] = top_pairs[col].apply(lambda x: f"{x:.2f}")
    top_pairs["Risk Score"] = top_pairs["Risk Score"].apply(lambda x: f"{x:.1f}")
    
    st.dataframe(
        top_pairs,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Risk Tier": st.column_config.TextColumn(width="small"),
            "Risk Score": st.column_config.TextColumn(width="small"),
        },
    )
    
    # Focus country analysis
    st.markdown("---")
    st.markdown(f"#### {focus_name}: Vulnerability Profile")
    
    focus_incoming = risk_df[risk_df["reporter"] == focus_code].nlargest(10, "risk_score_100")
    
    if len(focus_incoming) > 0:
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown(f"**Top Coercion Risks to {focus_name}**")
            fig_focus = px.bar(
                focus_incoming,
                y="partner_name",
                x="risk_score_100",
                orientation="h",
                color="risk_tier",
                color_discrete_map=RISK_COLORS,
                labels={"risk_score_100": "Risk Score", "partner_name": "Partner"},
                category_orders={"risk_tier": ["Critical", "High", "Elevated", "Moderate", "Low"]},
            )
            fig_focus.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(family="Inter", color=COLORS["text"]),
                height=350,
                margin=dict(t=10, b=40, l=100, r=20),
                showlegend=False,
                yaxis=dict(categoryorder="total ascending"),
            )
            st.plotly_chart(fig_focus, use_container_width=True)
        
        with col_b:
            st.markdown(f"**{focus_name} Risk Dimension Breakdown (Top Partner)**")
            top_partner_row = focus_incoming.iloc[0]
            dims = {
                "Trade Dependence": top_partner_row["trade_dep_score"],
                "Commodity Conc.": top_partner_row["commodity_score"],
                "Diplomatic Tension": top_partner_row["diplomatic_score"],
                "Sector Exposure": top_partner_row["sector_score"],
                "Route Exposure": top_partner_row["route_score"],
                "Alliance Risk": top_partner_row["alliance_risk"],
            }
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=list(dims.values()),
                theta=list(dims.keys()),
                fill="toself",
                fillcolor="rgba(32, 128, 141, 0.2)",
                line=dict(color=COLORS["teal"], width=2),
                name=f"vs {top_partner_row['partner_name']}",
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                height=350,
                margin=dict(t=30, b=30, l=60, r=60),
                font=dict(family="Inter", color=COLORS["text"]),
            )
            st.plotly_chart(fig_radar, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════
# TAB 2: RISK HEATMAP
# ═════════════════════════════════════════════════════════════════════
with tab_heatmap:
    st.markdown("#### Bilateral Coercion Risk Heatmap")
    st.markdown(
        "Rows = reporter (vulnerable party), Columns = partner (potential coercer). "
        "Darker cells indicate higher coercion risk."
    )
    
    # Country filter for heatmap
    heatmap_mode = st.radio(
        "Display mode",
        ["Major Powers + Focus", "All Countries", "Custom Selection"],
        horizontal=True,
        key="heatmap_mode",
    )
    
    if heatmap_mode == "Major Powers + Focus":
        major_codes = ["USA", "CHN", "RUS", "DEU", "GBR", "FRA", "JPN", "IND", "KOR",
                       "AUS", "TWN", "TUR", "SAU", focus_code]
        major_codes = list(dict.fromkeys(major_codes))  # deduplicate
    elif heatmap_mode == "All Countries":
        major_codes = list(COUNTRIES.keys())
    else:
        country_options = {v["name"]: k for k, v in COUNTRIES.items()}
        selected_names = st.multiselect(
            "Select countries",
            list(country_options.keys()),
            default=["United States", "China", "Russia", "Germany", "Ghana", "Serbia", "Taiwan"],
        )
        major_codes = [country_options[n] for n in selected_names]
    
    if len(major_codes) >= 2:
        heatmap_data = risk_df[
            (risk_df["reporter"].isin(major_codes)) &
            (risk_df["partner"].isin(major_codes))
        ].copy()
        
        pivot = heatmap_data.pivot_table(
            index="reporter_name",
            columns="partner_name",
            values="risk_score_100",
            aggfunc="first",
        )
        
        fig_heat = px.imshow(
            pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            color_continuous_scale=[
                [0, "#F3F3EE"],
                [0.25, "#D6F5FA"],
                [0.5, "#FFC553"],
                [0.75, "#944454"],
                [1.0, "#A84B2F"],
            ],
            labels=dict(color="Risk Score"),
            aspect="auto",
        )
        fig_heat.update_layout(
            height=max(400, len(major_codes) * 35),
            font=dict(family="Inter", color=COLORS["text"]),
            margin=dict(t=30, b=40, l=120, r=30),
            xaxis=dict(tickangle=45),
        )
        fig_heat.update_traces(
            text=np.round(pivot.values, 1),
            texttemplate="%{text:.0f}",
            textfont=dict(size=9),
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Select at least 2 countries to generate the heatmap.")


# ═════════════════════════════════════════════════════════════════════
# TAB 3: PAIR-LEVEL ANALYSIS
# ═════════════════════════════════════════════════════════════════════
with tab_pair:
    st.markdown("#### Pair-Level Deep Dive")
    
    col_src, col_dst = st.columns(2)
    country_names = {v["name"]: k for k, v in COUNTRIES.items()}
    
    with col_src:
        src_name = st.selectbox("Reporter (vulnerable party)", list(country_names.keys()),
                                index=list(country_names.keys()).index("Ghana"), key="pair_src")
    with col_dst:
        dst_options = [n for n in country_names.keys() if n != src_name]
        dst_name = st.selectbox("Partner (potential coercer)", dst_options,
                                index=dst_options.index("China") if "China" in dst_options else 0,
                                key="pair_dst")
    
    src_code = country_names[src_name]
    dst_code = country_names[dst_name]
    
    pair_data = risk_df[(risk_df["reporter"] == src_code) & (risk_df["partner"] == dst_code)]
    
    if len(pair_data) > 0:
        row = pair_data.iloc[0]
        
        # Risk score header
        risk_class = str(row["risk_tier"]).lower()
        st.markdown(
            f'<div style="display:flex; align-items:center; gap:1rem; margin-bottom:1rem;">'
            f'<span style="font-size:2.5rem; font-weight:700; color:#13343B;">{row["risk_score_100"]:.1f}</span>'
            f'<span class="risk-badge risk-{risk_class}">{row["risk_tier"]}</span>'
            f'<span style="color:#2E565D;">Composite Coercion Risk Score</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        
        # Dimension scores
        st.markdown("**Component Scores**")
        dim_cols = st.columns(6)
        dimensions = [
            ("Trade Dep.", row["trade_dep_score"], COLORS["teal"]),
            ("Commodity", row["commodity_score"], COLORS["rust"]),
            ("Diplomatic", row["diplomatic_score"], COLORS["mauve"]),
            ("Sector", row["sector_score"], COLORS["dark_teal"]),
            ("Route", row["route_score"], COLORS["gold"]),
            ("Alliance", row["alliance_risk"], COLORS["olive"]),
        ]
        
        for col, (name, val, color) in zip(dim_cols, dimensions):
            with col:
                st.markdown(
                    f'<div style="background:#F3F3EE; border-radius:6px; padding:0.8rem; text-align:center;">'
                    f'<div style="font-size:0.7rem; text-transform:uppercase; color:#2E565D; letter-spacing:0.04em;">{name}</div>'
                    f'<div style="font-size:1.5rem; font-weight:700; color:{color};">{val:.2f}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        
        st.markdown("---")
        
        # Detailed breakdown
        left, right = st.columns(2)
        
        with left:
            st.markdown("**Trade Profile**")
            trade_row = data["trade"][
                (data["trade"]["reporter"] == src_code) & (data["trade"]["partner"] == dst_code)
            ]
            if len(trade_row) > 0:
                tr = trade_row.iloc[0]
                st.markdown(f"- Total bilateral trade: **${tr['total_trade_b']:.1f}B**")
                st.markdown(f"- {src_name} exports to {dst_name}: **${tr['exports_b']:.1f}B**")
                st.markdown(f"- {src_name} imports from {dst_name}: **${tr['imports_b']:.1f}B**")
                st.markdown(f"- Trade dependence: **{row['trade_dependence']:.2%}**")
                st.markdown(f"- Import dependence: **{row['import_dependence']:.2%}**")
                asymmetry = row['dependence_asymmetry']
                direction = "more" if asymmetry > 0 else "less"
                st.markdown(f"- Dependence asymmetry: **{abs(asymmetry):.2%}** ({src_name} is {direction} dependent)")
            
            st.markdown("**Commodity Concentration**")
            st.markdown(f"- HHI: **{row['commodity_hhi']:.4f}**")
            sector_label = STRATEGIC_SECTORS.get(row['top_sector'], {}).get('label', row['top_sector'])
            st.markdown(f"- Top sector: **{sector_label}**")
            st.markdown(f"- Top sector share: **{row['top_sector_share']:.1%}**")
        
        with right:
            st.markdown("**Diplomatic & Alliance Profile**")
            st.markdown(f"- Net diplomatic tone: **{row['net_diplomatic_tone']:.3f}** (-1=hostile, 1=cooperative)")
            st.markdown(f"- Conflict score: **{row['conflict_score']:.1f}**/10")
            st.markdown(f"- Cooperation score: **{row['cooperation_score']:.1f}**/10")
            st.markdown(f"- 3-month trend: **{row['recent_trend_3m']:.3f}**")
            st.markdown(f"- Alliance type: **{row['alliance_type']}**")
            st.markdown(f"- Alignment score: **{row['alignment_score']:.3f}**")
            
            st.markdown("**Route Exposure**")
            st.markdown(f"- Route exposure score: **{row['route_score']:.3f}**")
            st.markdown(f"- Chokepoint dependencies: **{row['n_chokepoint_dependencies']}**")
        
        # Radar comparison
        st.markdown("---")
        st.markdown("**Risk Dimension Radar**")
        
        fig_radar2 = go.Figure()
        cats = ["Trade Dep.", "Commodity", "Diplomatic", "Sector", "Route", "Alliance"]
        vals = [row["trade_dep_score"], row["commodity_score"], row["diplomatic_score"],
                row["sector_score"], row["route_score"], row["alliance_risk"]]
        
        fig_radar2.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]],
            fill="toself",
            fillcolor="rgba(168, 75, 47, 0.15)",
            line=dict(color=COLORS["rust"], width=2),
            name=f"{src_name} → {dst_name}",
        ))
        fig_radar2.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=400,
            margin=dict(t=30, b=30, l=80, r=80),
            font=dict(family="Inter", color=COLORS["text"]),
        )
        st.plotly_chart(fig_radar2, use_container_width=True)
        
        # Scenario narrative
        st.markdown("---")
        st.markdown("**Escalation Scenario**")
        
        scenario_type = st.selectbox(
            "Scenario type",
            ["trade_coercion", "supply_disruption", "sanctions_escalation", "route_vulnerability"],
            format_func=lambda x: x.replace("_", " ").title(),
            key="pair_scenario_type",
        )
        
        scenario = generate_scenario_narrative(row, scenario_type)
        
        tier_lower = scenario["risk_tier"].lower()
        st.markdown(
            f'<div class="scenario-card" style="border-left-color:{RISK_COLORS.get(scenario["risk_tier"], "#20808D")}">'
            f'<h4>{scenario["trigger"]}</h4>'
            f'<p>{scenario["narrative"]}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("No data found for this pair.")


# ═════════════════════════════════════════════════════════════════════
# TAB 4: SECTOR RISK FLAGS
# ═════════════════════════════════════════════════════════════════════
with tab_sector:
    st.markdown("#### Strategic Sector Risk Flags")
    st.markdown(
        "Sector-level vulnerabilities by country: dependency on adversarial or concentrated suppliers "
        "for critical inputs."
    )
    
    sector_view = st.radio(
        "View by",
        ["Country", "Sector"],
        horizontal=True,
        key="sector_view_mode",
    )
    
    if sector_view == "Country":
        sector_country = st.selectbox(
            "Select country",
            [v["name"] for v in COUNTRIES.values()],
            index=list(COUNTRIES.values()).index(COUNTRIES[focus_code]),
            key="sector_country_sel",
        )
        sector_code = {v["name"]: k for k, v in COUNTRIES.items()}[sector_country]
        
        country_flags = sector_flags[sector_flags["country"] == sector_code].copy()
        
        # Color-coded flag display
        flag_order = {"Critical": 0, "High": 1, "Moderate": 2, "Low": 3}
        country_flags["sort_key"] = country_flags["flag_level"].map(flag_order)
        country_flags = country_flags.sort_values("sort_key")
        
        fig_sector = px.bar(
            country_flags,
            y="sector_label",
            x="import_dependence",
            color="flag_level",
            color_discrete_map=RISK_COLORS,
            orientation="h",
            labels={"import_dependence": "Import Dependence", "sector_label": "Sector"},
            category_orders={"flag_level": ["Critical", "High", "Moderate", "Low"]},
        )
        fig_sector.update_layout(
            height=450,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter", color=COLORS["text"]),
            margin=dict(t=10, b=40, l=180, r=20),
            yaxis=dict(categoryorder="total ascending"),
            legend_title="Risk Flag",
        )
        st.plotly_chart(fig_sector, use_container_width=True)
        
        # Detail table
        display_flags = country_flags[
            ["sector_label", "flag_level", "import_dependence", "supplier_concentration",
             "substitutability", "supplier_name", "reason"]
        ].copy()
        display_flags.columns = ["Sector", "Flag", "Import Dep.", "Supplier Conc.",
                                  "Substitutability", "Dominant Supplier", "Assessment"]
        display_flags["Import Dep."] = display_flags["Import Dep."].apply(lambda x: f"{x:.1%}")
        display_flags["Supplier Conc."] = display_flags["Supplier Conc."].apply(lambda x: f"{x:.1%}")
        display_flags["Substitutability"] = display_flags["Substitutability"].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(display_flags, use_container_width=True, hide_index=True)
    
    else:  # Sector view
        sector_choice = st.selectbox(
            "Select sector",
            [v["label"] for v in STRATEGIC_SECTORS.values()],
            key="sector_sel",
        )
        sector_key = {v["label"]: k for k, v in STRATEGIC_SECTORS.items()}[sector_choice]
        
        sector_data = sector_flags[sector_flags["sector"] == sector_key].copy()
        
        flag_order = {"Critical": 0, "High": 1, "Moderate": 2, "Low": 3}
        sector_data["sort_key"] = sector_data["flag_level"].map(flag_order)
        sector_data = sector_data.sort_values("sort_key")
        
        fig_sec2 = px.bar(
            sector_data,
            y="country_name",
            x="import_dependence",
            color="flag_level",
            color_discrete_map=RISK_COLORS,
            orientation="h",
            labels={"import_dependence": "Import Dependence", "country_name": "Country"},
            category_orders={"flag_level": ["Critical", "High", "Moderate", "Low"]},
        )
        fig_sec2.update_layout(
            height=max(400, len(sector_data) * 25),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter", color=COLORS["text"]),
            margin=dict(t=10, b=40, l=120, r=20),
            yaxis=dict(categoryorder="total ascending"),
            legend_title="Risk Flag",
        )
        st.plotly_chart(fig_sec2, use_container_width=True)
        
        # Chokepoint holders
        holders = STRATEGIC_SECTORS[sector_key]["chokepoint_holders"]
        holder_names = [COUNTRIES.get(h, {}).get("name", h) for h in holders]
        st.markdown(f"**Chokepoint holders for {sector_choice}:** {', '.join(holder_names)}")


# ═════════════════════════════════════════════════════════════════════
# TAB 5: SCENARIO EXPLORER
# ═════════════════════════════════════════════════════════════════════
with tab_scenario:
    st.markdown("#### Escalation Scenario Explorer")
    st.markdown(
        "Auto-generated scenario narratives for the highest-risk bilateral pairs. "
        "Each scenario describes a plausible escalation path based on the composite risk dimensions."
    )
    
    n_scenarios = st.slider("Number of scenarios", 5, 30, 15, key="n_scenarios")
    scenarios = generate_all_scenarios(risk_df, top_n=n_scenarios)
    
    # Filter by scenario type
    scenario_types = list(set(s["scenario_type"] for s in scenarios))
    type_filter = st.multiselect(
        "Filter by scenario type",
        scenario_types,
        default=scenario_types,
        format_func=lambda x: x.replace("_", " ").title(),
        key="scenario_filter",
    )
    
    filtered = [s for s in scenarios if s["scenario_type"] in type_filter]
    
    for i, sc in enumerate(filtered):
        risk_tier = sc["risk_tier"]
        color = RISK_COLORS.get(risk_tier, "#20808D")
        risk_class = risk_tier.lower()
        
        st.markdown(
            f'<div class="scenario-card" style="border-left-color:{color};">'
            f'<div style="display:flex; justify-content:space-between; align-items:center;">'
            f'<h4>{sc["reporter_name"]} → {sc["partner_name"]}</h4>'
            f'<div><span class="risk-badge risk-{risk_class}">{risk_tier}</span> '
            f'<span style="font-weight:600; color:#13343B; margin-left:0.5rem;">{sc["risk_score"]:.1f}</span></div>'
            f'</div>'
            f'<div style="font-size:0.75rem; color:{color}; text-transform:uppercase; '
            f'letter-spacing:0.04em; margin-bottom:0.5rem; font-weight:600;">'
            f'{sc["trigger"]}</div>'
            f'<p>{sc["narrative"]}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    
    if not filtered:
        st.info("No scenarios match the selected filters.")


# ═════════════════════════════════════════════════════════════════════
# TAB 6: DATA EXPORT
# ═════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown("#### Data Export")
    st.markdown("Download risk scores, sector flags, and scenario narratives for further analysis.")
    
    col_e1, col_e2, col_e3 = st.columns(3)
    
    with col_e1:
        st.markdown("**Risk Scores (All Pairs)**")
        export_risk = risk_df[[
            "reporter", "reporter_name", "partner", "partner_name",
            "risk_score_100", "risk_tier",
            "trade_dep_score", "commodity_score", "diplomatic_score",
            "sector_score", "route_score", "alliance_risk",
            "trade_dependence", "commodity_hhi", "net_diplomatic_tone",
            "alignment_score", "alliance_type",
        ]].copy()
        
        csv_risk = export_risk.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Risk Scores CSV",
            data=csv_risk,
            file_name="coercion_risk_scores.csv",
            mime="text/csv",
        )
        st.caption(f"{len(export_risk)} bilateral pairs")
    
    with col_e2:
        st.markdown("**Sector Risk Flags**")
        csv_sector = sector_flags.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Sector Flags CSV",
            data=csv_sector,
            file_name="sector_risk_flags.csv",
            mime="text/csv",
        )
        st.caption(f"{len(sector_flags)} country-sector flags")
    
    with col_e3:
        st.markdown("**Scenario Narratives**")
        all_scenarios = generate_all_scenarios(risk_df, top_n=30)
        df_scenarios = pd.DataFrame(all_scenarios)
        csv_scenarios = df_scenarios.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Scenarios CSV",
            data=csv_scenarios,
            file_name="escalation_scenarios.csv",
            mime="text/csv",
        )
        st.caption(f"{len(all_scenarios)} scenario narratives")
    
    # Excel export
    st.markdown("---")
    st.markdown("**Full Export (Excel)**")
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        export_risk.to_excel(writer, sheet_name="Risk Scores", index=False)
        sector_flags.to_excel(writer, sheet_name="Sector Flags", index=False)
        pd.DataFrame(all_scenarios).to_excel(writer, sheet_name="Scenarios", index=False)
        data["trade"][["reporter", "reporter_name", "partner", "partner_name",
                        "exports_b", "imports_b", "total_trade_b",
                        "trade_dependence", "export_dependence", "import_dependence"]
        ].to_excel(writer, sheet_name="Trade Data", index=False)
    
    st.download_button(
        label="Download Full Report (Excel)",
        data=buffer.getvalue(),
        file_name="geoeconomic_coercion_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ─────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center; font-size:0.75rem; color:#2E565D; padding:1rem 0;">'
    'Geoeconomic Coercion Early-Warning Model · Calibrated synthetic data · '
    'Designed for integration with UN Comtrade, GDELT, and World Bank APIs · '
    'Built with Streamlit'
    '</div>',
    unsafe_allow_html=True,
)
