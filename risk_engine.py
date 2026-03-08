"""
risk_engine.py — Composite risk scoring engine for geoeconomic coercion.

Combines six dimensions into a pair-level coercion risk score:
1. Trade dependence (asymmetric bilateral exposure)
2. Commodity concentration (HHI-based)
3. Diplomatic event intensity (conflict / cooperation tone)
4. Strategic sector concentration (chokepoint dependency)
5. Maritime route exposure
6. Alliance / bloc dynamics (adversarial vs. aligned)

Also generates:
- Sector-level risk flags
- Scenario narratives for escalation paths
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from data_layer import COUNTRIES, STRATEGIC_SECTORS, CHOKEPOINTS


# ─────────────────────────────────────────────────────────────────────
# Score normalization helpers
# ─────────────────────────────────────────────────────────────────────
def min_max_normalize(series: pd.Series, invert: bool = False) -> pd.Series:
    """Normalize a series to 0-1 range."""
    smin, smax = series.min(), series.max()
    if smax == smin:
        return pd.Series(0.5, index=series.index)
    normalized = (series - smin) / (smax - smin)
    if invert:
        normalized = 1 - normalized
    return normalized


def sigmoid_transform(x: float, midpoint: float = 0.5, steepness: float = 10) -> float:
    """Apply sigmoid to create non-linear risk mapping."""
    return 1.0 / (1.0 + np.exp(-steepness * (x - midpoint)))


# ─────────────────────────────────────────────────────────────────────
# Dimension scorers
# ─────────────────────────────────────────────────────────────────────
def score_trade_dependence(trade_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score bilateral trade dependence.
    Higher score = reporter is more dependent on partner (vulnerable).
    Key insight: asymmetry matters. If A depends on B more than B depends on A,
    B has coercive leverage.
    """
    df = trade_df[["reporter", "partner", "trade_dependence", "import_dependence", "export_dependence"]].copy()
    
    # Create reverse lookup for asymmetry
    reverse = df[["reporter", "partner", "trade_dependence"]].rename(
        columns={"reporter": "partner", "partner": "reporter", "trade_dependence": "partner_dependence_on_reporter"}
    )
    df = df.merge(reverse, on=["reporter", "partner"], how="left")
    df["partner_dependence_on_reporter"] = df["partner_dependence_on_reporter"].fillna(0)
    
    # Asymmetry: how much more the reporter depends on partner than vice versa
    df["dependence_asymmetry"] = df["trade_dependence"] - df["partner_dependence_on_reporter"]
    
    # Weighted score: high dependence + high asymmetry = high vulnerability
    df["trade_dep_raw"] = (
        0.4 * min_max_normalize(df["trade_dependence"]) +
        0.3 * min_max_normalize(df["import_dependence"]) +
        0.3 * min_max_normalize(df["dependence_asymmetry"].clip(lower=0))
    )
    df["trade_dep_score"] = min_max_normalize(df["trade_dep_raw"])
    
    return df[["reporter", "partner", "trade_dep_score", "trade_dependence",
               "dependence_asymmetry", "import_dependence", "export_dependence"]]


def score_commodity_concentration(commodity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score commodity concentration risk.
    Higher HHI = more concentrated trade = higher vulnerability to targeted restriction.
    """
    df = commodity_df.copy()
    df["commodity_score"] = (
        0.6 * min_max_normalize(df["commodity_hhi"]) +
        0.4 * min_max_normalize(df["top_sector_share"])
    )
    return df[["reporter", "partner", "commodity_score", "commodity_hhi",
               "top_sector", "top_sector_share"]]


def score_diplomatic_intensity(diplo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score diplomatic tension.
    Lower net tone + negative trend = higher coercion risk.
    """
    df = diplo_df.copy()
    
    # Invert net_diplomatic_tone (more negative = more risk)
    df["tone_risk"] = min_max_normalize(df["net_diplomatic_tone"], invert=True)
    
    # Negative trend amplifies risk
    df["trend_risk"] = min_max_normalize(-df["recent_trend_3m"])
    
    # High event volume with negative tone = escalation signal
    df["salience_risk"] = min_max_normalize(df["event_volume"]) * df["tone_risk"]
    
    df["diplomatic_score"] = (
        0.5 * df["tone_risk"] +
        0.25 * df["trend_risk"] +
        0.25 * df["salience_risk"]
    )
    df["diplomatic_score"] = min_max_normalize(df["diplomatic_score"])
    
    return df[["reporter", "partner", "diplomatic_score", "net_diplomatic_tone",
               "conflict_score", "cooperation_score", "recent_trend_3m"]]


def score_sector_exposure(sector_df: pd.DataFrame, trade_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score strategic sector concentration for each bilateral pair.
    If reporter imports critical sectors from partner, and partner dominates supply,
    the pair has high strategic sector risk.
    """
    # For each reporter-partner pair, check which strategic sectors
    # the reporter depends on from the partner
    codes = list(COUNTRIES.keys())
    rows = []
    
    for src in codes:
        src_sectors = sector_df[sector_df["country"] == src]
        
        for dst in codes:
            if src == dst:
                continue
            
            # Check: for each sector where src has high import dependence,
            # is dst the dominant supplier?
            sector_risk = 0
            critical_sectors = []
            n_critical = 0
            
            for _, row in src_sectors.iterrows():
                if row["dominant_supplier"] == dst and row["import_dependence"] > 0.3:
                    # Weight by (1 - substitutability) * import_dependence * supplier_concentration
                    risk = (1 - row["substitutability"]) * row["import_dependence"] * row["supplier_concentration"]
                    sector_risk += risk
                    if risk > 0.15:
                        critical_sectors.append(row["sector_label"])
                        n_critical += 1
            
            rows.append({
                "reporter": src,
                "partner": dst,
                "sector_risk_raw": round(sector_risk, 4),
                "n_critical_sectors": n_critical,
                "critical_sectors": "; ".join(critical_sectors) if critical_sectors else "None",
            })
    
    df = pd.DataFrame(rows)
    df["sector_score"] = min_max_normalize(df["sector_risk_raw"])
    return df


def score_route_exposure(route_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score route exposure.
    This is country-level, not pair-level. We'll use it as a multiplier.
    """
    df = route_df.copy()
    df["route_score"] = min_max_normalize(df["route_exposure_score"])
    return df[["country", "route_score", "route_exposure_score",
               "n_chokepoint_dependencies", "critical_chokepoints"]]


def score_alliance_dynamics(alliance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score alliance / bloc dynamics.
    Adversarial relationships = higher coercion risk.
    Cross-bloc trade = more vulnerable to weaponization.
    """
    df = alliance_df.copy()
    # Invert alignment: more negative = higher risk
    df["alliance_risk"] = min_max_normalize(df["alignment_score"], invert=True)
    return df


# ─────────────────────────────────────────────────────────────────────
# Composite risk scoring
# ─────────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = {
    "trade_dependence": 0.22,
    "commodity_concentration": 0.18,
    "diplomatic_intensity": 0.22,
    "sector_exposure": 0.18,
    "route_exposure": 0.08,
    "alliance_dynamics": 0.12,
}


def compute_composite_risk(
    data: Dict[str, pd.DataFrame],
    weights: Dict[str, float] = None,
) -> pd.DataFrame:
    """
    Compute the composite coercion risk score for each bilateral pair.
    Returns a dataframe with pair-level risk scores and all component scores.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    
    # Normalize weights to sum to 1
    total_w = sum(weights.values())
    if total_w == 0:
        # Equal weights if all zero
        weights = {k: 1.0 / len(weights) for k in weights}
    else:
        weights = {k: v / total_w for k, v in weights.items()}
    
    # Score each dimension
    trade_scores = score_trade_dependence(data["trade"])
    commodity_scores = score_commodity_concentration(data["commodity"])
    diplomatic_scores = score_diplomatic_intensity(data["diplomatic"])
    sector_scores = score_sector_exposure(data["sector"], data["trade"])
    route_scores = score_route_exposure(data["route"])
    alliance_scores = score_alliance_dynamics(data["alliance"])
    
    # Merge all on reporter-partner
    base = trade_scores[["reporter", "partner", "trade_dep_score", "trade_dependence",
                          "dependence_asymmetry", "import_dependence"]].copy()
    
    base = base.merge(
        commodity_scores[["reporter", "partner", "commodity_score", "commodity_hhi",
                          "top_sector", "top_sector_share"]],
        on=["reporter", "partner"], how="left"
    )
    base = base.merge(
        diplomatic_scores[["reporter", "partner", "diplomatic_score", "net_diplomatic_tone",
                            "conflict_score", "cooperation_score", "recent_trend_3m"]],
        on=["reporter", "partner"], how="left"
    )
    base = base.merge(
        sector_scores[["reporter", "partner", "sector_score", "n_critical_sectors", "critical_sectors"]],
        on=["reporter", "partner"], how="left"
    )
    
    # Route exposure: country-level, apply to reporter
    base = base.merge(
        route_scores[["country", "route_score", "n_chokepoint_dependencies"]].rename(
            columns={"country": "reporter"}
        ),
        on="reporter", how="left"
    )
    
    # Alliance dynamics: need to handle directionality
    # Create both directions from the undirected alliance data
    alliance_a = alliance_scores[["country_a", "country_b", "alliance_risk", "alignment_score", "alliance_type"]].rename(
        columns={"country_a": "reporter", "country_b": "partner"}
    )
    alliance_b = alliance_scores[["country_a", "country_b", "alliance_risk", "alignment_score", "alliance_type"]].rename(
        columns={"country_b": "reporter", "country_a": "partner"}
    )
    alliance_both = pd.concat([alliance_a, alliance_b], ignore_index=True)
    
    base = base.merge(
        alliance_both[["reporter", "partner", "alliance_risk", "alignment_score", "alliance_type"]],
        on=["reporter", "partner"], how="left"
    )
    
    # Fill any NaN scores
    for col in ["trade_dep_score", "commodity_score", "diplomatic_score",
                "sector_score", "route_score", "alliance_risk"]:
        base[col] = base[col].fillna(0.0)
    
    # Compute composite score
    base["composite_risk"] = (
        weights["trade_dependence"] * base["trade_dep_score"] +
        weights["commodity_concentration"] * base["commodity_score"] +
        weights["diplomatic_intensity"] * base["diplomatic_score"] +
        weights["sector_exposure"] * base["sector_score"] +
        weights["route_exposure"] * base["route_score"] +
        weights["alliance_dynamics"] * base["alliance_risk"]
    )
    
    # Apply sigmoid to create sharper risk separation
    base["composite_risk"] = base["composite_risk"].apply(
        lambda x: sigmoid_transform(x, midpoint=0.45, steepness=8)
    )
    
    # Normalize final score to 0-100
    base["risk_score_100"] = min_max_normalize(base["composite_risk"]) * 100
    
    # Risk tier
    base["risk_tier"] = pd.cut(
        base["risk_score_100"],
        bins=[0, 25, 50, 70, 85, 100],
        labels=["Low", "Moderate", "Elevated", "High", "Critical"],
        include_lowest=True,
    )
    
    # Add country names
    base["reporter_name"] = base["reporter"].map(lambda x: COUNTRIES.get(x, {}).get("name", x))
    base["partner_name"] = base["partner"].map(lambda x: COUNTRIES.get(x, {}).get("name", x))
    
    return base


# ─────────────────────────────────────────────────────────────────────
# Sector-level risk flags
# ─────────────────────────────────────────────────────────────────────
def compute_sector_risk_flags(
    data: Dict[str, pd.DataFrame],
    risk_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate sector-level risk flags.
    For each country x sector, flag if:
    - High import dependence from an adversarial supplier
    - Low substitutability
    - Supplier is in a hostile or competitive bloc
    """
    sector_df = data["sector"]
    alliance_df = data["alliance"]
    
    rows = []
    for _, row in sector_df.iterrows():
        country = row["country"]
        supplier = row["dominant_supplier"]
        
        if supplier == "domestic" or supplier not in COUNTRIES:
            flag_level = "Low"
            reason = "Domestic production or diversified supply"
        else:
            # Check alignment with supplier
            alignment_row = alliance_df[
                ((alliance_df["country_a"] == country) & (alliance_df["country_b"] == supplier)) |
                ((alliance_df["country_a"] == supplier) & (alliance_df["country_b"] == country))
            ]
            
            if len(alignment_row) > 0:
                alignment = alignment_row.iloc[0]["alignment_score"]
            else:
                alignment = 0.0
            
            # Risk logic
            risk_val = (
                row["import_dependence"] * 0.35 +
                row["supplier_concentration"] * 0.25 +
                (1 - row["substitutability"]) * 0.25 +
                max(0, -alignment) * 0.15
            )
            
            if risk_val > 0.55:
                flag_level = "Critical"
                reason = f"High dependency on {COUNTRIES.get(supplier, {}).get('name', supplier)} with low substitutability"
            elif risk_val > 0.40:
                flag_level = "High"
                reason = f"Significant exposure to {COUNTRIES.get(supplier, {}).get('name', supplier)}"
            elif risk_val > 0.25:
                flag_level = "Moderate"
                reason = f"Moderate dependency, some substitution possible"
            else:
                flag_level = "Low"
                reason = "Manageable exposure"
        
        rows.append({
            "country": country,
            "country_name": row["country_name"],
            "sector": row["sector"],
            "sector_label": row["sector_label"],
            "flag_level": flag_level,
            "import_dependence": row["import_dependence"],
            "supplier_concentration": row["supplier_concentration"],
            "substitutability": row["substitutability"],
            "dominant_supplier": supplier,
            "supplier_name": COUNTRIES.get(supplier, {}).get("name", supplier),
            "reason": reason,
        })
    
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Scenario narrative generator
# ─────────────────────────────────────────────────────────────────────
SCENARIO_TEMPLATES = {
    "trade_coercion": {
        "trigger": "Trade dependence + diplomatic deterioration",
        "template": (
            "**{partner_name}** imposes targeted restrictions on exports to **{reporter_name}** "
            "in the **{top_sector}** sector, where bilateral trade concentration is {concentration_pct:.0%}. "
            "{reporter_name}'s import dependence on {partner_name} stands at {import_dep:.1%} of total imports. "
            "The diplomatic tone has been {tone_desc} (net score: {tone:.2f}), with a {trend_desc} trend. "
            "Given the {alignment_desc} between {reporter_name} ({bloc_r}) and {partner_name} ({bloc_p}), "
            "retaliatory measures would {retaliation_desc}."
        ),
    },
    "supply_disruption": {
        "trigger": "Commodity concentration + strategic sector chokepoint",
        "template": (
            "A supply disruption scenario emerges as **{reporter_name}** relies on **{partner_name}** "
            "for {n_critical} critical sector(s): {critical_sectors}. "
            "With commodity HHI at {hhi:.3f} (above 0.25 threshold), the bilateral trade is "
            "dangerously concentrated. {partner_name} holds chokepoint control, and "
            "substitutability is limited. Current diplomatic tensions ({conflict_score:.1f}/10) "
            "increase the likelihood of politically motivated supply curtailment."
        ),
    },
    "sanctions_escalation": {
        "trigger": "Alliance divergence + rising conflict signals",
        "template": (
            "An escalation path exists between **{reporter_name}** and **{partner_name}**, "
            "currently characterized as {alliance_type}. "
            "Diplomatic conflict intensity is {conflict_score:.1f}/10, while cooperation "
            "measures only {coop_score:.1f}/10. The recent 3-month trend is {trend_dir}, "
            "suggesting {forecast}. "
            "{reporter_name}'s trade dependence ({dep_pct:.1%}) creates leverage for "
            "{partner_name} to apply economic pressure as part of broader strategic competition."
        ),
    },
    "route_vulnerability": {
        "trigger": "Maritime chokepoint exposure + hostile actor control",
        "template": (
            "**{reporter_name}** faces maritime route vulnerability with {n_chokepoints} "
            "chokepoint dependencies. Trade flows through {chokepoints_list}, "
            "any of which could be disrupted during escalation with **{partner_name}**. "
            "Combined with bilateral trade of ${trade_b:.1f}B and {alignment_desc}, "
            "a scenario of informal trade disruption through route interference is plausible."
        ),
    },
}


def generate_scenario_narrative(row: pd.Series, scenario_type: str) -> Dict:
    """Generate a detailed scenario narrative for a given pair and scenario type."""
    reporter = row["reporter"]
    partner = row["partner"]
    
    # Descriptive mappings
    tone_val = row.get("net_diplomatic_tone", 0)
    tone_desc = "hostile" if tone_val < -0.3 else "tense" if tone_val < 0 else "cautious" if tone_val < 0.3 else "cooperative"
    
    trend_val = row.get("recent_trend_3m", 0)
    trend_desc = "deteriorating" if trend_val < -0.05 else "stable" if abs(trend_val) < 0.05 else "improving"
    trend_dir = "negative" if trend_val < -0.05 else "flat" if abs(trend_val) < 0.05 else "positive"
    
    alignment_val = row.get("alignment_score", 0)
    alignment_desc = ("strategic adversarial posture" if alignment_val < -0.5
                      else "competitive relationship" if alignment_val < -0.1
                      else "limited alignment" if alignment_val < 0.3
                      else "partial alignment" if alignment_val < 0.6
                      else "strong alliance")
    
    retaliation_desc = ("likely be ineffective given asymmetric dependence" if row.get("dependence_asymmetry", 0) > 0.05
                        else "carry significant mutual cost" if abs(row.get("dependence_asymmetry", 0)) < 0.05
                        else "be a credible deterrent given mutual exposure")
    
    forecast = ("continued escalation absent diplomatic intervention" if trend_val < -0.05 and tone_val < 0
                else "a holding pattern with latent risk" if abs(trend_val) < 0.05
                else "possible de-escalation, though structural risks remain")
    
    bloc_r = COUNTRIES.get(reporter, {}).get("bloc", "Unknown")
    bloc_p = COUNTRIES.get(partner, {}).get("bloc", "Unknown")
    
    try:
        if scenario_type == "trade_coercion":
            narrative = SCENARIO_TEMPLATES[scenario_type]["template"].format(
                reporter_name=row.get("reporter_name", reporter),
                partner_name=row.get("partner_name", partner),
                top_sector=STRATEGIC_SECTORS.get(row.get("top_sector", ""), {}).get("label", row.get("top_sector", "N/A")),
                concentration_pct=row.get("top_sector_share", 0),
                import_dep=row.get("import_dependence", 0),
                tone_desc=tone_desc,
                tone=tone_val,
                trend_desc=trend_desc,
                alignment_desc=alignment_desc,
                bloc_r=bloc_r,
                bloc_p=bloc_p,
                retaliation_desc=retaliation_desc,
            )
        elif scenario_type == "supply_disruption":
            narrative = SCENARIO_TEMPLATES[scenario_type]["template"].format(
                reporter_name=row.get("reporter_name", reporter),
                partner_name=row.get("partner_name", partner),
                n_critical=row.get("n_critical_sectors", 0),
                critical_sectors=row.get("critical_sectors", "N/A"),
                hhi=row.get("commodity_hhi", 0),
                conflict_score=row.get("conflict_score", 0),
            )
        elif scenario_type == "sanctions_escalation":
            narrative = SCENARIO_TEMPLATES[scenario_type]["template"].format(
                reporter_name=row.get("reporter_name", reporter),
                partner_name=row.get("partner_name", partner),
                alliance_type=row.get("alliance_type", "undefined"),
                conflict_score=row.get("conflict_score", 0),
                coop_score=row.get("cooperation_score", 0),
                trend_dir=trend_dir,
                forecast=forecast,
                dep_pct=row.get("trade_dependence", 0),
            )
        elif scenario_type == "route_vulnerability":
            narrative = SCENARIO_TEMPLATES[scenario_type]["template"].format(
                reporter_name=row.get("reporter_name", reporter),
                partner_name=row.get("partner_name", partner),
                n_chokepoints=row.get("n_chokepoint_dependencies", 0),
                chokepoints_list="critical maritime corridors",
                trade_b=row.get("total_trade_b", 0) if "total_trade_b" in row.index else 0,
                alignment_desc=alignment_desc,
            )
        else:
            narrative = "Scenario type not recognized."
    except (KeyError, TypeError) as e:
        narrative = f"Narrative generation error: {str(e)}"
    
    return {
        "scenario_type": scenario_type,
        "trigger": SCENARIO_TEMPLATES.get(scenario_type, {}).get("trigger", "Unknown"),
        "narrative": narrative,
        "risk_score": row.get("risk_score_100", 0),
        "risk_tier": str(row.get("risk_tier", "Unknown")),
    }


def generate_all_scenarios(risk_df: pd.DataFrame, top_n: int = 20) -> List[Dict]:
    """Generate scenario narratives for highest-risk pairs."""
    top_pairs = risk_df.nlargest(top_n, "risk_score_100")
    scenarios = []
    
    for _, row in top_pairs.iterrows():
        # Determine most relevant scenario type
        scores = {
            "trade_coercion": row.get("trade_dep_score", 0) + row.get("commodity_score", 0),
            "supply_disruption": row.get("sector_score", 0) + row.get("commodity_score", 0),
            "sanctions_escalation": row.get("diplomatic_score", 0) + row.get("alliance_risk", 0),
            "route_vulnerability": row.get("route_score", 0),
        }
        primary = max(scores, key=scores.get)
        
        scenario = generate_scenario_narrative(row, primary)
        scenario["reporter"] = row["reporter"]
        scenario["partner"] = row["partner"]
        scenario["reporter_name"] = row.get("reporter_name", row["reporter"])
        scenario["partner_name"] = row.get("partner_name", row["partner"])
        scenarios.append(scenario)
    
    return scenarios
