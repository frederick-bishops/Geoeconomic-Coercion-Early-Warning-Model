"""
data_layer.py — Synthetic + calibrated data generation for the Geoeconomic Coercion
Early-Warning Model. Generates realistic bilateral trade, commodity, diplomatic,
infrastructure, and alliance data for a curated set of countries.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple

# ─────────────────────────────────────────────────────────────────────
# Country universe: major powers + exposed mid/small states
# ─────────────────────────────────────────────────────────────────────
COUNTRIES = {
    # Major powers
    "USA": {"name": "United States", "region": "North America", "bloc": "Western", "gdp_b": 27360, "lat": 38.9, "lon": -77.0},
    "CHN": {"name": "China", "region": "East Asia", "bloc": "China-aligned", "gdp_b": 17790, "lat": 39.9, "lon": 116.4},
    "DEU": {"name": "Germany", "region": "Europe", "bloc": "Western", "gdp_b": 4430, "lat": 52.5, "lon": 13.4},
    "GBR": {"name": "United Kingdom", "region": "Europe", "bloc": "Western", "gdp_b": 3340, "lat": 51.5, "lon": -0.1},
    "FRA": {"name": "France", "region": "Europe", "bloc": "Western", "gdp_b": 3050, "lat": 48.9, "lon": 2.3},
    "RUS": {"name": "Russia", "region": "Eurasia", "bloc": "Russia-aligned", "gdp_b": 1860, "lat": 55.8, "lon": 37.6},
    "JPN": {"name": "Japan", "region": "East Asia", "bloc": "Western", "gdp_b": 4210, "lat": 35.7, "lon": 139.7},
    "IND": {"name": "India", "region": "South Asia", "bloc": "Non-aligned", "gdp_b": 3730, "lat": 28.6, "lon": 77.2},
    "KOR": {"name": "South Korea", "region": "East Asia", "bloc": "Western", "gdp_b": 1710, "lat": 37.6, "lon": 127.0},
    "BRA": {"name": "Brazil", "region": "South America", "bloc": "Non-aligned", "gdp_b": 2170, "lat": -15.8, "lon": -47.9},
    # EU members
    "ITA": {"name": "Italy", "region": "Europe", "bloc": "Western", "gdp_b": 2190, "lat": 41.9, "lon": 12.5},
    "NLD": {"name": "Netherlands", "region": "Europe", "bloc": "Western", "gdp_b": 1090, "lat": 52.4, "lon": 4.9},
    "POL": {"name": "Poland", "region": "Europe", "bloc": "Western", "gdp_b": 842, "lat": 52.2, "lon": 21.0},
    # Mid-power / exposed states
    "TUR": {"name": "Turkey", "region": "Middle East", "bloc": "Swing", "gdp_b": 1110, "lat": 39.9, "lon": 32.9},
    "AUS": {"name": "Australia", "region": "Oceania", "bloc": "Western", "gdp_b": 1720, "lat": -35.3, "lon": 149.1},
    "SAU": {"name": "Saudi Arabia", "region": "Middle East", "bloc": "Swing", "gdp_b": 1060, "lat": 24.7, "lon": 46.7},
    "ZAF": {"name": "South Africa", "region": "Africa", "bloc": "Non-aligned", "gdp_b": 377, "lat": -25.7, "lon": 28.2},
    "TWN": {"name": "Taiwan", "region": "East Asia", "bloc": "Western", "gdp_b": 790, "lat": 25.0, "lon": 121.5},
    "SGP": {"name": "Singapore", "region": "Southeast Asia", "bloc": "Swing", "gdp_b": 497, "lat": 1.3, "lon": 103.8},
    "VNM": {"name": "Vietnam", "region": "Southeast Asia", "bloc": "Swing", "gdp_b": 449, "lat": 21.0, "lon": 105.8},
    # Key small / exposed states
    "GHA": {"name": "Ghana", "region": "Africa", "bloc": "Non-aligned", "gdp_b": 76, "lat": 5.6, "lon": -0.2},
    "SRB": {"name": "Serbia", "region": "Europe", "bloc": "Swing", "gdp_b": 75, "lat": 44.8, "lon": 20.5},
    "LTU": {"name": "Lithuania", "region": "Europe", "bloc": "Western", "gdp_b": 78, "lat": 54.7, "lon": 25.3},
    "GEO": {"name": "Georgia", "region": "Eurasia", "bloc": "Swing", "gdp_b": 28, "lat": 41.7, "lon": 44.8},
    "MNG": {"name": "Mongolia", "region": "East Asia", "bloc": "Non-aligned", "gdp_b": 19, "lat": 47.9, "lon": 106.9},
    "PHL": {"name": "Philippines", "region": "Southeast Asia", "bloc": "Western", "gdp_b": 435, "lat": 14.6, "lon": 121.0},
}

# ─────────────────────────────────────────────────────────────────────
# Strategic sectors
# ─────────────────────────────────────────────────────────────────────
STRATEGIC_SECTORS = {
    "semiconductors": {"label": "Semiconductors & Advanced Chips", "chokepoint_holders": ["TWN", "KOR", "USA", "NLD", "JPN"]},
    "rare_earths": {"label": "Rare Earth Elements", "chokepoint_holders": ["CHN", "MNG"]},
    "energy_oil": {"label": "Oil & Gas", "chokepoint_holders": ["SAU", "RUS", "USA"]},
    "energy_lng": {"label": "LNG", "chokepoint_holders": ["USA", "AUS", "RUS"]},
    "critical_minerals": {"label": "Critical Minerals (Lithium, Cobalt)", "chokepoint_holders": ["AUS", "CHN", "BRA", "ZAF"]},
    "agriculture": {"label": "Grain & Agricultural Commodities", "chokepoint_holders": ["USA", "BRA", "RUS", "AUS"]},
    "pharma_api": {"label": "Pharmaceutical Ingredients (API)", "chokepoint_holders": ["CHN", "IND"]},
    "defense_tech": {"label": "Defense Technology & Dual-Use", "chokepoint_holders": ["USA", "GBR", "FRA", "RUS", "CHN"]},
    "telecom_infra": {"label": "Telecom Infrastructure (5G)", "chokepoint_holders": ["CHN", "KOR", "USA"]},
    "shipping_routes": {"label": "Shipping & Maritime Chokepoints", "chokepoint_holders": ["SGP", "TUR", "SAU"]},
}

# ─────────────────────────────────────────────────────────────────────
# Bloc alignment matrix (higher = more aligned, -1 to 1)
# ─────────────────────────────────────────────────────────────────────
BLOC_ALIGNMENT = {
    ("Western", "Western"): 0.85,
    ("Western", "China-aligned"): -0.55,
    ("Western", "Russia-aligned"): -0.70,
    ("Western", "Non-aligned"): 0.15,
    ("Western", "Swing"): 0.10,
    ("China-aligned", "China-aligned"): 0.90,
    ("China-aligned", "Russia-aligned"): 0.55,
    ("China-aligned", "Non-aligned"): 0.30,
    ("China-aligned", "Swing"): 0.20,
    ("Russia-aligned", "Russia-aligned"): 0.85,
    ("Russia-aligned", "Non-aligned"): 0.10,
    ("Russia-aligned", "Swing"): 0.05,
    ("Non-aligned", "Non-aligned"): 0.30,
    ("Non-aligned", "Swing"): 0.20,
    ("Swing", "Swing"): 0.15,
}

def get_bloc_alignment(bloc_a: str, bloc_b: str) -> float:
    """Return alignment score between two blocs (-1 to 1)."""
    key = (bloc_a, bloc_b)
    if key in BLOC_ALIGNMENT:
        return BLOC_ALIGNMENT[key]
    key_rev = (bloc_b, bloc_a)
    if key_rev in BLOC_ALIGNMENT:
        return BLOC_ALIGNMENT[key_rev]
    return 0.0


# ─────────────────────────────────────────────────────────────────────
# Calibrated bilateral trade data generator
# ─────────────────────────────────────────────────────────────────────
# Known major bilateral trade relationships (approx $B, bidirectional total)
KNOWN_TRADE_FLOWS = {
    ("USA", "CHN"): 575, ("USA", "DEU"): 260, ("USA", "JPN"): 230,
    ("USA", "GBR"): 150, ("USA", "KOR"): 190, ("USA", "IND"): 130,
    ("USA", "TWN"): 125, ("USA", "BRA"): 85, ("USA", "FRA"): 95,
    ("USA", "ITA"): 80, ("USA", "NLD"): 70, ("USA", "AUS"): 55,
    ("USA", "SAU"): 40, ("USA", "VNM"): 130, ("USA", "SGP"): 75,
    ("USA", "TUR"): 30, ("USA", "PHL"): 25,
    ("CHN", "JPN"): 320, ("CHN", "KOR"): 310, ("CHN", "DEU"): 240,
    ("CHN", "AUS"): 220, ("CHN", "TWN"): 260, ("CHN", "BRA"): 160,
    ("CHN", "RUS"): 190, ("CHN", "VNM"): 175, ("CHN", "IND"): 120,
    ("CHN", "SGP"): 130, ("CHN", "GBR"): 95, ("CHN", "NLD"): 85,
    ("CHN", "SAU"): 80, ("CHN", "TUR"): 45, ("CHN", "PHL"): 60,
    ("CHN", "GHA"): 10, ("CHN", "SRB"): 5, ("CHN", "ZAF"): 40,
    ("CHN", "MNG"): 10,
    ("DEU", "FRA"): 200, ("DEU", "NLD"): 230, ("DEU", "GBR"): 140,
    ("DEU", "ITA"): 170, ("DEU", "POL"): 160, ("DEU", "RUS"): 25,
    ("DEU", "TUR"): 50, ("DEU", "CHN"): 240,
    ("RUS", "IND"): 65, ("RUS", "TUR"): 60, ("RUS", "GEO"): 2.5,
    ("RUS", "SRB"): 4, ("RUS", "LTU"): 1.5, ("RUS", "MNG"): 3,
    ("GBR", "IND"): 35, ("GBR", "GHA"): 3,
    ("FRA", "ITA"): 95, ("FRA", "GBR"): 85, ("FRA", "NLD"): 55,
    ("JPN", "KOR"): 85, ("JPN", "TWN"): 75, ("JPN", "AUS"): 65,
    ("JPN", "IND"): 22, ("JPN", "VNM"): 45, ("JPN", "SGP"): 30,
    ("IND", "SAU"): 50, ("IND", "ZAF"): 15,
    ("AUS", "JPN"): 65, ("AUS", "KOR"): 45, ("AUS", "IND"): 30,
    ("SAU", "IND"): 50, ("SAU", "JPN"): 40, ("SAU", "KOR"): 35,
    ("SGP", "VNM"): 25, ("GHA", "IND"): 4, ("GHA", "ZAF"): 2,
    ("SRB", "DEU"): 7, ("SRB", "ITA"): 4, ("LTU", "DEU"): 6,
    ("LTU", "POL"): 8, ("GEO", "TUR"): 2, ("GEO", "CHN"): 1,
    ("MNG", "RUS"): 3, ("PHL", "JPN"): 22, ("PHL", "KOR"): 16,
}


def generate_trade_data(seed: int = 42) -> pd.DataFrame:
    """Generate bilateral trade dataframe with realistic values."""
    rng = np.random.default_rng(seed)
    codes = list(COUNTRIES.keys())
    rows = []
    
    for i, src in enumerate(codes):
        for j, dst in enumerate(codes):
            if i == j:
                continue
            key = (src, dst)
            key_rev = (dst, src)
            
            if key in KNOWN_TRADE_FLOWS:
                total = KNOWN_TRADE_FLOWS[key]
            elif key_rev in KNOWN_TRADE_FLOWS:
                total = KNOWN_TRADE_FLOWS[key_rev]
            else:
                # Gravity model approximation
                gdp_src = COUNTRIES[src]["gdp_b"]
                gdp_dst = COUNTRIES[dst]["gdp_b"]
                alignment = get_bloc_alignment(COUNTRIES[src]["bloc"], COUNTRIES[dst]["bloc"])
                base = (gdp_src * gdp_dst) ** 0.4 * 0.005
                base *= (1 + alignment * 0.3)
                total = max(0.1, base * rng.lognormal(0, 0.3))
            
            # Split into exports/imports with some asymmetry
            gdp_ratio = COUNTRIES[src]["gdp_b"] / (COUNTRIES[src]["gdp_b"] + COUNTRIES[dst]["gdp_b"])
            export_share = np.clip(gdp_ratio + rng.normal(0, 0.1), 0.2, 0.8)
            exports = total * export_share
            imports = total * (1 - export_share)
            
            rows.append({
                "reporter": src,
                "partner": dst,
                "reporter_name": COUNTRIES[src]["name"],
                "partner_name": COUNTRIES[dst]["name"],
                "exports_b": round(exports, 2),
                "imports_b": round(imports, 2),
                "total_trade_b": round(total, 2),
            })
    
    return pd.DataFrame(rows)


def compute_trade_dependence(trade_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute trade dependence metrics for each bilateral pair.
    Trade dependence = bilateral trade / reporter's total trade
    Export dependence = exports to partner / total exports
    Import dependence = imports from partner / total imports
    """
    # Total trade by reporter
    reporter_totals = trade_df.groupby("reporter").agg(
        total_exports=("exports_b", "sum"),
        total_imports=("imports_b", "sum"),
        total_trade=("total_trade_b", "sum"),
    ).reset_index()
    
    df = trade_df.merge(reporter_totals, on="reporter", suffixes=("", "_all"))
    df["trade_dependence"] = df["total_trade_b"] / df["total_trade"]
    df["export_dependence"] = df["exports_b"] / df["total_exports"]
    df["import_dependence"] = df["imports_b"] / df["total_imports"]
    
    return df


# ─────────────────────────────────────────────────────────────────────
# Commodity concentration (HHI-like index per bilateral pair)
# ─────────────────────────────────────────────────────────────────────
def generate_commodity_concentration(seed: int = 42) -> pd.DataFrame:
    """
    Generate commodity concentration data.
    HHI for imports: if a country imports mainly one commodity from a partner,
    that pair has high concentration risk.
    """
    rng = np.random.default_rng(seed)
    codes = list(COUNTRIES.keys())
    sectors = list(STRATEGIC_SECTORS.keys())
    rows = []
    
    for src in codes:
        for dst in codes:
            if src == dst:
                continue
            
            # Generate sector shares (Dirichlet distribution)
            # Some pairs have naturally concentrated trade
            concentration_alpha = rng.uniform(0.3, 2.0)
            shares = rng.dirichlet([concentration_alpha] * len(sectors))
            
            # Boost shares for sectors where dst is a chokepoint holder
            for k, sec in enumerate(sectors):
                if dst in STRATEGIC_SECTORS[sec]["chokepoint_holders"]:
                    shares[k] *= rng.uniform(2.0, 5.0)
            shares = shares / shares.sum()
            
            hhi = float(np.sum(shares ** 2))
            top_sector = sectors[np.argmax(shares)]
            top_share = float(np.max(shares))
            
            rows.append({
                "reporter": src,
                "partner": dst,
                "commodity_hhi": round(hhi, 4),
                "top_sector": top_sector,
                "top_sector_share": round(top_share, 4),
                "n_sectors_above_10pct": int(np.sum(shares > 0.10)),
            })
    
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Diplomatic event intensity (simulated GDELT-style data)
# ─────────────────────────────────────────────────────────────────────
DIPLOMATIC_EVENT_BASELINES = {
    ("USA", "CHN"): {"cooperation": 3.0, "conflict": 7.5},
    ("USA", "RUS"): {"cooperation": 1.5, "conflict": 8.0},
    ("CHN", "TWN"): {"cooperation": 1.0, "conflict": 9.0},
    ("CHN", "AUS"): {"cooperation": 3.0, "conflict": 6.0},
    ("CHN", "LTU"): {"cooperation": 0.5, "conflict": 7.0},
    ("RUS", "GEO"): {"cooperation": 0.5, "conflict": 8.5},
    ("RUS", "LTU"): {"cooperation": 0.3, "conflict": 7.5},
    ("RUS", "POL"): {"cooperation": 1.0, "conflict": 6.5},
    ("CHN", "PHL"): {"cooperation": 2.5, "conflict": 6.0},
    ("USA", "IND"): {"cooperation": 7.0, "conflict": 2.0},
    ("CHN", "SRB"): {"cooperation": 7.5, "conflict": 0.5},
    ("RUS", "SRB"): {"cooperation": 7.0, "conflict": 0.5},
}


def generate_diplomatic_intensity(seed: int = 42) -> pd.DataFrame:
    """Generate diplomatic event intensity scores (Goldstein-scale inspired)."""
    rng = np.random.default_rng(seed)
    codes = list(COUNTRIES.keys())
    rows = []
    
    for src in codes:
        for dst in codes:
            if src == dst:
                continue
            
            key = (src, dst)
            key_rev = (dst, src)
            
            if key in DIPLOMATIC_EVENT_BASELINES:
                base = DIPLOMATIC_EVENT_BASELINES[key]
            elif key_rev in DIPLOMATIC_EVENT_BASELINES:
                base = DIPLOMATIC_EVENT_BASELINES[key_rev]
            else:
                alignment = get_bloc_alignment(COUNTRIES[src]["bloc"], COUNTRIES[dst]["bloc"])
                base = {
                    "cooperation": max(0.5, 5 + alignment * 4 + rng.normal(0, 0.5)),
                    "conflict": max(0.5, 5 - alignment * 4 + rng.normal(0, 0.5)),
                }
            
            # Add noise
            coop = np.clip(base["cooperation"] + rng.normal(0, 0.5), 0, 10)
            conf = np.clip(base["conflict"] + rng.normal(0, 0.5), 0, 10)
            
            # Net diplomatic tone: higher = more cooperative, lower = more hostile
            net_tone = (coop - conf) / 10  # normalized to -1 to 1
            
            # Event volume (proxy for salience)
            volume = rng.poisson(max(1, int((coop + conf) * 3)))
            
            # Recent trend: are things getting worse? (negative = escalating)
            trend = rng.normal(0, 0.15)
            
            rows.append({
                "reporter": src,
                "partner": dst,
                "cooperation_score": round(coop, 2),
                "conflict_score": round(conf, 2),
                "net_diplomatic_tone": round(net_tone, 3),
                "event_volume": volume,
                "recent_trend_3m": round(trend, 3),
            })
    
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Strategic sector exposure
# ─────────────────────────────────────────────────────────────────────
def generate_sector_exposure(seed: int = 42) -> pd.DataFrame:
    """
    Generate sector-level exposure data per country.
    Shows how dependent each country is on imports of each strategic sector,
    and who the dominant suppliers are.
    """
    rng = np.random.default_rng(seed)
    codes = list(COUNTRIES.keys())
    rows = []
    
    for code in codes:
        for sec_key, sec_info in STRATEGIC_SECTORS.items():
            # Is this country a producer or a consumer?
            is_producer = code in sec_info["chokepoint_holders"]
            
            if is_producer:
                import_dependence = rng.uniform(0.0, 0.15)
                export_significance = rng.uniform(0.3, 0.8)
            else:
                import_dependence = rng.uniform(0.2, 0.95)
                export_significance = rng.uniform(0.0, 0.1)
            
            # Who is the dominant supplier?
            producers = sec_info["chokepoint_holders"]
            if code in producers:
                dominant_supplier = producers[1] if len(producers) > 1 else "domestic"
            else:
                dominant_supplier = rng.choice(producers)
            
            supplier_share = rng.uniform(0.3, 0.85) if not is_producer else rng.uniform(0.0, 0.2)
            
            # Substitutability score (0 = no substitutes, 1 = easily substitutable)
            substitutability = rng.uniform(0.1, 0.4) if sec_key in ["semiconductors", "rare_earths"] else rng.uniform(0.3, 0.8)
            
            rows.append({
                "country": code,
                "country_name": COUNTRIES[code]["name"],
                "sector": sec_key,
                "sector_label": sec_info["label"],
                "is_producer": is_producer,
                "import_dependence": round(import_dependence, 3),
                "export_significance": round(export_significance, 3),
                "dominant_supplier": dominant_supplier,
                "supplier_concentration": round(supplier_share, 3),
                "substitutability": round(substitutability, 3),
            })
    
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Port / route exposure
# ─────────────────────────────────────────────────────────────────────
CHOKEPOINTS = {
    "strait_of_malacca": {"label": "Strait of Malacca", "controller": "SGP", "lat": 1.4, "lon": 103.8,
                          "exposed": ["JPN", "KOR", "CHN", "TWN", "AUS", "IND", "VNM", "PHL"]},
    "suez_canal": {"label": "Suez Canal", "controller": "SAU", "lat": 30.0, "lon": 32.6,
                   "exposed": ["DEU", "GBR", "FRA", "ITA", "NLD", "POL", "IND", "CHN", "TUR"]},
    "strait_of_hormuz": {"label": "Strait of Hormuz", "controller": "SAU", "lat": 26.6, "lon": 56.3,
                         "exposed": ["JPN", "KOR", "IND", "CHN", "DEU", "FRA", "ITA"]},
    "turkish_straits": {"label": "Turkish Straits (Bosporus)", "controller": "TUR", "lat": 41.1, "lon": 29.0,
                        "exposed": ["RUS", "GEO", "SRB", "DEU", "ITA", "GBR", "FRA"]},
    "panama_canal": {"label": "Panama Canal", "controller": "USA", "lat": 9.1, "lon": -79.7,
                     "exposed": ["USA", "BRA", "CHN", "JPN", "KOR"]},
    "cape_of_good_hope": {"label": "Cape of Good Hope", "controller": "ZAF", "lat": -34.4, "lon": 18.5,
                          "exposed": ["GBR", "DEU", "FRA", "BRA", "IND", "GHA"]},
}


def generate_route_exposure(seed: int = 42) -> pd.DataFrame:
    """Generate maritime route exposure data."""
    rng = np.random.default_rng(seed)
    codes = list(COUNTRIES.keys())
    rows = []
    
    for code in codes:
        total_exposure = 0
        chokepoint_deps = []
        
        for cp_key, cp_info in CHOKEPOINTS.items():
            is_exposed = code in cp_info["exposed"]
            exposure = rng.uniform(0.4, 0.9) if is_exposed else rng.uniform(0.0, 0.1)
            total_exposure += exposure
            
            if is_exposed:
                chokepoint_deps.append(cp_info["label"])
        
        rows.append({
            "country": code,
            "country_name": COUNTRIES[code]["name"],
            "route_exposure_score": round(min(total_exposure / len(CHOKEPOINTS), 1.0), 3),
            "n_chokepoint_dependencies": len(chokepoint_deps),
            "critical_chokepoints": ", ".join(chokepoint_deps) if chokepoint_deps else "None",
        })
    
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Alliance / bloc dynamics
# ─────────────────────────────────────────────────────────────────────
def generate_alliance_data() -> pd.DataFrame:
    """Generate alliance alignment data for all pairs."""
    codes = list(COUNTRIES.keys())
    rows = []
    
    for i, src in enumerate(codes):
        for j, dst in enumerate(codes):
            if i >= j:
                continue
            
            alignment = get_bloc_alignment(COUNTRIES[src]["bloc"], COUNTRIES[dst]["bloc"])
            
            # Determine alliance type
            if alignment > 0.7:
                alliance_type = "Formal alliance"
            elif alignment > 0.3:
                alliance_type = "Strategic partnership"
            elif alignment > -0.2:
                alliance_type = "Neutral / limited"
            elif alignment > -0.5:
                alliance_type = "Competitive"
            else:
                alliance_type = "Adversarial"
            
            rows.append({
                "country_a": src,
                "country_b": dst,
                "country_a_name": COUNTRIES[src]["name"],
                "country_b_name": COUNTRIES[dst]["name"],
                "bloc_a": COUNTRIES[src]["bloc"],
                "bloc_b": COUNTRIES[dst]["bloc"],
                "alignment_score": round(alignment, 3),
                "alliance_type": alliance_type,
            })
    
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Master data loader
# ─────────────────────────────────────────────────────────────────────
def load_all_data(seed: int = 42) -> Dict[str, pd.DataFrame]:
    """Load or generate all datasets."""
    trade_raw = generate_trade_data(seed)
    trade = compute_trade_dependence(trade_raw)
    commodity = generate_commodity_concentration(seed)
    diplomatic = generate_diplomatic_intensity(seed)
    sector = generate_sector_exposure(seed)
    route = generate_route_exposure(seed)
    alliance = generate_alliance_data()
    
    return {
        "trade": trade,
        "commodity": commodity,
        "diplomatic": diplomatic,
        "sector": sector,
        "route": route,
        "alliance": alliance,
    }
