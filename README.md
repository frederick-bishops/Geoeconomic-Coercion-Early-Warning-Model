# Geoeconomic Coercion Early-Warning Model

A forecasting system for trade-related coercion, identifying when bilateral trade relationships drift toward weaponization, informal retaliation, export restrictions, or supply-chain pressure.

## System Scope

**Regional framing:** US / EU / China / Russia system-level competition, with countries like Ghana and Serbia as exposed mid- or smaller-state nodes.

**26 countries** spanning major powers, EU members, mid-powers, and exposed smaller states.

## Risk Dimensions

The model combines six dimensions into a composite risk score (0-100):

1. **Trade Dependence** — Asymmetric bilateral trade exposure (who needs whom more)
2. **Commodity Concentration** — HHI-based measure of how concentrated bilateral trade is in specific sectors
3. **Diplomatic Event Intensity** — Conflict vs. cooperation tone, modeled on GDELT event coding
4. **Strategic Sector Exposure** — Dependency on adversarial suppliers for critical inputs (semiconductors, rare earths, energy, etc.)
5. **Maritime Route Exposure** — Chokepoint dependencies (Malacca, Suez, Hormuz, etc.)
6. **Alliance / Bloc Dynamics** — Cross-bloc trade vulnerability (Western, China-aligned, Russia-aligned, Non-aligned, Swing)

## Features

- **Interactive Risk Heatmap** — Bilateral coercion risk across all pairs
- **Pair-Level Deep Dives** — Detailed breakdown of any bilateral relationship
- **Sector Risk Flags** — Country-by-sector vulnerability matrix
- **Scenario Narratives** — Auto-generated escalation paths for highest-risk pairs
- **Adjustable Weights** — Real-time recomputation with custom dimension weights
- **Data Export** — CSV and Excel downloads for all risk scores, flags, and scenarios

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this directory to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select the repository and set `app.py` as the main file
5. Deploy

## Data

The current version uses **calibrated synthetic data** modeled on:
- Real bilateral trade volumes (approximating UN Comtrade / WITS data)
- GDELT-style diplomatic event intensity patterns
- Known geopolitical alignments and alliance structures
- Strategic sector chokepoint analysis

### Integrating Live Data

The architecture supports plugging in real data sources:

- **UN Comtrade API** — Replace `generate_trade_data()` in `data_layer.py`
- **GDELT Event Database** — Replace `generate_diplomatic_intensity()`
- **World Bank Development Indicators** — Enhance GDP and macro data
- **ACLED** — Add conflict event data for route/regional risk

## File Structure

```
├── app.py              # Main Streamlit application
├── data_layer.py       # Data generation and loading
├── risk_engine.py      # Composite risk scoring and scenario generation
├── requirements.txt    # Python dependencies
├── .streamlit/
│   └── config.toml     # Streamlit theme and server config
└── README.md           # This file
```

## Methodology

### Composite Score Calculation

1. Each dimension is scored independently (0-1 scale) using min-max normalization
2. Dimension scores are weighted (user-adjustable) and summed
3. A sigmoid transformation is applied for sharper risk separation
4. Final score is normalized to 0-100 and assigned a risk tier

### Risk Tiers

| Score Range | Tier | Interpretation |
|---|---|---|
| 85-100 | Critical | Imminent coercion risk; active monitoring required |
| 70-85 | High | Elevated structural vulnerability; contingency planning needed |
| 50-70 | Elevated | Notable exposure; trend monitoring advised |
| 25-50 | Moderate | Manageable risk; periodic review |
| 0-25 | Low | Minimal coercion vulnerability |

### Scenario Types

- **Trade Coercion** — Targeted tariffs or export restrictions leveraging dependence asymmetry
- **Supply Disruption** — Chokepoint control over critical sector inputs
- **Sanctions Escalation** — Diplomatic deterioration leading to economic pressure
- **Route Vulnerability** — Maritime chokepoint disruption during bilateral tension
