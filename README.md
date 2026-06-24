# MLB Shortstop Replacement Analysis

A data science project analyzing MLB shortstop performance across batting, defense, and baserunning dimensions to identify optimal replacement candidates. Built using Statcast and Baseball Savant data from the 2023–2025 seasons.

---

## Project Overview

The goal is to evaluate all active MLB shortstops across three performance pillars and synthesize a player similarity + risk scoring framework to recommend replacement targets. The pipeline goes from raw Statcast CSVs → merged dataset → per-domain analysis (correlation, PCA, clustering) → risk scoring → interactive web tool.

**Team:** Jangmin Song, Marvin, Gregory, Day Xu

---

## Repository Structure

```
├── data/
│   ├── combined-SS-all_years.csv          # Combined shortstop dataset (all years)
│   ├── combined-SS-all_years_UPDATED.xlsx # Updated dataset with engineered columns
│   ├── dataset_converge.py                # Merges raw per-category CSVs into one file
│   └── whole_data.py                      # Data validation / exploration script
│
├── analysis/
│   ├── batted_ball/
│   │   ├── bb_correlations.py             # Pearson correlation + heatmap
│   │   ├── bb_pca.py                      # PCA on batted ball variables
│   │   └── bb_clustering.py              # K-Means clustering
│   │
│   ├── defense/
│   │   ├── defense_correlations.py        # Correlation analysis on defensive metrics
│   │   ├── defense_pca.py                 # PCA on defensive variables
│   │   ├── defense_clustering.py         # K-Means clustering
│   │   └── defense_recommend.py          # Variable recommendation (redundancy/independence)
│   │
│   ├── running/
│   │   ├── running_correlations.py        # Correlation: hp_to_1b vs sprint_speed
│   │   ├── running_pca.py                 # PCA on speed variables
│   │   ├── running_clustering.py         # K-Means clustering
│   │   └── running_recommendation.py     # Variable recommendation
│   │
│   ├── clustering/
│   │   ├── k_medoids.py                   # Custom K-Medoids implementation
│   │   ├── spectral_embedding.py          # Spectral embedding + clustering
│   │   └── marvin_correlation.py          # Correlation analysis (Marvin)
│   │
│   └── risk/
│       └── risk.py                        # Age + injury + contract risk scoring model
│
├── outputs/
│   ├── batted_ball/                       # Correlation, PCA, and clustering plots
│   ├── defense/                           # Correlation, PCA, clustering, recommendation plots
│   └── running/                           # Correlation, PCA, clustering, recommendation plots
│
└── website/
    ├── index.html                         # Final interactive player comparison tool
    ├── app.py                             # Streamlit version (spectral embedding search)
    ├── tablebuilder.py                    # Builds pairwise recommendation table
    └── data/
        ├── pairwise_recommendation_table.csv
        └── player_year_xy_risk.csv
```

---

## Methodology

### 1. Data Collection & Merging
Raw data was pulled from Baseball Savant across 13 stat categories (bat tracking, batted ball, exit velocity, sprint speed, fielding run value, etc.) for all shortstops from 2023–2025. `dataset_converge.py` merges these on `(player_id, year)`.

### 2. Per-Domain Analysis
Each domain follows the same pipeline:

| Step | Script | Description |
|------|--------|-------------|
| Correlation | `*_correlations.py` | Pearson correlation matrix + pairwise bar charts |
| PCA | `*_pca.py` | Explained variance, PC loadings, and biplot |
| Clustering | `*_clustering.py` | K-Means with silhouette scoring to find optimal k |
| Recommendation | `*_recommend.py` | Variable selection based on independence + low redundancy |

### 3. Alternative Clustering Methods
- **Spectral Embedding** (`spectral_embedding.py`): Nonlinear dimensionality reduction followed by K-Means in the embedded space.
- **K-Medoids** (`k_medoids.py`): Custom implementation using pairwise distances — more robust to outliers than K-Means.
- **Gaussian Mixture Models** and **Dunn Index** were tested in exploratory notebooks.

### 4. Risk Scoring
`risk.py` builds a composite risk score (0–1) for each player-season combining:
- **Age risk** (40%) — based on age relative to typical SS decline curve
- **Injury risk** (40%) — weighted from actual missed days, expected injury days, frequency, and severity
- **Contract risk** (20%) — AAV and years remaining, discounted by WAR

### 5. Player Similarity & Website
The final web tool uses spectral embedding coordinates + risk scores to rank replacement candidates pairwise. `tablebuilder.py` builds the recommendation matrix as:

```
rec_score = 0.5 × similarity + 0.5 × (1 − risk_of_candidate)
```

---

## Key Output Visualizations

### Batted Ball Profile — Clustering
K-Means clustering of shortstops by batted ball profile (gb_rate, fb_rate, ld_rate, pull_rate, etc.)

| Silhouette | Cluster Scatter | Cluster Means |
|---|---|---|
| ![Silhouette](outputs/batted_ball/bb_clust_1_silhouette.png) | ![Scatter](outputs/batted_ball/bb_clust_2_scatter.png) | ![Means](outputs/batted_ball/bb_clust_3_means.png) |

### Batted Ball Profile — Correlation
![Correlation Heatmap](outputs/batted_ball/bb_corr_1_heatmap.png)
![Pairwise Bars](outputs/batted_ball/bb_corr_2_pairwise_bars.png)

### Batted Ball Profile — PCA
| Explained Variance | Loadings | Biplot |
|---|---|---|
| ![EV](outputs/batted_ball/bb_pca_1_explained_variance.png) | ![Loadings](outputs/batted_ball/bb_pca_2_loadings.png) | ![Biplot](outputs/batted_ball/bb_pca_3_biplot.png) |

---

### Defensive Performance — Clustering
K-Means clustering on total_runs, range_runs, arm_runs, dp_runs, OAA, arm_ss

| Silhouette | Cluster Scatter | Cluster Means |
|---|---|---|
| ![Silhouette](outputs/defense/defensive_clustering_silhouette.png) | ![Scatter](outputs/defense/defensive_clustering_scatter.png) | ![Means](outputs/defense/defensive_clustering_means.png) |

### Defensive Performance — Correlation
![Correlation Heatmap](outputs/defense/def_corr_1_heatmap.png)
![Pairwise Bars](outputs/defense/def_corr_2_pairwise_bars.png)

### Defensive Performance — Variable Recommendation
![Independence](outputs/defense/defense_recommendation_independence.png)
![Redundancy](outputs/defense/defense_recommendation_redundancy_heatmap.png)

---

### Running Ability — Clustering
K-Means clustering on sprint speed and home-to-first time

| Silhouette | Cluster Scatter | Cluster Means |
|---|---|---|
| ![Silhouette](outputs/running/spd_clust_1_silhouette.png) | ![Scatter](outputs/running/spd_clust_2_scatter.png) | ![Means](outputs/running/spd_clust_3_means.png) |

### Running Ability — Correlation
![Correlation Scatter](outputs/running/1_running_correlation_scatter.png)
![Distributions](outputs/running/2_running_correlation_istributions.png)

### Running Ability — Variable Recommendation
![Shared Variance](outputs/running/1_running_recommendation_shared_variance.png)
![Percentiles](outputs/running/2_running_recommendation_percentiles.png)

---

## Interactive Web Tool

![Website Preview](outputs/website_preview.png)

`website/index.html` — open directly in a browser (no server required).

The tool lets you select any shortstop and see their top replacement candidates ranked by spectral-embedding similarity and adjusted for injury/contract risk. Built with vanilla HTML/CSS/JS.

A Streamlit version (`website/app.py`) provides the same functionality with a Python backend:

```bash
pip install streamlit scikit-learn pandas openpyxl
streamlit run website/app.py
```

---

## Data Sources

All statistics from [Baseball Savant](https://baseballsavant.mlb.com/) Statcast export (2023–2025 seasons). Contract and injury data manually compiled.

---

## Dependencies

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn openpyxl streamlit
```
