import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dataset Description

    The data comes from the **National Health and Nutrition Examination Survey (NHANES)**, a public health survey conducted by the **U.S. Centers for Disease Control and Prevention (CDC)**.

    The training set contains **16,962 participants**. It can be found in the `hypertension_train.csv` file, which contains the following columns:

    | Column | Description |
    |---|---|
    | `id` | A unique identifier for each participant. |
    | `RIDAGEYR` | Age in years. |
    | `BMXBMI` | Body mass index, a general indicator of body fat. |
    | `LBXTC` | Total cholesterol (mg/dL), a fatty substance in the blood whose high levels are associated with cardiovascular risk. |
    | `LBDHDD` | HDL cholesterol (mg/dL), often called the “good” cholesterol. |
    | `LBXTR` | Triglycerides (mg/dL), another type of fat present in the blood. |
    | `LBXGLU` | Fasting glucose (mg/dL), the level of sugar in the blood measured after several hours without eating. |
    | `LBXSCR` | Creatinine (mg/dL), a waste product filtered by the kidneys, used as an indicator of kidney function. |
    | `LBXSBU` | Urea or BUN (mg/dL), another indicator of kidney function. |
    | `smoker` | `1` if the participant currently smokes cigarettes, `0` otherwise. |
    | `vigorous_pa` | `1` if the participant does vigorous physical activity (at work or in leisure) in a typical week, `0` otherwise. |
    | `moderate_pa` | `1` if the participant does moderate physical activity (at work, walking or cycling to get around, or in leisure) in a typical week, `0` otherwise. |
    | `label` | The target variable. It is `1` if the participant has hypertension and `0` otherwise. |

    In addition to the training set, there is a separate test set consisting of **5,655 participants** in the `hypertension_test_features.csv` file. The test set contains only the `id` column and the feature columns (no `label`).
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from pathlib import Path
    import seaborn as sns
    import matplotlib.pyplot as plt

    return Path, mo, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 1: **Config and read data**
    """)
    return


@app.cell
def _(Path):
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent / "data"
    return (data_dir,)


@app.cell
def _(data_dir, pd):
    df_train = pd.read_csv(data_dir / "hypertension_train.csv")
    df_test = pd.read_csv(data_dir / "hypertension_test_features.csv")
    return df_test, df_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 2: **Data Quality**
    2.1: Exploring NULL / missing values
    """)
    return


@app.cell(disabled=True, hide_code=True)
def _(df_test, df_train, pd):
    print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")
    print("Missing values in train:\n", df_train.isna().sum())

    print(f"Duplicate feature rows in train: {df_train.drop(columns=['id', 'label']).duplicated().sum()}")
    print("\n--- Missing Values ---")
    print(pd.DataFrame({
        "train_missing": df_train.isna().sum(),
        "test_missing": df_test.isna().sum()
    }))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.2 Check for duplicates
    """)
    return


@app.cell
def _():
    continuous_cols = [
        "RIDAGEYR", "BMXBMI", "LBXTC", "LBDHDD", 
        "LBXTR", "LBXGLU", "LBXSCR", "LBXSBU"
    ]
    binary_cols = ["smoker", "vigorous_pa", "moderate_pa"]
    feature_cols = continuous_cols + binary_cols
    return continuous_cols, feature_cols


@app.cell
def _(df_test, df_train):
    # --- CHECK 2: DUPLICATES & OVERLAP ---
    print("=== 2. DUPLICATE AUDIT ===")
    print("Duplicate IDs in train:", df_train["id"].duplicated().sum())
    print("Duplicate IDs in test: ", df_test["id"].duplicated().sum())
    print("Overlapping IDs (train vs test):", len(set(df_train["id"]).intersection(set(df_test["id"]))))
    return


@app.cell
def _(df_train, feature_cols):
    # Exact feature duplicates (ignoring id and label) 
    train_feat_dupes = df_train[df_train.duplicated(subset=feature_cols, keep=False)]

    print(f"Train rows with identical feature values: {len(train_feat_dupes)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.3 Checking for outliers & Skewness
    """)
    return


@app.cell
def _(continuous_cols, df_test, df_train):
    # --- CHECK 4: OUTLIERS, QUANTILES & SKEWNESS ---
    print("\n=== 4. CONTINUOUS DISTRIBUTIONS & EXTREMES (TRAIN) ===")
    audit_stats = df_train[continuous_cols].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    audit_stats["skew"] = df_train[continuous_cols].skew()
    print(audit_stats[["min", "1%", "5%", "50%", "95%", "99%", "max", "skew"]].round(2))

    print("\n=== CONTINUOUS EXTREMES (TEST) ===")
    test_stats = df_test[continuous_cols].describe(percentiles=[0.01, 0.5, 0.99]).T
    test_stats["skew"] = df_test[continuous_cols].skew()
    print(test_stats[["min", "1%", "50%", "99%", "max", "skew"]].round(2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 3: **Visual Exploratory Data Analysis**
    #### 3.1 Target Class Balance (`label`)
    Before looking at features, check how balanced the target variable (`0` = No Hypertension, `1` = Hypertension) is in the training set.
    """)
    return


@app.cell
def _(df_train, plt, sns):
    fig_target, ax_target = plt.subplots(figsize=(6, 4))

    sns.countplot(data=df_train, x="label", ax=ax_target)
    ax_target.set_title("Target Distribution (0 = No Hypertension, 1 = Hypertension)")
    ax_target.set_xlabel("Hypertension (label)")
    ax_target.set_ylabel("Number of Participants")

    # Add percentage labels on top of each bar
    total = len(df_train)
    for p in ax_target.patches:
        count = int(p.get_height())
        pct = 100 * count / total
        ax_target.annotate(
            f"{count:,} ({pct:.1f}%)",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center", va="bottom", xytext=(0, 3), textcoords="offset points"
        )

    ax_target.margins(y=0.15)
    plt.tight_layout()
    fig_target
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.1 Interpretation

    - 74.5% of patients don't have hypertension

    **Implications**
    - 74.5% is our benchmark for scoring i.e. a model that guesses 0 everytime will have 74.5% accuracy
    - Tells us we should use stratified cross-validation, metric like AUC, F1, PR-AUC instead of raw accuracy
    - Use class_weight="balanced" or scale_pos_weight to penalise missing a hypertension case more heavily
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 3.2 Distribution of Continuous Features
    Look at the shape, spread, and skewness of each of the 8 continuous variables across all participants in the training set.
    """)
    return


@app.cell
def _(continuous_cols, df_train, plt, sns):
    fig_dist, axes_dist = plt.subplots(2, 4, figsize=(16, 7))
    axes_dist = axes_dist.flatten()

    for i, col in enumerate(continuous_cols):
        sns.histplot(df_train[col].dropna(), bins=35, kde=True, ax=axes_dist[i])
        skew_val = df_train[col].skew()
        axes_dist[i].set_title(f"{col} (skew: {skew_val:.2f})")
        axes_dist[i].set_xlabel("")

    plt.tight_layout()
    fig_dist
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.2 Interpretation
    - RIDAGEYR: Data is uniform from 20-79 with a massive spike at 80 (NHANES lumps all patients over 80 together..)
    - BMI & Cholestoral (BMXBMI, LBXTC, LBDHDD): are moderately right skewed (Most patients are normal but slightly skew toward higher values)
    - Metabolic & Kidney Biomarkers (LBXTR, LBXGLU, LBXSCR, LBXSBU): Extreme right skew, suggesting some patients have extremely high biomarker values (perhaps due to issues like ckd / renal failure)

    **Implications:**
    - Do not blindly drop outliers as these may represent real, sick patients
    - For Logistic Regression, SVM, KNN, large skewed values may dominate so must apply log transformation to handle thisand some scaler to normally distribute tails
    - Tree-based models handle this naturally
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 3.3 Features vs. Target (`label = 0` vs `label = 1`)
    Compare how each continuous and binary feature differs between participants without hypertension (`0`) and with hypertension (`1`).
    *(Note: x-axes on continuous plots are zoomed to the 1st–99th percentile so extreme outliers don't squish the curves).*
    """)
    return


@app.cell
def _(continuous_cols, df_train, plt, sns):
    def _():
        fig_vs, axes_vs = plt.subplots(2, 4, figsize=(16, 7))
        axes_vs = axes_vs.flatten()

        for i, col in enumerate(continuous_cols):
            sns.kdeplot(
                data=df_train, x=col, hue="label",
                common_norm=False, fill=True, alpha=0.3, ax=axes_vs[i]
            )
            low, high = df_train[col].quantile([0.01, 0.99])
            axes_vs[i].set_xlim(low, high)
            axes_vs[i].set_title(f"{col} by Target")
            axes_vs[i].set_xlabel("")

        plt.tight_layout()
        return fig_vs


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.3 Interpretation
    - Suggests age is the dominanat predictor of having hypertension
    """)
    return


@app.cell
def _(df_train, plt, sns):
    def _():
        fig_bin, axes_bin = plt.subplots(1, 3, figsize=(13, 4))

        for i, col in enumerate(["smoker", "vigorous_pa", "moderate_pa"]):
            sns.barplot(
                data=df_train, x=col, y="label",
                estimator=lambda y: y.mean() * 100,
                ax=axes_bin[i], capsize=0.1
            )
            axes_bin[i].axhline(df_train["label"].mean() * 100, color="red", linestyle="--", label="Avg (25.5%)")
            axes_bin[i].set_title(f"Hypertension Rate (%) by {col}")
            axes_bin[i].set_ylabel("Hypertension Rate (%)")
            axes_bin[i].legend()

        plt.tight_layout()
        return fig_bin


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation
    - Smoking increases hypertension by 7% rate
    - Vigorous physical activity appears to significantly reduce hypertension rate, ~18% hypertension rate for those who do vigorous_pa vs. ~30% for those who don't (12% drop)
    - Moderate physical activity also has an effect reducing hypertension rate by 6.6% compared with no physical activity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 3.4 Correlation Matrix Heatmap
    Check how strongly each feature correlates with `label` and whether any features are strongly correlated with each other (Spearman rank correlation is used so skewed outliers don't distort the numbers).
    """)
    return


@app.cell
def _(df_train, feature_cols, plt, sns):
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    corr_matrix = df_train[feature_cols + ["label"]].corr(method="spearman")

    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-0.6, vmax=0.6, ax=ax_corr
    )
    ax_corr.set_title("Spearman Correlation Matrix (Features & Target)")
    plt.tight_layout()
    fig_corr
    return


if __name__ == "__main__":
    app.run()
