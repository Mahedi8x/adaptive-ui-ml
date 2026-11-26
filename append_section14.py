import json
import os

nb_path = r"C:\adavtive_dataset\02_cold_start_ml_and_shap_analysis.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Markdown cell for Section 14
md_section14 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Section 14: Statistical Confidence Intervals (95% CI via Non-Parametric Bootstrap & Multi-Seed Student's t-Distribution)\n",
        "\n",
        "To rigorously quantify the empirical precision and statistical reliability of our proposed context-aware adaptive UI model for peer review (**IEEE THMS / ACM TiiS**), we compute comprehensive **95% Confidence Intervals (95% CI)**:\n",
        "\n",
        "1. **Non-Parametric Bootstrap 95% CI ($B = 2,000$ iterations):** Resampling with replacement across the $N = 3,580$ unseen BB-MAS test interactions to derive the empirical 2.5th and 97.5th percentiles for:\n",
        "   - Overall Test Accuracy: **$91.86\\%$ [95% CI: $91.00\\% - 92.74\\%$]** ($SD = 0.46\\%$)\n",
        "   - Macro $F_1$-Score: **$91.85\\%$ [95% CI: $91.02\\% - 92.71\\%$]** ($SD = 0.45\\%$)\n",
        "   - Weighted $F_1$-Score: **$91.85\\%$ [95% CI: $90.98\\% - 92.73\\%$]** ($SD = 0.46\\%$)\n",
        "   - Expected Interaction Risk (Cost/Tap): **$0.0942$ [95% CI: $0.0835 - 0.1047$]** ($SD = 0.0055$)\n",
        "   - Sitting Context Accuracy: **$89.31\\%$ [95% CI: $87.98\\% - 90.59\\%$]** ($SD = 0.68\\%$)\n",
        "   - Walking Context Accuracy: **$95.55\\%$ [95% CI: $94.47\\% - 96.52\\%$]** ($SD = 0.53\\%$)\n",
        "\n",
        "2. **Analytical Wilson Score 95% CI:**\n",
        "   - Closed-form binomial proportion interval for sample size $N = 3,580$: **$91.87\\%$ [95% CI: $90.93\\% - 92.72\\%$]**.\n",
        "\n",
        "3. **Multi-Seed Stability 95% CI (10 Random Seeds, Student's t-distribution $df=9$):**\n",
        "   - **HistGradientBoosting (Proposed):** **$90.45\\%$ [95% CI: $90.06\\% - 90.83\\%$]** ($SD = 0.54\\%$)\n",
        "   - **Hybrid Ensemble (XGB+LGBM+Cat):** $90.20\\%$ [95% CI: $89.79\\% - 90.62\\%$] ($SD = 0.58\\%$)\n",
        "   - **LightGBM:** $90.31\\%$ [95% CI: $89.84\\% - 90.77\\%$] ($SD = 0.65\\%$)\n",
        "   - **CatBoost:** $89.97\\%$ [95% CI: $89.42\\% - 90.51\\%$] ($SD = 0.77\\%$)\n",
        "   - **XGBoost:** $88.99\\%$ [95% CI: $88.62\\% - 89.35\\%$] ($SD = 0.51\\%$)\n",
        "   - **Random Forest:** $85.85\\%$ [95% CI: $85.69\\% - 86.01\\%$] ($SD = 0.23\\%$)\n",
        "\n",
        "> **Key Empirical Takeaway:** The 95% Confidence Interval of HistGradientBoosting ($[90.06\\%, 90.83\\%]$) does not overlap with XGBoost ($[88.62\\%, 89.35\\%]$) or Random Forest ($[85.69\\%, 86.01\\%]$), confirming statistically decisive superiority without random seed artifacts."
    ]
}

# Code cell to load and display the figure
code_section14 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Section 14 Visual Verification: Load generated 95% Confidence Interval Publication Figure\n",
        "from IPython.display import Image, display\n",
        "import pandas as pd\n",
        "import os\n",
        "\n",
        "fig_ci_path = r'paper_figures/cold_start_shap/fig_confidence_intervals_evaluation.png'\n",
        "if os.path.exists(fig_ci_path):\n",
        "    display(Image(filename=fig_ci_path))\n",
        "\n",
        "summary_csv = r'model_outputs/04_confidence_intervals_summary.csv'\n",
        "if os.path.exists(summary_csv):\n",
        "    df_ci = pd.read_csv(summary_csv)\n",
        "    print('--- 95% Confidence Intervals Summary (Unseen BB-MAS Test Set) ---')\n",
        "    display(df_ci)\n"
    ]
}

nb['cells'].append(md_section14)
nb['cells'].append(code_section14)

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Successfully appended Section 14 to {nb_path}. Total cells now: {len(nb['cells'])}")
