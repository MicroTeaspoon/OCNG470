import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Import data
chl = pd.read_csv("chlorophyll_anova_revised.csv")

# Summary statistics
summary_data = (
    chl.groupby(["Region", "Season"])["Chlorophyll_mg_m3"]
    .agg(["count", "mean", "std"])
    .reset_index()
)

summary_data["stderr"] = (
    summary_data["std"] / summary_data["count"] ** 0.5
)

print(summary_data)

# Interaction plot with standard error bars
plt.figure(figsize=(9, 6))

for region, data in summary_data.groupby("Region"):
    plt.errorbar(
        data["Season"],
        data["mean"],
        yerr=data["stderr"],
        marker="o",
        capsize=4,
        linewidth=2,
        label=region
    )

plt.xlabel("Season")
plt.ylabel("Mean chlorophyll-a (mg m$^{-3}$)")
plt.title("Mean Chlorophyll Concentration by Region and Season")
plt.legend(title="Region")
plt.tight_layout()
plt.show()

# Two-way ANOVA with interaction
model = smf.ols(
    "Chlorophyll_mg_m3 ~ C(Region) * C(Season)",
    data=chl
).fit()

anova_results = anova_lm(model, typ=2)
print(anova_results)


# Tukey HSD comparisons
chl["Group"] = chl["Region"] + "_" + chl["Season"]

tukey = pairwise_tukeyhsd(
    endog=chl["Chlorophyll_mg_m3"],
    groups=chl["Group"],
    alpha=0.05
)

print(tukey)