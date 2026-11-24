# imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import statsmodels.api as sm
import statsmodels.formula.api as smf

from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd



# Import data
#chl = pd.read_csv("input_data_hw470_2026.csv") # generic
chl = pd.read_csv("/home/noahr/pythonProjects/OCNG470/ANOVA_HW/input_data_hw470_2026.csv") # laptop
#chl = pd.read_csv(r"C:\Users\noahr\Documents\Python Scripts\OCNG470\ANOVA_HW\input_data_hw470_2026.csv") # desktop

# print(chl.head())


# Summary statistics
summary_data = (
    chl.groupby(["Region", "Season"])["Chlorophyll_mg_m3"]
    .agg(["count", "mean", "std"])
    .reset_index()
)
# print(summary_data)


# Add standard error column
summary_data["stderr"] = (
    summary_data["std"] / summary_data["count"] ** 0.5
)
# print(summary_data)


# Boxplot of chlorophyll-a by season
plt.figure(figsize=(9, 6))

sns.boxplot(data=chl, x="Season", y="Chlorophyll_mg_m3")

plt.xlabel("Season")
plt.ylabel("Chlorophyll-a (mg m$^{-3}$)")
plt.title("Chlorophyll-a Concentration by Season")
plt.tight_layout()
plt.show()


# Boxplot of chlorophyll-a by region
plt.figure(figsize=(9, 6))

sns.boxplot(data=chl, x="Region", y="Chlorophyll_mg_m3")

plt.xlabel("Region")
plt.ylabel("Chlorophyll-a (mg m$^{-3}$)")
plt.title("Chlorophyll-a Concentration by Region")
plt.tight_layout()
plt.show()


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
plt.suptitle("Mean Chlorophyll Concentration by Region and Season")
plt.title("Error bars represent standard error of the mean", fontsize=8)
plt.legend(title="Region")
plt.tight_layout()
plt.show()


# Two-way ANOVA with interaction
model = smf.ols(
    "Chlorophyll_mg_m3 ~ C(Region) * C(Season)",
    data=chl
).fit()

anova_results = anova_lm(model, typ=1)
# print(anova_results)


# Tukey HSD comparisons
chl["Group"] = chl["Region"] + "_" + chl["Season"]

tukey = pairwise_tukeyhsd(
    endog=chl["Chlorophyll_mg_m3"],
    groups=chl["Group"],
    alpha=0.05
)
print(tukey)