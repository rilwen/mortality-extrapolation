import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CASES = [("female", 30), ("male", 85)]
MAX_YEAR = 2061
LAST_KNOWN_YEAR = 2019
VERSION = 3
RESULTS_DIR = "ft_results%d_%d" % (LAST_KNOWN_YEAR, VERSION)
INPUT_SIZE = 40

def save_heatmap(df, filename, sex, age,vmin, vmax):
    ax = sns.heatmap(df, yticklabels=10, xticklabels=5, vmin=vmin, vmax=vmax, cmap="copper")
    ax.set_xlabel("historical mortality rates")
    ax.set_ylabel("extrapolated mortality rates")
    # ax.set_title("%ss of age %d, England and Wales" % (sex.title(), age))
    ax.get_figure().savefig(filename + ".eps")
    ax.get_figure().savefig(filename + ".pdf")
    ax.get_figure().savefig(filename + ".png")


if __name__ == "__main__":
    min_grad = 0
    max_grad = 0
    gradients = {}
    for sex, age in CASES:
        df = pd.read_csv("%s/apply_IS=%d/0/gradient-predicted-%d-ew-%s-mortality-period-qx-%d.csv" % (RESULTS_DIR, INPUT_SIZE, age, sex, LAST_KNOWN_YEAR),
                         index_col=0)
        df = df.loc[:MAX_YEAR]
        df.columns = df.columns.astype(str)
        save_heatmap(df, "%s/gradient-heatmap-log-log-%s-%d-%d" % (RESULTS_DIR, sex, age, LAST_KNOWN_YEAR), sex, age, None, None)
        plt.close()
        r = pd.read_csv("%s/merged_mortality_ew_%s_%d.csv" % (RESULTS_DIR, sex, LAST_KNOWN_YEAR), index_col=0, delimiter="\t")
        r.columns = r.columns.astype(int)
        r = r.T        
        r = r.T[age]        
        # r_i = exp(y_i)
        # dr_1 / dr_2 = d exp(y_1(ln(r_2))) / dr_2 = r1 * dy_1/dy_2 / r_2
        for extrap_year in df.index:
            df.loc[extrap_year] *= r[extrap_year]
        for historic_year in df.columns:
            df[historic_year] /= r[historic_year]
        gradients[(sex, age)] = df
        values = df.values.flatten()
        min_grad = min(np.amin(values), min_grad)
        max_grad = max(np.amax(values), max_grad)
        print("Sex=%s, Age=%d" % (sex, age))
        print("Gradient statistics: %s" % pd.Series(values).describe())
    for sex, age in CASES:
        df = gradients[(sex, age)]
        save_heatmap(df, "%s/gradient-heatmap-%s-%d-%d" % (RESULTS_DIR, sex, age, LAST_KNOWN_YEAR),
                     sex, age, min_grad, max_grad)
        plt.close()