import itertools
from joblib import parallel_backend, Parallel, delayed, load, dump
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.stats import ttest_rel
import seaborn as sns
import time

from parameters import *


if __name__ == '__main__':
    # dump results
    speedUp = []
    allResults = []
    for d in dList:
        results = load(f"results/{d}.dump")
        results["SST"].keys()
        try:
            fv_sst = np.array(results["SST"]["Acc"])
            fv_ncv = np.array(results["NCV"]["Acc"])
        except:
            fv_sst = np.array(results["SST"]["Acc:"])
            fv_ncv = np.array(results["NCV"]["Acc:"])
        time_sst = results["SST"]["Time"]
        time_ncv = results["NCV"]["Time"]
        print (f"SST: {np.mean(fv_sst):.2f} +/- {np.std(fv_sst):.2f}, time: {time_sst:.1f}")
        print (f"NCV: {np.mean(fv_ncv):.2f} +/- {np.std(fv_ncv):.2f}, time: {time_ncv:.1f}")
        print (f"Diff: {np.mean(fv_sst-fv_ncv):.3f} +/- {np.std(fv_sst-fv_ncv):.2f}")
        speedUp.append(time_ncv/time_sst)

        # apply simple t-test, this is actually a superiority test with margin 0.
        # p < 0.05 means that H1=SST > NCV is 'true', so SST is superior in terms of accuracy.
        _, p = ttest_rel (fv_sst, fv_ncv, alternative = "greater")
        print ("p-Value", np.round(p,3))
        df = pd.DataFrame({"Difference": fv_sst-fv_ncv})
        df["Dataset"] = d
        allResults.append(df)

    print ("Mean speedup", np.mean(speedUp))
    data = pd.concat(allResults)

    # diff
    print ("Mean differences", np.mean(data["Difference"]))
    data = pd.concat(allResults)

    # plot a small figure
    plt.figure()
    sns.set(style="white", context="talk")
    f, axs  = plt.subplots(1, 1, figsize = (12,10)) #gridspec_kw={'width_ratios': [1,2]})
    g = sns.boxplot(y='Difference', x='Dataset', data=data)
    g.axhline(0.0, color = "r", lw = 2)
    plt.xticks(rotation=45, horizontalalignment='right')
    f.savefig("./results/Figure.png", dpi = 300, bbox_inches='tight')
    plt.close('all')



#
