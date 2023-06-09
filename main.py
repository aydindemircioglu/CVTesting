import itertools
from joblib import parallel_backend, Parallel, delayed, load, dump
import os
from scipy.stats import ttest_rel
import time

from select_shuffle_test import *
from nested_cross_validation import *

from loadData import *
from parameters import *


#    wie CV: alle parameter gehen einmal durch
def getExperiments (experimentList, expParameters, sKey, inject = None):
    newList = []
    for exp in experimentList:
        for cmb in list(itertools.product(*expParameters.values())):
            pcmb = dict(zip(expParameters.keys(), cmb))
            if inject is not None:
                pcmb.update(inject)
            _exp = exp.copy()
            _exp.append((sKey, pcmb))
            newList.append(_exp)
    experimentList = newList.copy()
    return experimentList



# this is pretty non-generic, maybe there is a better way, for now it works.
def generateAllExperiments (experimentParameters, verbose = False):
    experimentList = [ [] ]
    for k in experimentParameters.keys():
        if verbose == True:
            print ("Adding", k)
        if k == "FeatureSelection":
            # this is for each N too
            print ("Adding feature selection")
            newList = []
            for n in experimentParameters[k]["N"]:
                for m in experimentParameters[k]["Methods"]:
                    fmethod = experimentParameters[k]["Methods"][m].copy()
                    fmethod["nFeatures"] = [n]
                    newList.extend(getExperiments (experimentList, fmethod, m))
            experimentList = newList.copy()
        elif k == "Classification":
            newList = []
            for m in experimentParameters[k]["Methods"]:
                newList.extend(getExperiments (experimentList, experimentParameters[k]["Methods"][m], m))
            experimentList = newList.copy()
        else:
            experimentList = getExperiments (experimentList, experimentParameters[k], k)

    return experimentList



def createHyperParameters():
    # infos about hyperparameters
    print ("Have", len(fselParameters["FeatureSelection"]["Methods"]), "Feature Selection Methods.")
    print ("Have", len(clfParameters["Classification"]["Methods"]), "Classifiers.")

    # generate all experiments
    fselExperiments = generateAllExperiments (fselParameters)
    print ("Created", len(fselExperiments), "feature selection parameter settings")
    clfExperiments = generateAllExperiments (clfParameters)
    print ("Created", len(clfExperiments), "classifier parameter settings")
    print ("Total", len(clfExperiments)*len(fselExperiments), "experiments")

    # generate list of experiment combinations
    hyperParameters = []
    for fe in fselExperiments:
        for clf in clfExperiments:
            hyperParameters.append( (fe, clf))

    return hyperParameters



if __name__ == '__main__':
    # get hyperparamters
    hyperParameters = createHyperParameters()

    # iterate over datasets
    datasets = {}
    for d in dList:
        eval (d+"().info()")
        datasets[d] = eval (d+"().getData('./data/')")
        data = datasets[d]

        # test transform too
        y = data["Target"]
        X = data.drop(["Target"], axis = 1)
        X, y = preprocessData (X, y)


        # we apply R repeats -- serial
        # for r in range(nRepeats):
        #     select_shuffle_test (X, y, hyperParameters, kFold)
        # for r in range(nRepeats):
        #     nested_cross_validation (X, y, hyperParameters, kFold)
        #
        # execute parallel
        startSST = time.time()
        with parallel_backend("loky", inner_max_num_threads=1):
            fv_sst = Parallel (n_jobs = ncpus)(delayed(select_shuffle_test)(X, y, hyperParameters, kFold) for r in range(nRepeats))
        endSST = time.time()

        startNCV = time.time()
        with parallel_backend("loky", inner_max_num_threads=1):
            fv_ncv = Parallel (n_jobs = ncpus)(delayed(nested_cross_validation)(X, y, hyperParameters, kFold) for r in range(nRepeats))
        endNCV = time.time()

        # dump results
        os.makedirs("./results", exist_ok = True)
        results = {}
        results["SST"] = {"Time": endSST - startSST, "Acc": fv_sst}
        results["NCV"] = {"Time": endNCV - startNCV, "Acc": fv_ncv}
        dump(results, f"results/{d}.dump")


#
