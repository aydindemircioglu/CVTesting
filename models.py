#
import cv2
from functools import partial
import numpy as np

from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest, SelectFromModel

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ITMO_FS.filters.univariate import anova


def bhattacharyya_score_fct (X, y):
    yn = y/np.sum(y)
    yn = np.asarray(yn, dtype = np.float32)
    scores = [0]*X.shape[1]
    for j in range(X.shape[1]):
        xn = (X[:,j] - np.min(X[:,j]))/(np.max(X[:,j] - np.min(X[:,j])))
        xn = xn/np.sum(xn)
        xn = np.asarray(xn, dtype = np.float32)
        scores[j] = cv2.compareHist(xn, yn, cv2.HISTCMP_BHATTACHARYYA)

    scores = np.asarray(scores, dtype = np.float32)
    return -scores



def train_model(paramSet, X_train, y_train):
    p_FSel, p_Clf = paramSet
    fselector = createFSel (p_FSel)
    classifier = createClf (p_Clf)

    # apply both
    with np.errstate(divide='ignore',invalid='ignore'):
        fselector.fit (X_train.copy(), y_train.copy())
        X_fs_train = fselector.transform (X_train)
        y_fs_train = y_train

        classifier.fit (X_fs_train, y_fs_train)

    return [fselector, classifier]



def test_model(trained_model, X_test, y_test):
    fselector, classifier = trained_model
    # apply model
    X_fs_test = fselector.transform (X_test)
    y_fs_test = y_test

    y_pred = classifier.predict_proba (X_fs_test)[:,1]
    t = np.array(y_test)
    p = np.array(y_pred)

    # naive bayes can produce nan-- on ramella2018 it happens.
    # in that case we replace nans by 0
    p = np.nan_to_num(p)
    y_pred_int = [int(k>=0.5) for k in p]

    acc = accuracy_score(t, y_pred_int)
    return acc



def createFSel (fExp, cache = True):
    method = fExp[0][0]
    nFeatures = fExp[0][1]["nFeatures"]

    if method == "LASSO":
        C = fExp[0][1]["C"]
        clf = LogisticRegression(penalty='l1', max_iter=500, solver='liblinear', C = C, random_state = 42)
        pipe = SelectFromModel(clf, prefit=False, max_features=nFeatures, threshold=-np.inf)

    if method == "Anova":
        pipe = SelectKBest(anova, k = nFeatures)

    if method == "Bhattacharyya":
        pipe = SelectKBest(bhattacharyya_score_fct, k = nFeatures)

    return pipe



def createClf (cExp):
    method = cExp[0][0]

    if method == "LDA":
        model = LinearDiscriminantAnalysis(solver = 'lsqr', shrinkage = 'auto')

    if method == "LogisticRegression":
        C = cExp[0][1]["C"]
        model = LogisticRegression(max_iter=500, solver='liblinear', C = C, random_state = 42)

    if method == "NaiveBayes":
        model = GaussianNB()

    return model

#
