
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, LeavePOut
from models import train_model, test_model
import numpy as np


def select_shuffle_test (data, targets, hyperparameters = None, k_fold_selection = 5):
    # safely assume this
    k_fold_test = k_fold_selection

    acc_matrix = np.zeros((k_fold_selection, len(hyperparameters)))

    # sets up the k folds of the data
    skf_select = StratifiedKFold(n_splits=k_fold_selection, shuffle=True)

    # loop through all of the folds of the data
    for j, (inner_train_index, inner_valid_index) in enumerate(skf_select.split(data, targets)):

        # sets up the training and valid loader
        X_train_inner = data.iloc[inner_train_index]
        Y_train_inner = targets.iloc[inner_train_index]
        X_valid_inner = data.iloc[inner_valid_index]
        Y_valid_inner = targets.iloc[inner_valid_index]

        # for each of the models or hyperparamters you want to try train and evaluate
        for idx, paramSet in enumerate(hyperparameters):
            trained_model = train_model(paramSet, X_train_inner, Y_train_inner)
            acc = test_model(trained_model, X_valid_inner, Y_valid_inner)
            acc_matrix[j, idx] = acc

    #print(acc_matrix)

    # calculates the mean of the models for each fold given the parameter your testing
    fold_mean = np.mean(acc_matrix[:, :], axis=0)
    # picks out the index of the best parameter/model
    best_parameter = np.argmax(fold_mean)
    #print(f"Selected hyperparameter index: {best_parameter} with parameters {hyperparameters[best_parameter]} ")

    # k fold
    skf_test = StratifiedKFold(n_splits=k_fold_test, shuffle=True)
    acc_list = []

    # loops through each fold up to k and trains a model and test on the kth fold
    for train_index, test_index in skf_test.split(data, targets):
        # gets data from our largers array for k-1 folds
        X_train = data.iloc[train_index]
        Y_train = targets.iloc[train_index]
        # gets the test set
        X_test = data.iloc[test_index]
        Y_test = targets.iloc[test_index]

        # use that best parameter to train a new model on all of the outer fold data
        trained_model = train_model(hyperparameters[best_parameter], X_train, Y_train)
        # test this best hyperparamter set on the shuffled data
        acc = test_model(trained_model, X_test, Y_test)
        acc_list.append(acc)

    #print(f"List of Accuracies For Each Fold: = {acc_list}")
    #print(f"Average Fold Accuracy: = {np.mean(np.asarray(acc_list))}")

    print ("X", end = '', flush = True)
    estAcc = np.mean(np.asarray(acc_list))
    return estAcc

#
