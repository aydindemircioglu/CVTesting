#

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, LeavePOut
from models import train_model, test_model
import numpy as np

# Nested cross validation is similar to the K-fold with holdout test set expect instead of splitting off a holdout
# test at the beginning and use that to evaluate we are going to use folds of the data where we use each fold once
# to evaluate the model selected by the inner loop. Then we average this outer loop set of accuries to get the final
# perforamnce.
def nested_cross_validation(data, targets, hyperparameters = None, k_fold_selection = 5):

    k_outer_folds = k_inner_folds = k_fold_selection
    acc_matrix = np.zeros((k_outer_folds, k_inner_folds, len(hyperparameters)))

    # sets up the k folds of the data
    skf_outer = StratifiedKFold(n_splits=k_outer_folds, shuffle=True)
    skf_inner = StratifiedKFold(n_splits=k_inner_folds, shuffle=True)

    outer_acc = []
    acc_list = []

    # Outer loop for performance estimation
    for k, (outer_train_index, outer_test_index) in enumerate(skf_outer.split(data, targets)):
        # these are the images used in the k folds
        X_inner = data.iloc[outer_train_index]
        Y_inner = targets.iloc[outer_train_index]

        # these are the k fold test images used for model selection
        X_test = data.iloc[outer_test_index]
        Y_test = targets.iloc[outer_test_index]

        # inner loop for model selection
        # for j in range(0,k_folds - 1):
        for j, (inner_train_index, inner_valid_index) in enumerate(skf_inner.split(X_inner, Y_inner)):

            # sets up the training and valid loader
            X_train_inner = X_inner.iloc[inner_train_index]
            Y_train_inner = Y_inner.iloc[inner_train_index]
            X_valid_inner = X_inner.iloc[inner_valid_index]
            Y_valid_inner = Y_inner.iloc[inner_valid_index]

            # for each of the models or hyperparamters you want to try train and evaluate
            for idx, paramSet in enumerate(hyperparameters):
                trained_model = train_model(paramSet, X_train_inner, Y_train_inner)
                acc = test_model(trained_model, X_valid_inner, Y_valid_inner)
                acc_matrix[k, j, idx] = acc

        # calculates the mean of the models for each fold given the parameter your testing
        fold_mean = np.mean(acc_matrix[k, :, :], axis=0)
        # picks out the index of the best parameter/model
        best_parameter = np.argmax(fold_mean)

        # use that best parameter to train a new model on all of the outer fold data
        trained_model = train_model(hyperparameters[best_parameter], X_inner, Y_inner)
        # test this best model on the outer fold hold out fold and add it to the list of accuracies
        acc = test_model(trained_model, X_test, Y_test)
        acc_list.append(acc)

        outer_acc.append(acc)

    # outer_acc = np.array(outer_acc)
    # print(f"Average of outer loop folds: = {np.mean(outer_acc)}")
    # print(f"Standard deviation of fold accuracies: = {np.std(outer_acc)}")

    print ("X", end = '', flush = True)
    estAcc = np.mean(np.asarray(outer_acc))
    return estAcc

#
