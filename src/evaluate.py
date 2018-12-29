import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve


def get_optimal_models(data, models):
    """
    Performs grid search hyperparameter optimization on the full training set,
    to later retrain in cross-validation in `run_k_fold`.
    """
    X, y = data
    optimal_models = []
    for m in models:
        m.optimize(X, y)
        optimal_models.append(m)
    return optimal_models


def run_k_fold(data, model, k=10):
    """
    Runs `k` Kfold predictions.
    """
    X, y = data
    shuffle_mask = np.random.permutation(range(X.shape[0]))
    X, y = X[shuffle_mask], y[shuffle_mask]
    k_folds = KFold(n_splits=k)
    y_preds, y_tests = [], []
    for idx_train, idx_test in k_folds.split(X):
        model.fit(X[idx_train], y[idx_train])
        y_pred = model.predict_proba(X[idx_test])
        y_preds.append(y_pred)
        y_tests.append(y[idx_test])
    return np.concatenate(y_tests).flatten(), np.concatenate(y_preds).flatten()


def get_average_roc_results(data, optimized_models, k=10, plot=True):
    """
    Predicts on `k` validation sets, finds the total AUC of each, and plots the
    ROC.
    """
    model_results = []
    for model in optimized_models:
        y_true, y_pred = run_k_fold(data, model, k=k)
        auc = model_auc(y_true, y_pred, name=model.name, plot=plot)
        model_results.append((model.name, auc))
    return model_results


def model_auc(y_true, y_pred, name=None, plot=True):
    auc = roc_auc_score(y_true, y_pred)
    if plot:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        plt.plot(fpr, tpr, label='{}: {}'.format(auc, name))
        plt.xlabel("1-Specificity")
        plt.ylabel("Sensitivity")
        plt.title("ROC Curve (ROC: {})".format(round(auc, 3)))
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    return auc


def evaluate(data_types, models, k=10):
    # iterate over datatypes
    for data_tup in data_types:
        data_type_name, X, y = data_tup
        for model in models:
            model.initialize_params(X)
        optimal_models = get_optimal_models((X, y), models)

        plt.figure()
        average_results = get_average_roc_results((X, y), optimal_models)

        # modify plot
        plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.title(data_type_name)

        # return results
        print(average_results)
