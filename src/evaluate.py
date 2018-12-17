import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve


def get_optimal_models(data, models):
    X, y = data
    optimal_models = []
    for m in models:
        m.optimize()
        optimal_models.append(m)
    return optimal_models

def run_k_fold(data, trained_model, append=False, k=10):
    """
    Runs `k` Kfold predictions.
    """
    X, y = data.copy()
    shuffle_mask = np.random.permutation(X.shape[0])
    X, y = X[shuffle_mask], y[shuffle_mask]
    k_folds = KFold(n_splits=k)

    y_preds, y_tests = [], []
    for idx_train, idx_test in k_folds.split(X):
        model.fit(X[idx_train], y[idx_train])
        y_preds.append(model.predict_proba(X[idx_test]))
        y_tests.append(y[idx_test])
    return np.hstack((np.array(y_preds).reshape(-1, 1),
                      np.array(y_tests).reshape(-1, 1)))

def get_average_roc_results(data, optimized_models, k=10, plot=True):
    """
    Predicts on `k` validation sets, finds the total AUC of each, and plots the
    ROC.
    """
    model_results = []
    for opt_model in optimized_models:
        results = run_k_fold(data, opt_model, k=k)
        auc = model_auc(y_true, y_pred, name=opt_model.name, plot=plot)
        model_results.append((model.name, auc))
    return model_results

def model_auc(y_true, y_pred, name=None, plot=True):
    auc = roc_auc_score(y_true, y_pred)
    if plot:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        plt.plot(fpr, tpr, label='(%0.2f) %s' % roc_auc[2], name)
    return auc

def evaluate(data_types, models, k=10):
    # iterate over datatypes
    for data_type_name, X, y in data_types:
        for model in optimal_models:
            model.initialize_params()
        optimal_models = get_optimal_models((X, y), models)
        average_results = get_average_roc_results((X, y), optimal_models)

        # begin plot
        plt.figure()
        get_average_roc_results(average_results, plot=True)

        # modify plot
        plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.title(data_type_name)

        # return results
        print(average_results)
