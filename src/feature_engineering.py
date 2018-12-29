import pandas as pd
import numpy as np
import prince
from sklearn.decomposition import PCA
from itertools import combinations


class ApneaData(object):
    """
    A class to manage all data transformations in one.
    All methods are static methods. The class itself also
    automatically stores important parameters and computed
    data.
    """
    def __init__(self):
        self.cat_cols = None
        self.non_cat_cols = None

    def categorize_cols(self, X, cat_cols=None, cat_thres=5):
        """
        Returns lists of categorical and non-categorical column names.
        """
        if cat_cols:
            self.cat_cols = cat_cols
        else:
            self.cat_cols = list(X.columns[X.nunique() < cat_thres])
        self.non_cat_cols = [i for i in X.columns
                             if i not in self.cat_cols]
        self.X_cat = X[self.cat_cols]
        self.X_non_cat = X[self.non_cat_cols]
        return self.cat_cols, self.non_cat_cols
    
    def categorize_data(self, X):
        """
        Returns a tuple of (categorical, non-categorical) data.
        """
        cat_cols, non_cat_cols = self.categorize_cols(X)
        return X[cat_cols], X[non_cat_cols]

    def create_categorical_dummies(self, X):
        """
        Generates categorical dummy columns.
        """
        X = X.copy()
        cat_cols, non_cat_cols = self.categorize_cols(X)
        cats = []
        for col in cat_cols:
            col_data = pd.get_dummies(X[col]).iloc[:, :-1]
            n_cols = col_data.shape[1]
            if n_cols == 1:
                col_data.columns = [col]
            elif n_cols > 1:
                col_data.columns = [col + '_{}'.format(i + 1)
                                    for i in range(n_cols)]
            cats.append(col_data)
        cat_data = pd.concat(cats, axis=1)
        return pd.concat([cat_data, X[non_cat_cols]], axis=1)

    def engineer_logs(self, X, vars_log=None):
        """
        Applies the natural logarithm to all continuous columns.
        First takes the absolute value of the columns.
        """
        X = X.copy()
        if vars_log is None:
            _, vars_log = self.categorize_cols(X)
        for var in vars_log:
            X['log_' + var] = np.log(np.abs(X[var]))
        return X

    def engineer_interactions(self, X):
        """
        Creates interaction variables out of all columns.
        """
        X = X.copy()
        _, interaction_cols = self.categorize_cols(X)
        for var_1, var_2 in combinations(interaction_cols, 2):
            title = "{}_{}".format(var_1.lower(), var_2.lower())
            X[title] = X[var_1] * X[var_2]
        return X
    
    def categorical_MCA(self, X, n_components=4, n_iter=1000):
        X = X.copy()
        cat_data, non_cat_data = self.categorize_data(X)
        mca = prince.MCA(n_components=n_components, n_iter=n_iter)
        cat_reduced = mca.fit_transform(cat_data)
        cat_reduced.columns = ["MCA_{}".format(i+1) for i, v in enumerate(cat_reduced.columns)]
        return pd.concat([cat_reduced, non_cat_data], axis=1)

    def non_categorical_PCA(self, X, n_components=4):
        """
        Applies PCA to self.X_all
        """
        X = X.copy()
        cat_data, non_cat_data = self.categorize_data(X)
        pca = PCA(n_components=n_components)
        non_cat_reduced = pd.DataFrame(pca.fit_transform(non_cat_data))
        non_cat_reduced.columns = ["PCA_{}".format(i+1) for i, v in enumerate(non_cat_reduced.columns)]
        return pd.concat([cat_data, non_cat_reduced], axis=1)

    def engineer_all_data(self, X):
        """
        Applies the entire pipline without PCA.
        """
        X = X.copy()
        X = self.create_categorical_dummies(X)
        X = self.engineer_logs(X)
        X = self.engineer_interactions(X)
        X = self.non_categorical_PCA(X)
        X = self.categorical_MCA(X)
        return X