import pandas as pd
import numpy as np
from itertools import combinations


class ApneaData(object):
    def __init__(self, X):
        self.X = X
        self.X_all = X
        self.cat_cols = None
        self.non_cat_cols = None
        self._classify_cols()

    def _classify_cols(self, cat_cols=None, cat_thres=5):
        if cat_cols:
            self.cat_cols = cat_cols
        else:
            self.cat_cols = list(self.X.columns[self.X.nunique() < cat_thres])
        self.non_cat_cols = [i for i in self.X.columns
                             if i not in self.cat_cols]
        self.X_cat = self.X[self.cat_cols]
        self.X_non_cat = self.X[self.non_cat_cols]

    def create_categorical_dummies(self, X):
        """
        Generates categorical dummy columns.
        """
        cats = []
        for col in self.cat_cols:
            col_data = pd.get_dummies(self.X_cat[col].copy()).iloc[:, -1]
            n_cols = col_data.shape[1]
            if n_cols == 1:
                col_data.columns = [col]
            elif n_cols > 1:
                col_data.columns = [col + '_{}'.format(i + 1)
                                    for i in range(n_cols)]
        all_cat_data = pd.concat(cats, axis=1)
        self.X_cat = all_cat_data
        self.X_all = pd.concat(self.X_cat, self.X_non_cat)

    def engineer_logs(self, vars_log=None):
        """
        Applies the natural logarithm to all continuous columns
        """
        if not vars_log:
            vars_log = self.non_cat_cols
        for var in vars_log:
            self.X_all['log_' + var] = np.log(self.X[var])

    def engineer_interactions(self):
        """
        Creates interaction variables out of all columns.
        """
        interaction_cols = self.X.columns
        for var_1, var_2 in combinations(interaction_cols, 2):
            title = "{}_{}".format(var_1.lower(), var_2.lower())
            self.X_all[title] = self.X[var_1] * self.X[var_2]

    def apply_PCA(self):
        """
        Applies PCA to self.X_all
        """
        raise NotImplementedError
