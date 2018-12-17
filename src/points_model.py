import pandas as pd
import numpy as np
import statsmodels.discrete.discrete_model as sm


class PointsModel():
    def __init__(self, bin_bmi=([[0, 20], [20, 25], [25, 30],
                                [30, 35], [35, float('inf')]], 5),
                       bin_age=([[0, 3], [3, 6], [6, 9], [9, 12],
                                [12, 15], [15, float('inf')]][::-1], 3),
                       bin_zscore=([[-float('inf'), -1], [-1, 0], [0, 1],
                                    [1, 2], [2, float('inf')]][::-1], 1)):
        self.bin_bmi = bin_bmi
        self.bin_age = bin_age
        self.bin_zscore = bin_zscore
        self.bin_dict = {'bmi': self.bin_bmi,
                         'age': self.bin_age,
                         'zscore': self.bin_zscore}
        self.X = None
        self.y = None
        self.coefficients = None
        self.fitted = False
        self.sig_params = None

    def _get_coefficients(self, X, y):
        """
        This function builds the reduced logistic regression model,
        finds the significant variables, performs another logistic
        regression using just those variables, and keeps the
        coefficients in a dict for later use.
        """

        sig_params = self._fit_logit(X, y,
                                     keep_significant=True,
                                     sig_value=0.05)[1]
        new_logit = self._fit_logit(X[sig_params], y,
                                    keep_significant=False)
        self.X_old = self.X.copy()
        self.X = X[sig_params]
        return dict(new_logit.params)

    def _fit_logit(self, X, y, keep_significant, sig_value=0.05):
        f = sm.Logit(y, X).fit()
        if keep_significant:
            sig_cols = list(f.params[f.pvalues <= sig_value].index)
            return f, sig_cols
        else:
            return f

    def _logistic(self, summed):
        """
        The _logistic function.
        """
        return 1. / (1 + np.exp(-(summed)))

    def _get_diff_score(self, value, rangelist, scale_factor):
        """
        Calculates Sullivan et al. difference score. Unnormalized.
        """
        for i, group in enumerate(rangelist):
            if (value >= group[0]) and (value < group[1]):
                return scale_factor * float(i)

    def fit(self, X, y):
        """
        Calculates the total points score of each entry
        and then associates them with a risk score.
        """
        if isinstance(X, pd.core.frame.DataFrame):
            self.X = X
        else:
            raise Exception("""X must be a Pandas dataframe.
                            (use pd.DataFrame(arr) if X is numpy array,
                            and add column titles (for interpretability)""")
        self.y = y
        self.coefficients = self._get_coefficients(self.X, self.y)
        total_points, risk_score = self.predict(self.X)
        self.X.loc[:, 'total_points'] = total_points
        self.X.loc[:, 'risk_score'] = risk_score
        self.fitted = True

    def predict(self, X):
        """
        The Kang et all predict function.
        Each obversation column value gets a rounded and scaled
        points score derived from a previous logistic regression.
        """
        X = X.copy()
        if not isinstance(X, pd.core.frame.DataFrame):
            raise Exception("""X must be a Pandas dataframe. \
                            (use pd.DataFrame(arr) if X is numpy array, \
                            and add column titles for interpretability)""")
        X['total_points'] = 0
        for column in self.coefficients:
            vector = X[column]
            if column in self.bin_dict:
                points = map(lambda x:
                             self._getDiffScore(x,
                                                self.bin_dict[column][0],
                                                self.bin_dict[column][1]),
                                                vector)
            else:
                points = vector
            final_points = self.coefficients[column] * np.array(list(points))
            X['total_points'] += final_points
        X['risk_score'] = X['total_points'].apply(self._logistic)
        return X['total_points'].values, X['risk_score'].values

    def predict_proba(self, X):
        """
        Calculates the points core and risk likelihood of a new input.
        """
        probabilities = np.array(self.predict(X)[1])
        inverse = 1 - probabilities
        return np.vstack((probabilities, inverse)).T