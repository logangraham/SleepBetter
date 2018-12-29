import re
import numpy as np
import pandas as pd
import tensorflow as tf
import gpflow
from gpflow.models import VGP
from gpflow.kernels import RBF
from gpflow.likelihoods import Bernoulli
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from tpot import TPOTClassifier
# from points_model import *


class ModelParamObject(object):
    """
    Base class for a model and its cross-validation paramgrid.

    Attributes:
        params (dict): param grid used for cross validation
        class_weights (list): class weights for methods that use them
        n_estimators (list): n_estimators for tree-based methods
        C_scale (np.array): penalization array for penalized methods.
        model (obj): model class.
        params_init (bool): whether or not params have been initialized.
        fitted (bool): whether the model has already been fit.
        name (str): the model name.
        to_grid_search (bool): whether or not to perform GridSearchCV on this model
        optimized (bool): whether model has been optimized
    """

    def __init__(self):
        self.X = None
        self.params = None
        self.class_weights = [{1: i} for i in [1, 3, 5, 10]]
        self.n_estimators = [10, 50, 100, 150, 200]
        self.C_scale = np.append(np.logspace(-4, 4, 10), 1e10)
        self.model = None
        self.params_init = False
        self.fitted = True
        self.name = re.findall("'(.*?)'", str(self.__class__))[0]
        self.to_grid_search = True
        self.optimized = False

    def initialize_params(self, X):
        """
        Initializes the parameters.
        """
        self.params = {}
        self.params_init = True

    def return_params(self):
        """
        Returns params.
        """
        return self.params

    def optimize(self, X, y):
        """
        Perform cross-validation on the model.
        """
        mod = GridSearchCV(estimator=self.model,
                           param_grid=self.params,
                           scoring='roc_auc',
                           verbose=2,
                           n_jobs=-1,
                           cv=15)
        mod.fit(X, y)
        self.model = mod.best_estimator_
        self.optimized = True
        print("Stored best model.")

    def fit(self, X, y):
        """
        Fits the model.
        """
        self.model.fit(X, y)
        self.fitted = True

    def predict(self, X):
        """
        Runs prediction using fitted model.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        In the classification setting: return simplex-constrained logits of
        predictions.
        """
        one_index = np.where(self.model.classes_ == 1.)[0]
        return self.model.predict_proba(X)[:, one_index]

# class KangModelParam(ModelParamObject):
#     """
#     The model class for the Kang et al. (2015) logistic-regression-based
#     points-based model.
#     """
#     def __init__(self):
#         ModelParamObject.__init__(self)
#         self.model = PointsModel()

#     def optimize(self, X, y):
#         self.fit(X, y)


class RFModelParam(ModelParamObject):
    """
    Random forest model and parameters.
    """

    def __init__(self):
        ModelParamObject.__init__(self)
        self.model = RandomForestClassifier()

    def initialize_params(self, X):
        self.params = {"n_estimators": self.n_estimators,
                       "min_samples_split": [2, 5, 10, 15, 20],
                       "min_samples_leaf": [3, 6, 10],
                       "max_features": ['auto'],
                       "bootstrap": [False, True],
                       "warm_start": [False],
                       "class_weight": self.class_weights}


class LRCVModelParam(ModelParamObject):
    """
    Regularized logistic regression.
    """

    def __init__(self):
        ModelParamObject.__init__(self)
        self.model = LogisticRegressionCV(solver='lbfgs')

    def initialize_params(self, X):
        self.params = {"penalty": ['l2','l1'],
                       "solver": ['liblinear'],
                       "scoring": ['roc_auc']}


class LRModelParam(ModelParamObject):
    """
    Non-regularized logistic regression.
    """

    def __init__(self):
        ModelParamObject.__init__(self)
        self.model = LogisticRegression(solver='lbfgs')

    def initialize_params(self, X):
        self.params = {"C": [1e10]}


class SVMModelParam(ModelParamObject):
    """
    SVM classifier.
    """
    def __init__(self):
        ModelParamObject.__init__(self)
        self.model = SVC(gamma='scale', probability=True)

    def initialize_params(self, X):
        self.params = {"C": self.C_scale,
                       "kernel": ['rbf', 'sigmoid', 'linear'],
                       "probability": [True],
                       "class_weight": self.class_weights}


class CatBoostModelParam(ModelParamObject):
    """
    Catboost model.
    """

    def __init__(self):
        ModelParamObject.__init__(self)
        self.model = CatBoostClassifier()

    def initialize_params(self, X):
        if not isinstance(X, pd.core.frame.DataFrame):
            raise TypeError("X must be Pandas DataFrame.")
        self.params = {"n_estimators": self.n_estimators,
                       "class_weight": self.class_weights,
                       "num_trees": self.n_estimators}
        self.params = {}
        cat_features = X.nunique() <= 4
        cat_features = list(cat_features[cat_features].index)
        if cat_features:
            self.params['cat_features'] = cat_features
        else:
            self.params['cat_features'] = [None]

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class NaiveBayesModelParam(ModelParamObject):
    """
    Naive Bayes classifier.
    """

    def __init__(self):
        ModelParamObject.__init__(self)
        self.model = GaussianNB()


class GPModelParam(ModelParamObject):
    """
    Gaussian process classifier
    """

    def __init__(self):
        ModelParamObject.__init__(self)
        self.fitted = False

    def fit(self, X, y):
        X, y = self._prep_data(X, y)
        tf.reset_default_graph()
        self.graph = tf.Graph()
        with tf.Session(graph=self.graph) as sess:
            m = self.get_model(X, y)
            opt = gpflow.train.ScipyOptimizer()
            opt.minimize(m, maxiter=2000)
            self.model = m
            self.model_params = dict(m.read_trainables())
            self.fitted = False

    def _prep_data(self, X, y):
        self.X = X
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.y = y
        return X, y

    def get_model(self, X, y):
        return gpflow.models.VGP(X, y,
                                 kern=gpflow.kernels.RBF(X.shape[1]),
                                 likelihood=gpflow.likelihoods.Bernoulli())

    def optimize(self, X, y):
        self.fit(X, y)

    def predict(self, X):
        tf.reset_default_graph()
        with tf.Session(graph=self.graph) as sess:
            m = self.get_model(self.X, self.y)
            m.assign(self.model_params)
            return m.predict_y(X)[0].round()

    def predict_proba(self, X):
        tf.reset_default_graph()
        with tf.Session(graph=self.graph) as sess:
            m = self.get_model(self.X, self.y)
            m.assign(self.model_params)
            return m.predict_y(X)[0]


class TPOTModelParam(ModelParamObject):
    def __init__(self):
        ModelParamObject.__init__(self)
        self.model = TPOTClassifier(verbosity=2, max_time_mins=0.5)
        self.fitted = False
        self.to_grid_search = False

    def optimize(self, X, y):
        """
        In the TPOT case, this runs the standard TPOT optimization algorithm.
        """
        print("Performing TPOT genetic optimization.")
        self.model.fit(X, y)
        self.optimized = True

    def predict_proba(self, X):
        return self.model.predict_proba(X)