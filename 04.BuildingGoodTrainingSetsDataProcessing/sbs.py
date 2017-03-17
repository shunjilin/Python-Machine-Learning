from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


class SBS():
    """ Sequential Backward Selection (SBS)
    Greedy search algorithm used to reduce initial d-dimensional feature space
    to k-dimensional feature space where k < d.


    Pseudocode:
    1. Initialize the algorithm with k = d 
    (d is the dimensionality of full feature space)

    2. Determine the feature x- that maximizes the criterion
    x- = argmaxJ(X_k - x) where x in X_k

    3. Remove x- from the feature set

    4. Terminate if k == desired number of features, else go to step 2.

    Parameters
    ----------
    scoring : 
             which scoring metric to use, default is accuracy
    estimator : 
             learning algorithm
    k_features : int
             Number of features to terminate sbs at
    test_size : float
             Proportion of split for the test set
    random_State: int
             Seed for random state
    """

    def __init__(self, estimator, k_features,
                 scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)  # deep copy of model
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ---------
        X : {array-like}, shape = [n_samples, n_features]
        Training vectors,
        where n_samples is the number of samples and n_features is the number
        of features.
        y: array-like, shape = [n_samples]
        Traget values

        Returns
        -------
        self.object

        """
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)

        dim = X_train.shape[1]  # n_features
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)

        dim = X_train.shape[1]  # train on all features
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]  # list of combinations of features
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            # all dim-1 combinations of features
            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train,
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)  # return indice of best score
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        """ Transform predictor variables by picking best features

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
        Training vectors.

        Returns
        -------
        X with best features selected according to scoring method

        """
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train,
                    X_test, y_test, indices):
        """ Get the score of the model (trained on training data) when fit
        on the test data

        Parameters
        ----------
        X_train : predictor variables of training data
        y_train : target values of training data
        X_test : predictor variables of test data
        y_test : target values of test data
        indices : array-like
                 selected features

        """
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score
