from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.base import clone

#from bagging import Custom_Bagging_Classifier

import numpy as np
import random



class CustomBaggingClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, base_estimator=None, n_estimators=10, random_state=None):
        """
        Parameters
        ----------
        base_estimator : object or None, optional (default=None)    The base estimator to fit on random subsets of the dataset. 
                                                                    If None, then the base estimator is a decision tree.
        n_estimators : int, optional (default=10)                   The number of base estimators in the ensemble.
        random_state : int or None, optional (default=None)         Controls the randomness of the estimator. 
        """

        #TODO: ...
        self.oob_scores_ = []
        self.base_estimator = base_estimator or DecisionTreeClassifier(max_depth=2)
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.trained_bootstrap_models_=[]
        self.estimators_ = []
        self.classes_ = None
        self.n_classes_ = None
        
        

    
    
    def fit(self, X, y):
        """
        Build a Bagging classifier from the training set (X, y).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)                 The input samples.
        y : ndarray of shape (n_samples,)                            The target values.        
        Returns
        -------
        self : object
            Returns self.

        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        #TODO: ...
        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        self.trained_bootstrap_models_ = []

        for i in range(self.n_estimators):
            indices = rng.choice(X.shape[0], size=X.shape[0], replace=True)
            oob_indices = np.setdiff1d(range(X.shape[0]), np.unique(indices))
            estimator = clone(self.base_estimator)
            estimator.fit(X[indices], y[indices])
            self.estimators_.append(estimator)
            self.trained_bootstrap_models_.append(estimator) # Store the fitted model
            accuracy = estimator.score(X[oob_indices], y[oob_indices])
            self.oob_scores_.append(accuracy)
        self.estimator_weights_ = np.array(self.oob_scores_)
        self.estimator_weights_ /= np.sum(self.estimator_weights_)

        # Return the classifier
        return self


    def predict(self, X):
        """
        Predict class for X.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)                 The input samples.
        
        Returns
        -------
        pred : ndarray of shape (n_samples,)                         The predicted classes.
        """

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        #TODO: ...
        pred_ests = np.empty((X.shape[0], self.n_estimators), dtype = int)
        for i, estimator in enumerate(self.estimators_):
            pred_ests[:, i] = estimator.predict(X)
            
        pred = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis = 1, arr=pred_ests)
       
        return pred


    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)                 The input samples.

        Returns
        -------
        probas : ndarray of shape (n_samples, n_classes)             The class probabilities of the input samples. The order of 
                                                                     the classes corresponds to that in the attribute classes_.
        """

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        #TODO: ...
        probas = np.zeros((X.shape[0], len(self.classes_)))
        for estimator in self.estimators_:
            estimator_probas = estimator.predict_proba(X)
            probas += estimator_probas
        probas /= len(self.estimators_)

        return probas

    def _get_bootstrap_sample(self, X, y):
        """
        Returns a bootstrap sample of the same size as the original input X, 
        and the out-of-bag (oob) sample. According to the theoretical analysis, about 63.2% 
        of the original indexes will be included in the bootsrap sample. Some indexes will
        appear multiple times.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)                  The input samples.
        y : ndarray of shape (n_samples,)                             The target values.

        Returns
        -------
        bootstrap_sample_X : ndarray of shape (n_samples, n_features) The bootstrap sample of the input samples.
        bootstrap_sample_y : ndarray of shape (n_samples,)            The bootstrap sample of the target values.
        oob_sample_X : ndarray of shape (n_samples, n_features)       The out-of-bag sample of the input samples.
        oob_sample_y : ndarray of shape (n_samples,)                  The out-of-bag sample of the target values.
        """

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        #TODO: ...
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace = True)
       
        #oob_indices = np.setdiff1d(np.arange(n_samples), np.unique(indices))
        oob_indices = np.setdiff1d(range(n_samples), np.unique(indices))

        bootstrap_sample_X, bootstrap_sample_y = X[indices], y[indices]
        oob_sample_X, oob_sample_y = X[oob_indices], y[oob_indices]
        
        return bootstrap_sample_X, bootstrap_sample_y, oob_sample_X, oob_sample_y

