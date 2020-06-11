'''
-To utilize OneVsRestClassifier, develop estimator based on
Sklearn API.
Resource : https://scikit-learn.org/stable/developers/develop.html

https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
estimatorestimator object
An estimator object implementing fit and one of decision_function or predict_proba.
'''
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
import pdb

class ActiveSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, pool_size=500,
                        max_iter=500):
        self.pool_size=pool_size
        self.max_iter=max_iter
        self.C = []
    #
    # def get_params(self, deep=True):
    #     return {"demo": self.demo_param}
    #
    # def set_params(self, **parameters):
    #     for parameter, value in parameters.items():
    #         setatrr(self,parameter,value)

    # return self
    def fit(self, X, y):
        """
        A reference implementation of a fitting function for a classifier.
        Updates self.w, SVM weights from active-learning algorithm.

        Parameters
        ----------
        X : input features (7769 documents x 20682 words per document)
            array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
           The target values. An array of int.
           ground truth labels (7769 documents, )

        Returns
        -------
        self : object
           Returns self.
        """

        n_examples, n_features = np.shape(X)  # (7769, 20682)

        complete_ind = np.arange(0, n_examples)
        self.unlabeled_ind = np.random.randint(low=0, high=n_examples, size=(self.pool_size))
		self.labeled_ind = [x for x in complete_ind if x not in self.unlabeled_ind]

        # initialize algorithm with 2 labeled data (one has +1 label, one has -1 label)
		np.random.shuffle(complete_ind)
		init_pos_ind, init_neg_ind = -1, -1

        for i in complete_ind:
			tmp_label = train_label[i, self.feature_of_interest]
			if tmp_label == 1 and init_pos_ind == -1:
				init_pos_ind = i
			elif tmp_label == -1 and init_neg_ind == -1:
				init_neg_ind = i
			if init_pos_ind != -1 and init_neg_ind != -1:
				break

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # `fit` should always return `self`
        return self

    def predict(self, X):
        """
        A reference implementation of a prediction for a classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
        The label for each sample is the label of the closest sample
        seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]
