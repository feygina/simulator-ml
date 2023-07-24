from typing import Tuple

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score
# Import necessary libraries
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from scipy.stats import norm


def roc_auc_percentile_ci(
    classifier: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    conf: float = 0.95,
    n_bootstraps: int = 10_000,
) -> Tuple[float, float]:
    """Returns confidence bounds of the ROC-AUC using percentile method"""
    #get prediction
    y_pred = classifier.predict_proba(X)[:, 1] 
    # Empty array to store bootstrapped AUC scores
    auc_scores = np.empty(n_bootstraps)
    # bootstap X, y
    for i in range(n_bootstraps):
        success = False
        max_retries = 10  # Maximum number of retries
        retry = 0
        while not success and retry < max_retries:
            # Generate random indices for bootstrap sampling
            indices = np.random.randint(0, len(y_pred), len(y_pred))
            # Bootstrap sample using the random indices
            y_pred_bootstrap = y_pred[indices]
            y_true_bootstrap = y[indices]
            # Calculate AUC score for the bootstrap sample
            try:
                auc_scores[i] = roc_auc_score(y_true_bootstrap, y_pred_bootstrap)
                success = True
                break # Break out of the "retry while" if successful
            except:
                retry += 1
        if not success and retry == max_retries:
            raise ValueError("AUC score computation failed too many times for bootstrap sample.")
    # Calculate the lower and upper bounds of the confidence interval
    lcb = np.percentile(auc_scores, 100 * (1 - conf)/2)
    ucb = np.percentile(auc_scores, 100 * (1 + conf)/2)
    return (lcb, ucb)


def roc_auc_normal_ci(
    classifier: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    conf: float = 0.95,
    n_bootstraps: int = 10_000,
) -> Tuple[float, float]:
    """Returns confidence bounds of the ROC-AUC using normal ci"""
    #get prediction
    y_pred = classifier.predict_proba(X)[:, 1]
    point_estimate = roc_auc_score(y, y_pred)
    # Empty array to store bootstrapped AUC scores
    auc_scores = np.empty(n_bootstraps)
    # bootstap X, y
    for i in range(n_bootstraps):
        success = False
        max_retries = 10  # Maximum number of retries
        retry = 0
        while not success and retry < max_retries:
            # Generate random indices for bootstrap sampling
            indices = np.random.randint(0, len(y_pred), len(y_pred))
            # Bootstrap sample using the random indices
            y_pred_bootstrap = y_pred[indices]
            y_true_bootstrap = y[indices]
            # Calculate AUC score for the bootstrap sample
            try:
                auc_scores[i] = roc_auc_score(y_true_bootstrap, y_pred_bootstrap)
                success = True
                break # Break out of the "retry while" if successful
            except:
                retry += 1
        if not success and retry == max_retries:
            raise ValueError("AUC score computation failed too many times for bootstrap sample.")
    # Calculate the lower and upper bounds of the confidence interval
    alpha = 1 - conf
    z = norm.ppf(1 - alpha / 2)
    se = np.std(auc_scores)
    lcb, ucb = point_estimate - z * se, point_estimate + z * se
    return (lcb, ucb)



# Generate a sample classification dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=41)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=41)

# Create an instance of a classifier (e.g., Logistic Regression)
classifier = LogisticRegression()
classifier = classifier.fit(X_train, y_train)
# Call the roc_auc_ci function
lower_bound, upper_bound = roc_auc_ci(classifier, X_test, y_test)

# Print the confidence bounds of the ROC-AUC
print(f"Confidence bounds of ROC-AUC: {lower_bound} - {upper_bound}")
