# packages
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# set seed
seed = 314

def train_model(X, y, seed=seed):
    """
    Build a GBM on given data
    """
    model = GradientBoostingClassifier(
        learning_rate=0.1, ##started at 0.1, increasing by 10x reduces false positives but increases false negatives, Increased misclassification by 0.001.
        n_estimators=100, ##started at 100, same impact as increasing learning rate, but reduced misclassification by 0.002.  Increasing w modifiedmin_samples leaf did not decrease misclassification.
        max_depth=8, ##started at 8, increasing by 10x reduces false negatives but also increases false positives, increased misclassification by 0.026.
        subsample=1, ##started at 1, same impact as increasing max depth, including misclassification.
        min_samples_leaf=275, ##started at 1, increasing by 10x reduced both false positive and false negative rate!. Misclassification reduction of 0.005.  Values between 250 and 275 got lowest misclassification of 0.11, a 0.022 decrease!
        random_state=seed
    )
    model.fit(X, y)
    return model