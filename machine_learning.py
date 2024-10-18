import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GridSearchCV

from my_stats import concat_features


def perform_ml_with_stats(data_arr, label_arr, grp_arr):
    features = []
    for d in data_arr:
        features.append(concat_features(d))
    feature_arr = np.array(features)
    clf = LogisticRegression()
    n_splits = min(5, len(np.unique(grp_arr)))
    gkf = GroupKFold(n_splits)
    pipe = Pipeline([('scalar', StandardScaler()), ('clf', clf)])
    param_grid = {'clf__C': [0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7]}
    gscv = GridSearchCV(pipe, param_grid, cv=gkf, n_jobs=12)
    gscv.fit(feature_arr, label_arr, groups=grp_arr)
    best_score = gscv.best_score_
    print(f'best_score {best_score}')
