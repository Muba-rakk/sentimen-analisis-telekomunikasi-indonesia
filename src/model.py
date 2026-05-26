import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from src.config import RANDOM_STATE

def train(X_train, y_train, params: dict) -> LogisticRegression:
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return model

def tune(X_train, y_train, C_values: list[float]) -> tuple[LogisticRegression, float, float]:
    param_grid = {'C': C_values}
    grid = GridSearchCV(
        LogisticRegression(solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_['C'], grid.best_score_

def predict(model: LogisticRegression, X_test) -> np.ndarray:
    return model.predict(X_test)
