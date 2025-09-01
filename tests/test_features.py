import pandas as pd
from src.data import generate_synthetic_churn, load_split
from src.features import get_feature_list, build_preprocessor


def test_preprocessor_fit_transform():
    df = generate_synthetic_churn(n_samples=500, random_state=0)
    X_train, X_test, y_train, y_test = load_split(df, test_size=0.2, random_state=0)
    numeric, categorical = get_feature_list(X_train)
    pre = build_preprocessor(numeric, categorical)
    pre.fit(X_train)
    Xt = pre.transform(X_test)
    assert Xt.shape[0] == X_test.shape[0]
