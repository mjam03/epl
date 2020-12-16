import numpy as np
import pandas as pd
import statsmodels.api as sm

from epl.dspy_preprocess_utils import apply_feature_scaling


def statsmodels_create_eval_df(data_set, model, add_int, X, y, x_train, x_test, y_train, y_test, stand_feats, norm_feats, std_scaler, norm_scaler):

    if data_set == 'train':
        # if train, then can access predictions directly in statsmodels
        pred_values = model.fittedvalues
        pred_values.name = 'lambda'
        eval_df = pd.concat([y_train, x_train, pred_values], axis=1)
    elif data_set == 'test':
        # if test we need to predict
        x_test_scaled, std_scaler, norm_scaler = apply_feature_scaling(
            x_test, stand_feats, norm_feats, std_scaler=std_scaler, norm_scaler=norm_scaler)
        pred_values = model.predict(
            (sm.add_constant(x_test_scaled) if add_int else x_test_scaled))
        if isinstance(pred_values, np.ndarray):
            pred_values = pd.Series(pred_values)
        pred_values.name = 'lambda'
        eval_df = pd.concat([y_test, x_test_scaled, pred_values], axis=1)
    elif data_set == 'all':
        # if all need to predict (as only trained on train)
        x_scaled, std_scaler, norm_scaler = apply_feature_scaling(
            X, stand_feats, norm_feats, std_scaler=std_scaler, norm_scaler=norm_scaler)
        pred_values = model.predict(
            (sm.add_constant(x_scaled) if add_int else x_scaled))
        if isinstance(pred_values, np.ndarray):
            pred_values = pd.Series(pred_values)
        pred_values.name = 'lambda'
        eval_df = pd.concat([y, x_scaled, pred_values], axis=1)

    return eval_df


if __name__ == '__main__':
    None
