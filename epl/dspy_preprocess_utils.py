from sklearn.preprocessing import StandardScaler, MinMaxScaler


def apply_feature_scaling(df, stand_feats, norm_feats, std_scaler=None, norm_scaler=None):
    '''
    df: Raw df of y and features
    pred_col: str col name of the y variable
    stand_feats: list of colsto standardise (sub mean, div std dev)
    norm_feats: list of cols to normalise (min /max)
    If opt arg scalers passed then will just apply these (rather than also fit)
    This is used for apply train data fit scalers to test data
    Returns tuple of (scaled_df, stand_scaler, norm_scaler)
    '''

    used_df = df.copy()
    # check if cols to standardise
    if len(stand_feats) > 0:
        # check if std_scaler passed:
        if std_scaler is None:
            # then we need to create and fit one
            std_scaler = StandardScaler()
            used_df[stand_feats] = std_scaler.fit_transform(
                used_df[stand_feats])
        else:
            # apply the scaler passed
            used_df[stand_feats] = std_scaler.transform(used_df[stand_feats])

    # same for normalise
    if len(norm_feats) > 0:
        # check if std_scaler passed:
        if norm_scaler is None:
            # then we need to create and fit one
            norm_scaler = MinMaxScaler()
            used_df[norm_feats] = norm_scaler.fit_transform(
                used_df[norm_feats])
        else:
            # apply the scaler passed
            used_df[norm_feats] = norm_scaler.transform(used_df[norm_feats])

    return (used_df, std_scaler, norm_scaler)


if __name__ == '__main__':
    None
