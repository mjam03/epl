from IPython.display import display, Markdown

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import metrics
import statsmodels.api as sm


def sm_logit_output_explanation():

    display(Markdown('#### StatsModels Logistic Regression Overview'))
    display(Markdown('Statsmodels estimates a set of K-1 models for binary classification - as such, the alphabetic first class becomes the base'))
    display(Markdown('For example, given a base class 0, the default coefficients for class k that statsmodels reports (in .summary()) are:'))
    display(Markdown(
        r'$$ln Odds_k = ln \frac{P(y=k)}{P(y=0)} = \alpha + \beta_1 x_1 + \cdots + \beta_m x_m$$'))
    display(Markdown('Exponentiating both sides, this gives us:'))
    display(Markdown(
        r'$$Odds_k = \frac{P(y=k)}{P(y=0)} = e^{\alpha + \beta_1 x_1 + \cdots + \beta_m x_m}$$'))

    display(
        Markdown('Now we can interpret what it means to increase variable x_m by 1:'))
    display(Markdown(
        r'$$\frac{Odds_{k,x_m+1}}{Odds_k} = \frac{e^{[\alpha + \beta_1 x_1 +...+ \beta_m (x_m + 1)]}}{e^{[\alpha + \beta_1 x_1 +...+ \beta_m x_m]}}$$'))
    display(Markdown('This simplifies to:'))
    display(Markdown(r'$$\frac{Odds_{k,x_m+1}}{Odds_k} = e^{\beta_m}$$'))

    display(Markdown('Thus, we take the raw output for the coefficients from stats models and exponentiate to give us the odds ratios'))
    display(Markdown(
        'We can then interpret these exponentiated coefficients in the following way:'))
    display(Markdown(
        'For a 1 unit increase in $x_m$, we expect the odds to be multiplied by $e^{\\beta_m}$'))
    return None


def print_logit_params(model, alpha=0.05):

    # get class map and base class
    class_field = model.model.endog_names
    col_map = model.model._ynames_map.copy()
    base_class = col_map[0]

    # create class map from res to odds ratio col name
    cat_map = {(k-1): v for k, v in col_map.items() if k != 0}
    odds_col_map = {v: 'P(y={})/P(y={})'.format(v, base_class)
                    for k, v in cat_map.items()}

    # handle param coefs
    # melt to right format, and map 0,1 to class names
    coef_df = pd.melt(model.params.reset_index(), id_vars='index', value_vars=[
                      0, 1], var_name=class_field, value_name='value')
    coef_df[class_field] = coef_df[class_field].map(cat_map)
    # exponentiate for odd ratios and rename cols
    coef_df.value = np.exp(coef_df.value)
    coef_df = coef_df.rename(columns={'index': 'Param', 'value': 'coef'})

    # handle confidence intervals
    conf_df = np.exp(model.conf_int(alpha=alpha)).reset_index()
    conf_df = conf_df.rename(
        columns={'level_1': 'Param', 'lower': 'lb', 'upper': 'ub'})

    # handle z stat
    z_df = pd.melt(model.tvalues.reset_index(), id_vars='index', value_vars=[
                   0, 1], var_name=class_field, value_name='value')
    z_df[class_field] = z_df[class_field].map(cat_map)
    z_df = z_df.rename(columns={'index': 'Param', 'value': 'z'})

    df = pd.merge(left=coef_df, right=conf_df,
                  how='inner', on=['FTR', 'Param'])
    df = pd.merge(left=df, right=z_df, how='inner', on=['FTR', 'Param'])

    df[class_field] = df[class_field].map(odds_col_map)
    df = pd.pivot_table(df, columns=[class_field], values=[
                        'coef', 'lb', 'ub', 'z'], index='Param')
    df.columns = df.columns.swaplevel(0, 1)
    df = df.sort_index(axis=1, level=1)
    df = df.sort_index(axis=1, level=[0, 1])
    df = df.reindex(columns=['coef', 'lb', 'ub', 'z'], level=1)
    df = df.reindex(model.params.index)
    df = np.round(df, 3)

    display(Markdown('__Statsmodels Logistic Regression Model Output__'))
    display(df)

    return df


def gen_sm_logit_preds(model, x_test):

    if x_test.shape[1] == (model.K - 1):
        # add intercept
        x_test = sm.add_constant(x_test)

    # gen predicted probabilities
    y_pred = model.predict(x_test)
    # now we need to convert these to class output based on max prob
    probs = y_pred.values
    preds = []
    res_map = model.model._ynames_map.copy()

    for p in probs:
        max_p = max(p)
        class_n = list(p).index(max_p)
        pred = res_map[class_n]
        preds.append(pred)

    return preds


def print_conf_mat(model, y_test, x_test, lib='statsmodels', include_train=True, x_train=None, y_train=None):

    # function to output 2 confusion matrices
    # one for in sample and one for out of sample
    if lib == 'statsmodels':
        # gen test conf mat
        y_pred = gen_sm_logit_preds(model, x_test)
        classes = model.model._ynames_map.values()
        # gen train conf mat
        if include_train:
            train_mat = model.pred_table()
            train_acc = (len(model.model.endog) -
                         model.resid_misclassified.sum()) / len(model.model.endog)
    elif lib == 'sklearn':
        # gen test conf mat
        y_pred = model.predict(x_test)
        classes = model.classes_
        # gen train conf mat
        if include_train:
            if x_train is None:
                return 'Must include x_train with sklearn if include_train=True'
            else:
                train_pred = model.predict(x_train)
                train_mat = metrics.confusion_matrix(y_train, train_pred)
                train_acc = metrics.accuracy_score(y_train, train_pred)

    # compute test conf mat
    test_mat = metrics.confusion_matrix(y_test, y_pred)
    test_acc = metrics.accuracy_score(y_test, y_pred)

    max_acc = round(max((test_mat / test_mat.sum()).max(),
                        (train_mat / train_mat.sum()).max()), 1) + 0.1

    # create the subplots for train and test matrices
    cols = (2 if include_train else 1)
    fig, ax = plt.subplots(ncols=cols, figsize=(10*cols, 7))

    if include_train:
        sns.heatmap(train_mat / train_mat.sum(), annot=True, vmin=0,
                    vmax=max_acc, ax=ax[0], fmt='.2%', cmap='Blues')
        ax[0].set_xticklabels(classes)
        ax[0].set_yticklabels(classes)
        ax[0].set_xlabel('Pred Values')
        ax[0].set_ylabel('Act Values')
        ax[0].set_title('Training Confusion Matrix, Acc: {:.2%}'.format(
            train_acc), fontsize=14)

        sns.heatmap(test_mat / test_mat.sum(), annot=True, vmin=0,
                    vmax=max_acc, ax=ax[1], fmt='.2%', cmap='Blues')
        ax[1].set_xticklabels(classes)
        ax[1].set_yticklabels(classes)
        ax[1].set_xlabel('Pred Values')
        ax[1].set_ylabel('Act Values')
        ax[1].set_title('Test Confusion Matrix, Acc: {:.2%}'.format(
            test_acc), fontsize=14)

    else:
        sns.heatmap(test_mat / test_mat.sum(), annot=True, vmin=0,
                    vmax=max_acc, ax=ax, fmt='.2%', cmap='Blues')
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.set_xlabel('Pred Values')
        ax.set_ylabel('Act Values')
        ax.set_title('Test Confusion Matrix, Acc: {:.2%}'.format(
            test_acc), fontsize=14)

    return None


if __name__ == "__main__":
    None
