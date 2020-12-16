from IPython.display import display, Markdown

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import metrics
import statsmodels.api as sm


def multinomail_logit_overview():
    display(Markdown(
        '''
__Estimating a Multinomial Logit model__
<br>
__Model Overview__
<br>
Assume underlying categorical variable $y_i, i=1...K$ has a multinomial distribution - we want to estimate for a set of data $[y_i, X_i]$ a set of K probabilities, $P(y_i = k)$
<br>
Underlying probability distribution:
<br>
$$P(y_i = k) = p_1^{x_1} \\cdot p_2^{x_2} \\cdot ... p_n^{x_n}$$
where $x_i = 1$ when $k = x_i$
<br>
We estimate this using a logit model - we model the log odds i.e. the log of the ratio of probabilities as a linear function:
<br>
$$Odds_k = \\frac{P(y_i=k)}{P(y_i=1)} = e^{\\alpha + \\beta X_i}$$
<br>
where we fix one case $y_i = 1$ as the base case to which all else is compared i.e. likelihood vs base
<br>
Utilising a log link function we thus use MLE to estimate the following equation:
$$ln(Odds_k) = ln(\\frac{P(y_i=k)}{P(y_i=1)}) = \\alpha + \\beta X_i$$
<br>
In practise we estimate the following:
<br>
$$P(y_i = k) = \\frac{e^{\\alpha_k + \\beta_k X_i}}{\sum_{i=1}^{K}e^{\\alpha_k + \\beta_k X_i}}$$
<br>
i.e. estimate a set of $[\\alpha_k, \\beta_k]$ params for each outcome k, utilising softmax to normalise $\sum_{i=1}^K{P(y_i = k)} = 1$
<br>
Coefficients can be interpreted as the increase in likelihood of event k happening (vs base) for a 1 unit increase in the variable i.e. for $x_m$ feature: 
<br>
$$\\frac{Odds_{k,x_m+1}}{Odds_k} = e^{\\beta_m}$$
<br>
Solving steps:
- Estimate values for all $\\alpha_k$ and $\\beta_k$
- Compute the implied probabilities for all obs for all K
- Use max likelihood function to compute likelihood
- Use max likelihood function to see how we did - then iterate to maximise likelihood function
'''
    ))
    return None


def poisson_glm_overview():
    display(Markdown(
        '''
__Estimating a Poisson GLM model__
<br>
__Model Overview__
<br>
Assume underlying discrete variable $z_i$ has a poisson distribution - we want to estimate for a set of data $[y_i, X_i]$ what this $\lambda_i$ should be
<br>
Utilising the $e$ to ensure $\lambda>0$ we thus have the following model specification:
<br>
$$\lambda_i = e^{\\alpha + \\beta X_i}$$
<br>
Utilising a log link function we thus use MLE to estimate the following equation:
<br>
$$ln(\lambda_i) = \\alpha + \\beta X_i$$
Solving steps:
- Estimate values for $\\alpha$ and $\\beta$
- Using data for $X_i$ obtain estimates for $\lambda_i$
- Use these to generate $P(z_i = k | \lambda_i)$ for all observations
- Use max likelihood function to see how we did - then iterate on $\\alpha$ and $\\beta$ to maximise likelihood function
'''
    ))
    return None


def display_model_overview(model):

    # determine model and print context
    model_name = model.model.__class__.__name__
    if model_name == 'GLM':
        family_name = model.model.family.__class__.__name__
        if family_name == 'Poisson':
            # then print Poisson GLM description
            return poisson_glm_overview()
        else:
            print('No overview currently available for {} {}'.format(
                model_name, family_name))
    elif model_name == 'MNLogit':
        return multinomail_logit_overview()
    else:
        print('No overview currently available for {} model'.format(model_name))
        return None


def statsmodels_sort_glm(glm_df, alpha, model_name, param_sort_cols):

    if 'Sig' in param_sort_cols:
        glm_df['Sig'] = glm_df['p'] < alpha

    # if OLS then param base val is 0, else GLM so 1
    # i.e. if param is insignificant in Poisson GLM then indiff to 1
    if model_name == 'OLS':
        param_base = 0
    else:
        param_base = 1
    glm_df['coef_mag'] = (glm_df.coef - param_base).abs()

    # if we wanna sort by coef then sort by abs value version
    if 'coef' in param_sort_cols:
        param_sort_cols = [x.replace('coef', 'coef_mag')
                           for x in param_sort_cols]

    # sort the results and return
    glm_df = glm_df.sort_values(param_sort_cols, ascending=False)
    return glm_df


def statsmodels_glm_output(model, alpha=0.05, dp=3, param_sort_cols=['Sig', 'coef']):

    # work out model type --> if glm with log link then we need to exponentiate the params
    model_name = model.model.__class__.__name__
    expon = False
    if model_name != 'OLS':
        if model.model.family is not None:
            if model.model.family.link.__class__.__name__ == 'log':
                expon = True

    # get params
    params = model.params
    if expon:
        params = np.exp(params)
    params.name = 'coef'

    # get confidence interval for params
    conf_int = model.conf_int(alpha=alpha).rename(columns={0: 'lb', 1: 'ub'})
    if expon:
        conf_int = np.exp(conf_int)

    # get pval and z vals
    pvals = model.pvalues
    zvals = model.tvalues
    pvals.name = 'p'
    zvals.name = 'z'

    # concat them all together, add col for significance
    glm_df = pd.concat([params, conf_int, pvals, zvals], axis=1)

    # determine how to sort
    # can either sort by p value
    # or sort by sig or not, then sort by param value within
    if param_sort_cols is not None:
        glm_df = statsmodels_sort_glm(
            glm_df, alpha, model_name, param_sort_cols)
        df_out = np.round(glm_df, dp)
    else:
        df_out = np.round(glm_df, dp)

    return df_out


def fetch_mnl_values(mnl_values, col_name, class_field, cat_map, expon=True):

    df = pd.melt(mnl_values.reset_index(), id_vars='index',
                 value_vars=cat_map.keys(), var_name=class_field, value_name='value')
    df[class_field] = df[class_field].map(cat_map)
    if expon:
        df.value = np.exp(df.value)
    df = df.rename(columns={'index': 'Param', 'value': col_name})
    return df


def statsmodels_mnl_output(model, alpha=0.05, dp=3):

    # get some model data e.g. class names, cat variable name
    class_field = model.model.endog_names
    col_map = model.model._ynames_map.copy()
    base_class = col_map[0]

    # create class map from res to odds ratio col name
    cat_map = {(k-1): v for k, v in col_map.items() if k != 0}
    odds_col_map = {v: 'P(y={})/P(y={})'.format(v, base_class)
                    for k, v in cat_map.items()}

    # get coefs per class k
    coef_df = fetch_mnl_values(
        model.params, 'coef', class_field, cat_map, expon=True)

    # get p and z vals
    p_df = fetch_mnl_values(
        model.pvalues, 'p', class_field, cat_map, expon=False)
    z_df = fetch_mnl_values(
        model.tvalues, 'z', class_field, cat_map, expon=False)

    # get conf interval
    conf_df = np.exp(model.conf_int(alpha=alpha)).reset_index()
    conf_df = conf_df.rename(
        columns={'level_1': 'Param', 'lower': 'lb', 'upper': 'ub'})

    # combine them all together
    df = pd.concat([coef_df, conf_df[['lb', 'ub']],
                    p_df['p'], z_df['z']], axis=1)

    # rename class field to have odds formula
    df[class_field] = df[class_field].map(odds_col_map)

    # piv table, reorder multi index and order cols within each class k
    df = pd.pivot_table(df, columns=[class_field], values=[
                        'coef', 'lb', 'ub', 'p', 'z'], index='Param')
    df.columns = df.columns.swaplevel(0, 1)
    df = df.sort_index(axis=1, level=1)
    df = df.sort_index(axis=1, level=[0, 1])
    df = df.reindex(columns=['coef', 'lb', 'ub', 'p', 'z'], level=1)
    df = df.reindex(model.params.index)
    df = np.round(df, dp)
    return df


def statsmodels_pretty_print(model, alpha=0.05, dp=3, model_overview=True, param_sort_cols=['Sig', 'coef']):
    '''
    Helper function to provide model context and pretty print statsmodels output results
    Used instead of generic .summary() and .summary2() class methods to:
     - provide model overview for clarity
     - more easily demonstrate which variables are important
    Returns pd.DataFrame of result but also displays it
    model: sm model
    model_overview: bool to indicate whether or not to print model context before results
    param_sort_cols: list of cols to sort params by e.g. ['p', 'coef']
    '''
    # give model context if asked for
    if model_overview:
        display_model_overview(model)

    # get model type and return corresponding results
    model_name = model.model.__class__.__name__
    if model_name == 'GLM':
        df_out = statsmodels_glm_output(
            model, alpha=alpha, dp=dp, param_sort_cols=param_sort_cols)
    elif model_name == 'MNLogit':
        df_out = statsmodels_mnl_output(model, alpha=alpha, dp=dp)

    display(Markdown('__Model Output__'))
    return df_out


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


def pp_conf_matrices(list_of_act_preds, max_val=0.5):
    '''
    Accepts list of dictionaries of act, pred, label
    Returns plt.subplots grid of sns.heatmap formatted confusion matrix plots
    '''
    # create subplots to plot on
    # get how many pairs of pred, acts we have
    plot_count = len(list_of_act_preds)
    if plot_count == 4:
        ncols = 2
        nrows = 2
    else:
        # max 3 cols
        ncols = ncols = min(plot_count, 3)
        if plot_count > 3:
            nrows = plot_count / ncols
            nrows = int(-(-nrows // 1))
        else:
            nrows = 1

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows,
                             figsize=((ncols*10, nrows*7)))
    # convert ax to list and delete unneeded subplots from fig
    if isinstance(axes, np.ndarray):
        axes = axes.reshape(-1)
        # remove unneeded ax from the figure
        for a in axes[plot_count:]:
            fig.delaxes(a)
    else:
        axes = [axes]

    # iterate over each dict and plot conf matrix on ax
    matrices = []
    for conf_data, ax in zip(list_of_act_preds, axes):
        act = conf_data['act']
        pred = conf_data['pred']
        label = conf_data['label']
        m = pp_conf_matrix(act, pred, label, ax=ax, max_val=max_val)
        matrices.append(m)

    return None


def pp_conf_matrix(act, pred, label, ax=None, max_val=1.00):

    if ax is None:
        # then we need to create our own fig for the single plot
        fig, ax = plt.subplots(figsize=(10, 7))

    # convert to np.ndarrays if pd objects
    if isinstance(act, pd.Series):
        act = act.values
    if isinstance(pred, pd.Series):
        pred = pred.values

    # get labels as sorted unique actual values
    classes = np.sort(list(set(act)))
    conf_mat = metrics.confusion_matrix(act, pred)
    conf_mat = conf_mat / conf_mat.sum()
    # compute accuracy
    accuracy = conf_mat.trace() / conf_mat.sum()

    # plot the heatmap
    sns.heatmap(conf_mat, annot=True, vmin=0, ax=ax,
                vmax=max_val, fmt='.2%', cmap='Blues')
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Pred Values')
    ax.set_ylabel('Act Values')
    ax.set_title('{} Confusion Matrix, Acc: {:.2%}, Obs: {:,}'.format(
        label, accuracy, len(act), fontsize=14))
    return ax


if __name__ == "__main__":
    None
