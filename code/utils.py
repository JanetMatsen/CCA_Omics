import numpy as np
import pandas as pd

def split_df(df, split_by):
    """
    Function to split dataframe into sub-dataframes based on a given
    column label

    :param df: Pandas DataFrame to be split
    :param split_by: column to break apart by
    :return: A dictionary of DataFrames
    """
    cum_entries = 0
    sub_dfs = {}

    for lbl in df[split_by].unique():
        sub_dfs[lbl] = df[df[split_by]==lbl]
        cum_entries += sub_dfs[lbl].shape[0]

    assert(df.shape[0] == cum_entries)
    return sub_dfs


def aggregate_df(df, collapse_by, colnorm=False):
    """
    Aggregates values in a dataframe by a given column

    :param df: dataframe
    :param collapse_by: column (e.g. gene name) to sum rows by
    :param colnorm: whether to normalize by the column sum

    :return:
    """
    agg_df = df.groupby([collapse_by],axis=0).sum()
    if colnorm:
        agg_df = agg_df/agg_df.sum(axis=0)
    return agg_df

def prepare_toy_data():
    """
    Pasted directly from the sklearn example:
    http://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_compare_cross_decomposition.html#sphx-glr-auto-examples-cross-decomposition-plot-compare-cross-decomposition-py
    Note: their X, Y in Sklearn is x, z in R's sparse CCA
    """
    n = 500
    # 2 latents vars:
    l1 = np.random.normal(size=n)
    l2 = np.random.normal(size=n)

    latents = np.array([l1, l1, l2, l2]).T
    # X, Y are 250 examples with 4 features each.
    X = latents + np.random.normal(size=4 * n).reshape((n, 4))
    Y = latents + np.random.normal(size=4 * n).reshape((n, 4))

    X_train = X[:n / 2]
    Y_train = Y[:n / 2]
    X_test = X[n / 2:]
    Y_test = Y[n / 2:]

    # Python has column features and row samples
    # R has row samples and column features.
    #return X_train.T, Y_train.T, X_test.T, Y_test.T
    return X_train, Y_train, X_test, Y_test
