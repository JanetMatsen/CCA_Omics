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

def trim_features(df, min_sample_frac, val=None):
    """
    trim features that appear less than n times.
    If validation data is provided, include that data so there are more samples
    that can count, but also a larger total expected number of samples.
    """

    print('eliminate features that appear in less than {0:.0f}% of '
          'samples'.format(min_sample_frac*100))

    # slick way of counting nonzero elements in each column:
    counts = df.astype(bool).sum(axis=0) # Pandas series
    if val is not None:
       counts_val = val.astype(bool).sum(axis=0)  # Pandas series
       counts_sum = counts.add(counts_val)

    # Get list of features with more than min_sample_frac of nonzero values
    if val is None:
        criteria = counts > min_sample_frac * df.shape[0] # Pandas series
        trimmed_df = df[criteria.index[criteria]] # http://stackoverflow.com/questions/29281815/pandas-select-dataframe-columns-using-boolean
        print('trimmed from {} columns to {} columns'.format(df.shape[1], trimmed_df.shape[1]))
        return trimmed_df
    else:
        criteria = counts_sum > min_sample_frac * (df.shape[0] + val.shape[0])
        trimmed_df = df[criteria.index[criteria]]
        trimmed_val = val[criteria.index[criteria]]
        return trimmed_df, trimmed_val

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

    # apply standard scalar to each.
    from sklearn.preprocessing import StandardScaler
    ss_X = StandardScaler()
    ss_Y = StandardScaler()
    X_train = ss_X.fit_transform(X_train)
    Y_train = ss_Y.fit_transform(Y_train)
    X_test = ss_X.transform(X_test)
    Y_test = ss_Y.transform(Y_test)

    # Python has column features and row samples
    # R has row samples and column features.
    #return X_train.T, Y_train.T, X_test.T, Y_test.T
    return X_train, Y_train, X_test, Y_test
