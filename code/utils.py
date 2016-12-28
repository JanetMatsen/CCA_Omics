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
