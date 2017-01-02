import itertools
import matplotlib.pylab as plt
import pandas as pd

from sklearn.model_selection import KFold

from CCA import CcaExpression

class CrossValidateExpressionCca(object):
    """
    Take an input file, a number of folds, and prep data and .tsv files
    to do R analysis on.
    """
    def __init__(self, x, z, folds=4, random_state=27,
                 discard_hypotheticals=True):
        self.x = x
        self.z = z
        self.folds = folds
        self.discard_hypotheticals = discard_hypotheticals

        if self.discard_hypotheticals:
            self.x = self.remove_unknown_and_hypotheticals(self.x)
            self.z = self.remove_unknown_and_hypotheticals(self.z)

        self.folds = dict()
        self.k_fold = KFold(n_splits=4, shuffle=True,
                            random_state=random_state)

        self.break_into_xval()
        self.models = dict()
        self.summary = pd.DataFrame()
        self.model_each_fold(penalty_x=1, penalty_z=1)

    def run_R_permute(self):
        # TODO.  See if my cross-val matches theirs.
        pass

    def remove_unknown_and_hypotheticals(self, df):
        discard_labels = ['unknown', 'hypothetical']
        keepers = [s for s in df.columns if
                   not any(x in s for x in discard_labels)]
        print('discard {} columns'.format(len(keepers)))
        return df[keepers]

    def break_into_xval(self):
        fold_num = 1
        for train, val in self.k_fold.split(self.x):
            print('train: {}, val:{}'.format(train, val))
            assert len(train) + len(val) == self.x.shape[0]
            assert len(train) + len(val) == self.z.shape[0]
            # split the dataframe into train and val by row number
            self.folds[fold_num] = {'x train':self.x.iloc[train, :],
                                    'z train':self.z.iloc[train, :],
                                    'x val':self.x.iloc[val, :],
                                    'z val': self.z.iloc[val, :]}
            assert self.folds[fold_num]['x train'].shape[0] + \
                   self.folds[fold_num]['x val'].shape[0] == self.x.shape[0]
            assert self.folds[fold_num]['z train'].shape[0] + \
                   self.folds[fold_num]['z val'].shape[0] == self.z.shape[0]

            fold_num += 1

    def model_each_fold(self, penalty_x, penalty_z):
        for fold_num, fold_data in self.folds.items():
            print('Run CCA for fold # {}'.format(fold_num))
            model_id = "{}_{}_{}".format(fold_num, penalty_x, penalty_z)
            CCA_instance = CcaExpression(x=fold_data['x train'],
                              z=fold_data['z train'],
                              val_x=fold_data['x val'],
                              val_z=fold_data['z val'],
                              penalty_x=penalty_x, penalty_z=penalty_z)
            summary_row = CCA_instance.get_summary()
            summary_row['CCA obj id'] = model_id
            summary_row['fold'] = fold_num
            # prep for concat in Pandas
            summary_df_row = \
                pd.DataFrame({k:[v] for k, v in summary_row.items()})
            self.summary = pd.concat([self.summary, summary_df_row], axis=0)

            # save to model dict
            self.models[model_id] = CCA_instance
        print(self.summary)

    def model_list_of_penalty_tuples(self, tup_list):
        for reg_tuple in tup_list:
            penalty_x, penalty_z = reg_tuple
            print('model for penalty_x = {}, penalty_z = {}'.format(penalty_x,
                                                                    penalty_z))
            self.model_each_fold(penalty_x=penalty_x, penalty_z=penalty_z)

    def model_combos(self, list_to_make_combos_from):
        penalty_x_vals = list_to_make_combos_from[0]
        penalty_z_vals = list_to_make_combos_from[1]
        # e.g. [0.2, 0.3], [0.2, 0.3] --> [(0.2, 0.2), (0.2, 0.3), (0.3, 0.2), (0.3, 0.3)]
        penalty_tuples = list(itertools.product(penalty_x_vals, penalty_z_vals))
        self.model_list_of_penalty_tuples(penalty_tuples)

    def plot_cross_val(self):
        colors = {1:'#9e9ac8', 2:'#807dba', 3:'#6a51a3', 4:'#4a1486'}
        fig, axs = plt.subplots(2, 2, figsize=(6,5))
        plts = {'penalty_x, train': axs[0,0],
                'penalty_z, train': axs[1,0],
                'penalty_x, val': axs[0,1],
                'penalty_z, val': axs[1,1]}
        titles = {'penalty_x, train': 'train corr, penalty_x',
                'penalty_z, train': 'train corr, penalty_z',
                'penalty_x, val': 'val corr, penalty_x',
                'penalty_z, val': 'val corr, penalty_z'}

        #import pdb; pdb.set_trace()
        for p_key in plts.keys():
            print(p_key)
            ax = plts[p_key]

            # get x variable
            if 'x' in p_key:
                x = 'penalty_x'
            else:
                x = 'penalty_z'

            # get y variable
            if 'train' in p_key:
                y = 'train correlation'
            else:
                y = 'validation correlation'

            # plot each fold's trend.
            for fold, df in self.summary.groupby('fold'):
                df.sort_values(by=x, inplace=True)
                # plot
                ax.plot(df[x], df[y], color=colors[fold],
                        marker='o', label='fold {}'.format(fold))

            ax.set_title(titles[p_key])

        # couldn't get a legend to work.
        plt.tight_layout()
        return fig



if __name__ == '__main__':
    x = pd.read_csv('../data/m_nmm_summed_on_gene/m.tsv')
    z = pd.read_csv('../data/m_nmm_summed_on_gene/nmm.tsv')
    # set the descriptive (and poorly named) sample column as the index; it
    # should not be treated as a feature (gene) downstream.
    for df in [x, z]:
        df.rename(columns={'Unnamed: 0':'sample'}, inplace=True)
        df.set_index('sample', inplace=True)
    cv = CrossValidateExpressionCca(x=x, z=z, folds=4)
