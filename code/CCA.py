
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pprint
from sklearn.preprocessing import StandardScaler


from R_CCA_wrapper import CCA

class CcaAnalysis(object):
    """
    Canonical Correlation Analysis for two vectors.
    Uses the CCA function in R's PMA package.
    Python calls R directly via py2r
    """
    def __init__(self, x, z, penalty_x, penalty_z, val_x=None, val_z=None,
                 standardize_before_R=True):
        self.x = x
        self.z = z
        if standardize_before_R:
            self.center_and_standardize()

        self.penalty_x = penalty_x
        self.penalty_z = penalty_z

        # run CCA
        self.CCA = CCA(self.x, self.z,
                       penalty_x=penalty_x, penalty_z=penalty_z)
        uv_dict = self.CCA.extract_u_v()
        self.u = uv_dict['u']
        self.v = uv_dict['v']

        self.val_x = val_x
        self.val_z = val_z

        self.project()
        self.summary = None

    def center_and_standardize(self):
        """
        Make a given fold have unit-variance and zero mean
        :return:
        """
        ss_for_x = StandardScaler()
        x_ss = ss_for_x.fit_transform(self.x)
        self.x_orig = self.x
        self.x = x_ss
        self.ss_for_x = ss_for_x

        ss_for_z = StandardScaler()
        z_ss = ss_for_z.fit_transform(self.z)
        self.z_orig = self.z
        self.z = z_ss
        self.ss_for_z = ss_for_z

    def project(self):
        self.x_projected = self.x.dot(self.u)
        self.z_projected = self.z.dot(self.v)
        if (self.val_x is not None) and (self.val_z is not None):
            self.x_val_projected = self.val_x.dot(self.u)
            self.z_val_projected = self.val_z.dot(self.v)

    def plot_projections(self, filename=None):
        # scatter plot of x.dot(u) vs z.dot(v)
        # one color for train, one color for validation
        fig, ax = plt.subplots(1, 1, figsize=(3,3))
        colors = ['#bdbdbd','#31a354']
        if (self.val_x is not None) and (self.val_z is not None):
            plot_vars = [(self.x_projected, self.z_projected),
                        (self.x_val_projected, self.z_val_projected)]
        else:
            plot_vars = [(self.x_projected, self.z_projected)]

        series = 0
        for (x, y) in plot_vars:
            plt.scatter(x, y, linestyle='--', marker='o', color=colors[series])
            series += 1
        plt.legend(loc = 'best')

        if filename is not None:
            fig.savefig(filename + '.pdf')

        return fig

    @staticmethod
    def correlation(x, z):
        # return correlation of projections for x.dot(u) and z.dot(v)
        corr_matrix = np.corrcoef(x,z)
        assert corr_matrix.shape == (2,2)
        return corr_matrix[0,1]

    def summarise(self):
        summary = {}
        summary['train correlation'] = \
            self.correlation(self.x_projected, self.z_projected)
        if (self.val_x is not None) and (self.val_z is not None):
            summary['validation correlation'] = \
                self.correlation(self.x_val_projected, self.z_val_projected)
        summary['# nonzero u weights'] = self.num_nonzero(self.u)
        summary['# nonzero v weights'] = self.num_nonzero(self.v)

        #summary = {k:[v] for k, v in summary.items()}
        #return pd.DataFrame(summary)
        self.summary = summary

    def get_summary(self):
        if self.summary is None:
            self.summarise()
        return self.summary

    @staticmethod
    def num_nonzero(vector):
        # for counting # of nonzero components in u and v
        return sum(vector != 0)


class CcaExpression(CcaAnalysis):
    """
    Wrapper for CCA that is more specific to our expression data.
    Understands that there are gene features, which can be asked for by name
    for plotting and such.
    """
    def __init__(self, x, z, penalty_x, penalty_z, val_x=None, val_z=None):

        # save the gene names; they will be stripped off by the CCA instance
        self.x_genes = x.columns
        self.z_genes = z.columns
        super(CcaExpression, self).__init__(x=x, z=z,
                                            penalty_x=penalty_x,
                                            penalty_z=penalty_z,
                                            val_x=val_x, val_z=val_z)

    @staticmethod
    def strip_pandas_to_numpy(thing):
        if isinstance(thing, pd.DataFrame):
            print("convert dataframe to naked numpy")
            if 'Unnamed: 0' in thing.columns:
                del thing['Unnamed: 0']
            a = thing.as_matrix()
            return a
        else:
            return thing

    def hist_of_counts_for_feature(self, feature):
        # can pass a feature number (int) or gene name (string)
        if isinstance(feature, int):
            counts = self.x.ix[:, feature]
            title = self.x.columns[feature]
        elif isinstance(feature, str):
            counts = self.x[feature]
            title = feature
        else:
            print('oops; expected an int or string')

        fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))
        plt.hist(counts, bins=self.x.shape[0])
        plt.title(title)
        return fig

    def hist_for_top_features(self, n_features):
        # will plot histograms for the top (largest magnitude) n features
        pass


if __name__ == '__main__':
    # DEMO ONLY
    from utils import prepare_toy_data
    X_train, Y_train, X_test, Y_test = prepare_toy_data()
    cca_demo = CcaAnalysis(x=X_train, z=Y_train,
                           penalty_x=0.999, penalty_z=0.999,
                           val_x=X_test, val_z=Y_test)
    pprint.pprint(cca_demo.get_summary())
