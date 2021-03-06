
from functools import partial
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pprint
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from R_CCA_wrapper import CCA
from utils import trim_features


class CcaAnalysis(object):
    """
    Canonical Correlation Analysis for two vectors.
    Uses the CCA function in R's PMA package.
    Python calls R directly via py2r
    """
    def __init__(self, x, z, scaling, penalty_x, penalty_z,
                 upos=True, vpos=True,
                 val_x=None, val_z=None, standardize_before_R=True):
        """

        :param x:
        :param z:
        :param scaling:
        :param penalty_x:
        :param penalty_z:
        :param val_x:
        :param val_z:
        :param standardize_before_R:
        """
        self.x = x
        self.z = z

        if scaling is None:
            self.scaling = 'none'
        else:
            self.scaling = scaling # for summary
            self.scale_features()

        self.penalty_x = penalty_x
        self.penalty_z = penalty_z
        self.upos = upos
        self.vpos = vpos

        # run CCA
        self.CCA = CCA(self.x, self.z,
                       penalty_x=penalty_x, penalty_z=penalty_z,
                       upos=upos, vpos=vpos)
        uv_dict = self.CCA.extract_u_v()
        self.u = uv_dict['u']
        self.v = uv_dict['v']

        self.val_x = val_x
        self.val_z = val_z

        self.project()
        self.summary = None

    def scale_features(self):
        """
        Make a given fold have unit-variance and zero mean
        :return:
        """
        if self.scaling == 'StandardScaler':
            scaler = StandardScaler
        elif self.scaling == 'StandardScaler_without_mean':
            scaler = partial(StandardScaler, with_mean=False)
        elif self.scaling == "MinMaxScaler":
            scaler = MinMaxScaler
        else:
            raise Exception

        ss_for_x = scaler()
        x_ss = ss_for_x.fit_transform(self.x)
        self.x_orig = self.x
        self.x = x_ss
        self.ss_for_x = ss_for_x

        ss_for_z = scaler()
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
        summary['correlation (R)'] = self.CCA.extract_correlation()
        summary['train correlation'] = \
            self.correlation(self.x_projected, self.z_projected)
        if (self.val_x is not None) and (self.val_z is not None):
            summary['validation correlation'] = \
                self.correlation(self.x_val_projected, self.z_val_projected)
        summary['penalty_x'] = self.penalty_x
        summary['penalty_z'] = self.penalty_z
        summary['# nonzero u weights'] = self.num_nonzero(self.u)
        summary['# nonzero v weights'] = self.num_nonzero(self.v)
        summary['upos'] = str(self.upos)
        summary['vpos'] = str(self.vpos)
        summary['scaling'] = self.scaling

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
    def __init__(self, x, z, penalty_x, penalty_z,
                 scaler='StandardScaler',  #'MinMaxScaler',
                 val_x=None, val_z=None,
                 upos=True, vpos=True,
                 min_frac_of_samples=0.5):

        # We may want to restrict to features appearing in, say, 50% of samples.
        if min_frac_of_samples is not None:
            if val_x is None and val_z is None:
                x = trim_features(x, min_frac_of_samples)
                z = trim_features(z, min_frac_of_samples)
            else:
                x, val_x = trim_features(x, min_frac_of_samples, val_x)
                z, val_z = trim_features(z, min_frac_of_samples, val_z)

        # save the gene names; they will be stripped off by the CCA instance
        self.x_genes = x.columns.to_series()
        self.z_genes = z.columns.to_series()
        self.x_genes.reset_index(drop=True, inplace=True)
        self.z_genes.reset_index(drop=True, inplace=True)

        self.upos = upos
        self.vpos = vpos

        super(CcaExpression, self).__init__(x=x, z=z,
                                            scaling=scaler,
                                            penalty_x=penalty_x,
                                            penalty_z=penalty_z,
                                            upos=upos, vpos=vpos,
                                            val_x=val_x, val_z=val_z)

        self.x_feature_weights = pd.DataFrame({'gene':self.x_genes,
                                               'weight':self.u,
                                               'abs(weight)':np.abs(self.u)})
        self.z_feature_weights = pd.DataFrame({'gene':self.z_genes,
                                               'weight':self.v,
                                               'abs(weight)':np.abs(self.v)})

        self.summarise()
        self.summary['min frac of samples'] = min_frac_of_samples

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

    def feature_values(self, feature, vector, transformed=False):
        """
        Return the values (expression levels for each sample) for the given
        feature (specified by name or index), and vector (x or z).
        Can return the transformed (StandardScalar) or un-transformed values.
        """
        if vector == 'x' or vector == 'u':
            if transformed:
                print("histogram of normalized counts for x:")
                features = getattr(self, 'x')
            else:
                print("histogram of counts for x:")
                features = getattr(self, 'x_orig')
            genes = getattr(self, 'x_genes')

        elif vector == 'z' or vector == 'v':
            if transformed:
                print("histogram of transformed counts for z:")
                features = getattr(self, 'z')
            else:
                print("histogram of counts for z:")
                features = getattr(self, 'z_orig')
            genes = getattr(self, 'z_genes')

        # Make sure the features are numpy, not pandas.
        # As of writing, x_orig is still Pandas.  todo: (make consistent??)
        if not isinstance(features, np.ndarray):
            features = features.copy().as_matrix()

        # can pass a feature number (int) or gene name (string)
        if isinstance(feature, int):
            counts = features[:, feature]
            title = genes[feature]

        elif isinstance(feature, str):
            numpy_index = pd.Index(genes).get_loc(feature)
            assert type(numpy_index) == int
            counts = features[:, numpy_index]
            title = feature
        else:
            print('oops; expected an int or string')

        return counts, title

    def hist_of_counts_for_feature(self, feature, vector, transformed=False):

        counts, title = self.feature_values(feature, vector=vector,
                                            transformed=transformed)

        fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))
        ax.axvline(x=0, color='#6B706B', linestyle=':')

        if transformed:
            color = '#78c679'
        else:
            color = '#969696'

        plt.hist(counts, bins=len(counts), color=color)
        plt.title(title, y=1.08)

        return fig

    def hist_of_raw_and_transformed_counts_for_feature(self, feature, vector):
        raw, raw_title = self.feature_values(feature, vector=vector,
                                             transformed=False)
        trans, trans_title = self.feature_values(feature, vector=vector,
                                                 transformed=True)

        fig, axs = plt.subplots(2, 1, figsize=(3.5, 4))
        ax0 = axs[0]
        ax0.axvline(x=0, color='#6B706B', linestyle=':')

        ax1 = axs[1]
        ax1.axvline(x=0, color='#6B706B', linestyle=':')

        ax0.hist(raw, bins=len(raw), color='#969696')
        ax0.set_title('raw feature values', fontsize=10)

        ax1.hist(trans, bins=len(trans), color='#78c679')
        ax1.set_title('normalized ({})'.format(self.scaling), fontsize=10)
        assert raw_title == trans_title
        fig.suptitle(raw_title, y=1.08, fontsize=14)
        plt.tight_layout()
        return fig

    def top_features(self, vector, n_features='all', zeros=False):
        """
        Get the top features (max abs(weight)) according to the model fit.
        :param n_features: number of features
        """
        if vector == 'x':
            df = self.x_feature_weights
        elif vector == 'z':
            df = self.z_feature_weights

        if n_features == 'all':
            n_features = df.shape[0] # keep all the features.
        else:
            assert isinstance(n_features, int)

        vec_sorted = df.copy().sort_values('abs(weight)', ascending=False)

        if not zeros:
            vec_sorted = vec_sorted[vec_sorted['weight'] != 0]

        return vec_sorted.iloc[0:n_features, :]

    def hist_for_top_features(self, vector, transformed, n_features=3):
        top_features = self.top_features(vector=vector, n_features=n_features,
                                         zeros=False)
        if transformed == 'both':
            plot_fun = self.hist_of_raw_and_transformed_counts_for_feature
        elif transformed:
            plot_fun = partial(self.hist_of_counts_for_feature,
                               transformed=True)
        elif not transformed:
            plot_fun = partial(self.hist_of_counts_for_feature,
                               transformed=False)

        for gene_name in top_features['gene']:
            plot_fun(feature=gene_name, vector=vector)

    def hist_of_weights_for_top_features(self, n_features='all'):
        x_features = self.top_features(vector='x', n_features=n_features)
        z_features = self.top_features(vector='z', n_features=n_features)
        print('Num x features: {}.  Num z features: {}'.format(
            x_features.shape[0], z_features.shape[0]))

        fig, axs = plt.subplots(2, 1, figsize=(5, 4))
        plt_data = {1:x_features, 2:z_features}
        titles = {1:'x feature weights', 2:'z feature weights'}
        colors = {1:'#b3cde3', 2:'#decbe4'}

        for row, ax in enumerate(axs, start=1):
            print(row, ax)
            ax.hist(plt_data[row]['weight'], color=colors[row])
            ax.set_xlabel('weight')
            ax.set_title(titles[row])

        plt.tight_layout()
        return fig


if __name__ == '__main__':
    print("warning: demo only includes CcaAnalysis, not CcaExpression")
    # DEMO ONLY
    from utils import prepare_toy_data
    X_train, Y_train, X_test, Y_test = prepare_toy_data()
    cca_demo = CcaAnalysis(x=X_train, z=Y_train,
                           penalty_x=0.999, penalty_z=0.999,
                           val_x=X_test, val_z=Y_test)
    pprint.pprint(cca_demo.get_summary())
