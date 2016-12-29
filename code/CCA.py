
import matplotlib.pylab as plt
import numpy as np
import pprint
import os
import re
import subprocess

import pandas as pd

from R_CCA_wrapper import CCA

class CcaAnalysis(object):
    """
    Canonical Correlation Analysis for two vectors.
    Uses the CCA function in R's PMA package.
    Python calls R directly via py2r
    """
    def __init__(self, x, z, penalty_x, penalty_z, val_x=None, val_z=None):
        self.x = x
        self.z = z
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


if __name__ == '__main__':
    # DEMO ONLY
    from utils import prepare_toy_data
    X_train, Y_train, X_test, Y_test = prepare_toy_data()
    cca_demo = CcaAnalysis(x=X_train, z=Y_train,
                           penalty_x=0.999, penalty_z=0.999,
                           val_x=X_test, val_z=Y_test)
    pprint.pprint(cca_demo.get_summary())

    # Demo with our expression CCA
    x =
