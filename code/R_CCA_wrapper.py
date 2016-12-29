import numpy as np
import pandas as pd

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

from rpy2.robjects import pandas2ri
pandas2ri.activate()

import pandas.rpy.common as com

R_PMA = importr('PMA')
R_CCA = robjects.r('CCA')


class CCA(object):
    def __init__(self, x, z, penalty_x=0.2, penalty_z=0.2,
                 K=1, return_orig_object=False):
        self.x = x
        self.z = z
        self.penalty_x = penalty_x
        self.penalty_z = penalty_z
        self.K = K
        # initialize
        self.CCA = self.run_CCA()

    def run_CCA(self, ):
        # make R objects for the x and z vectors.
        x = com.convert_to_r_dataframe(pd.DataFrame(self.x))
        z = com.convert_to_r_dataframe(pd.DataFrame(self.z))
        # run CCA in R.
        return R_CCA(x, z, standardize=True,
                     typex="standard", typez="standard", K=1,
                     niter=1000, penaltyx=0.2, penaltyz=0.2)

    def extract_u_v(self):
        u = pandas2ri.ri2py_dataframe(self.CCA.rx('u')).T.as_matrix()
        v = pandas2ri.ri2py_dataframe(self.CCA.rx('v')).T.as_matrix()
        return {'u': u.reshape((u.shape[0])),
                'v': v.reshape((v.shape[0]))}


# DEMO
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


if __name__ == '__main__':
    # DEMO ONLY
    from utils import prepare_toy_data
    print('DEMO: generate toy sklearn data')
    X_train, Y_train, X_test, Y_test = prepare_toy_data()
    CCA = CCA(X_train, Y_train)
    uv_dict = CCA.extract_u_v()
    u = uv_dict['u']
    v = uv_dict['v']
    print('u: \n {}'.format(u))
    print('v: \n {}'.format(v))
