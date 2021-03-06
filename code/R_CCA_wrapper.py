import numpy as np

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

from rpy2.robjects import pandas2ri
pandas2ri.activate()
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

R_PMA = importr('PMA')
R_CCA = robjects.r('CCA')


class CCA(object):
    """
    Accepts two matrices (row samples, column features), and arguments for
    regularization strengths.
    """
    def __init__(self, x, z, penalty_x=0.2, penalty_z=0.2, K=1,
                 upos=True, vpos=True):
        """
        :param x: dataframe or numpy array
        :param z: dataframe or numpy array
        :param penalty_x: regularization strength for x.  Zero --> sparse
        coefficients; 1 --> minimal sparsity.
        :param penalty_z: regularization strength for z.  Zero --> sparse
        coefficients; 1 --> minimal sparsity.
        :param K: number of vectors to find.  Only tested for K=1.
        :param upos: force u values (weights) to be positive (R argument)
        :param vpos: force v values (weights) to be positive (R argument)
        """
        self.x = x
        self.z = z
        #self.check_standard_scalar(x)
        #self.check_standard_scalar(z)
        self.penalty_x = penalty_x
        self.penalty_z = penalty_z
        self.K = K
        # initialize
        self.upos = upos
        self.vpos = vpos
        self.CCA = self.run_CCA()

    @staticmethod
    def check_standard_scalar(m):
        # Make sure the matrix is zero-centered and has unit variance
        centers = np.sum(m, axis=0)
        var = np.var(m, axis=0)
        assert np.max(np.abs(centers)) < 1e-5, "center is not near zero"
        assert np.max(np.abs(var) - 1) < 1e-5, "variance is not 1"

    def run_CCA(self):
        # make R matrices for the x and z vectors.
        # The convert_to_r_matrix function can be replaced by the normal
        # pandas2ri.py2ri to convert dataframes, with a subsequent call to
        # R as.matrix function.
        xr, xc = self.x.shape
        zr, zc = self.z.shape
        x_R = robjects.r.matrix(self.x, nrow=xr, ncol=xc)
        z_R = robjects.r.matrix(self.z, nrow=zr, ncol=zc)

        # run CCA in R.  We want standardize = False because StandardScalar
        # will be applied in Python.  Also, R's PMA package doesn't give
        # access to the standardization method, or the transformed matrices
        cca = R_CCA(x_R, z_R, standardize=False,
                     typex="standard", typez="standard", K=1,
                     niter=1000, upos=self.upos, vpos=self.vpos,
                     penaltyx=self.penalty_x, penaltyz=self.penalty_z)
        return cca

    def extract_u_v(self):
        u = pandas2ri.ri2py_dataframe(self.CCA.rx('u')).T.as_matrix()
        v = pandas2ri.ri2py_dataframe(self.CCA.rx('v')).T.as_matrix()
        return {'u': u.reshape((u.shape[0])),
                'v': v.reshape((v.shape[0]))}

    def extract_correlation(self):
        corr_df = pandas2ri.ri2py_dataframe(self.CCA.rx('cors'))
        return corr_df.values[0,0]


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
