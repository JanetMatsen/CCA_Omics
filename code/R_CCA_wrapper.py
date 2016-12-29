import numpy as np
import pandas as pd

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

from rpy2.robjects import pandas2ri
pandas2ri.activate()

import pandas.rpy.common as com

from utils import prepare_toy_data

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
