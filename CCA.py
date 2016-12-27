
import matplotlib.pylab as plt
import numpy as np
import os
import re
import subprocess

import pandas as pd

class CcaAnalysis(object):
    """
    Analyze the results of sparse CCA results (run in R)
    """
    def __init__(self, x, z, u, v, val_x=None, val_z=None):
        self.x = x
        self.z = z
        self.u = u
        self.v = v
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

class ExpressionCCA(CcaAnalysis):
    """
    Wrapper class: prepare and run CCA for particular x, z data sets
    """
    def __init__(self, x_train_filepath, z_train_filepath,
                 x_gene_filepath, z_gene_filepath,
                 input_filepath, u_v_output_dir,
                 penalty_x, penalty_z,
                 noise=0,
                 # validation data is optional; final model won't have it.
                 x_val_filepath=None, z_val_filepath=None,
                 verbose = False,
                 path_to_R_script='../../code/sparse_CCA.R'):

        self.penalty_x = penalty_x
        self.penalty_z = penalty_z
        self.noise = noise
        self.path_to_R_script=path_to_R_script

        assert os.path.exists(input_filepath), \
            "input filepath, {}, doesn't exist".format(input_filepath)
        if not os.path.exists(u_v_output_dir):
            os.mkdir(u_v_output_dir)
        self.u_v_output_dir = u_v_output_dir

        self.x_train_filepath = x_train_filepath
        self.z_train_filepath = z_train_filepath
        self.x_val_filepath = x_val_filepath
        self.z_val_filepath = z_val_filepath
        self.x_gene_filepath = x_gene_filepath
        self.z_gene_filepath = z_gene_filepath

        self.x_genes = pd.read_csv(x_gene_filepath, sep='\t', header=None)
        self.z_genes = pd.read_csv(z_gene_filepath, sep='\t', header=None)
        self.x_genes.columns = ['gene'] # rename df column
        self.z_genes.columns = ['gene'] # rename df column

        self.penalty_x = penalty_x
        self.penalty_z = penalty_z

        # prepare u and v
        x, z, u, v = self.write_csv_and_run_R(verbose=verbose)

        if (self.x_val_filepath is not None) and (self.z_val_filepath is not None):
            val_x=self.load_array(self.x_val_filepath)
            val_z=self.load_array(self.z_val_filepath)
        else:
            val_x, val_z = None, None


        super(ExpressionCCA, self).__init__(
            x=x, z=z, u=u, v=v,
            val_x=val_x, val_z=val_z)

    @staticmethod
    def load_array(filepath, verbose=False):
        vector = np.genfromtxt(filepath, delimiter='\t')
        if verbose:
            print('vector {} has shape {}'.format(filepath, vector.shape))
        return vector

    def write_csv_and_run_R(self, delete_u_v=False, verbose=False):
        x = self.load_array(self.x_train_filepath)
        self.x = x
        z = self.load_array(self.z_train_filepath)
        self.z = z

        # Add noise per Sham's advice.  Note that this is only an ok place to
        # put it since our data has already been normalized with standard scalar.
        if self.noise > 0.0:
            print('add noise:')
            self.x_noised = np.random.normal(loc=0, scale=self.noise, size=self.x.shape)*self.noise
            self.z_noised = np.random.normal(loc=0, scale=self.noise, size=self.z.shape)*self.noise

        names = self.load_array(self.x_gene_filepath)
        self.gene_names = names

        # get the data back out
        def prepare_output_filename(input_filename, extra_string):
            # methylotroph_fold1_train.tsv --> fold1_train_u.tsv
            s = os.path.basename(input_filename)
            if verbose:
                print('input filename: {}'.format(input_filename))
                print('extra string: {}'.format(extra_string))
                print('output dir: {}'.format(self.u_v_output_dir))
                print('s: {}'.format(s))
            m = re.search('[_A-z]+(fold[0-9]+[._A-z]+.tsv)', s)
            if m is not None: # (match found)
                s = m.group(1)
                s = s.replace('.tsv', '_{}.tsv'.format(extra_string))
                s = os.path.join(self.u_v_output_dir, s)
            else:
                print('no group name found; must not be X-val data')
                s = s.replace('.tsv', '_{}.tsv'.format(extra_string))
                s = os.path.join(self.u_v_output_dir, s)

            if verbose:
                print('Will save output for {} to {}'.format(input_filename, s))
            return s

        u_path = prepare_output_filename(self.x_train_filepath,
                                         extra_string='u_penX' + str(self.penalty_x)
                                        + '_penZ' + str(self.penalty_z))
        v_path = prepare_output_filename(self.z_train_filepath ,
                                         extra_string='v_penX' + str(self.penalty_x)
                                        + '_penZ' + str(self.penalty_z))

        # Run R
        # todo: this will keep concatenating to the same file.  Need to delete
        # it sometimes, put a time stamp on it, or something else.
        stdout_file = open('stdout_CCA.txt', 'a')
        stderr_file = open('stderr_CCA.txt', 'a')

        if self.noise > 0.0:
            noisy_x_path = './noisy_x.tsv'
            #np.savetext(noisy_x_path, self.x_noised, delimiter='\t')
            self.x_noised.tofile(noisy_x_path, sep='\t')
            noisy_z_path = './noisy_z.tsv'
            #np.savetext(noisy_z_path, self.z_noised, delimiter='\t')
            self.z_noised.tofile(noisy_z_path, sep='\t')
            command = ['Rscript', self.path_to_R_script,
                       noisy_x_path, noisy_z_path,
                       u_path, v_path,
                       str(self.penalty_x), str(self.penalty_z)]
        else:
            command = ['Rscript', self.path_to_R_script,
                       self.x_train_filepath, self.z_train_filepath,
                       u_path, v_path,
                       str(self.penalty_x), str(self.penalty_z)]
        if verbose:
            print('command: \n {}'.format(" ".join(command)))
        subprocess.check_call(command, stdout=stdout_file, stderr=stderr_file)
        stdout_file.close()
        stderr_file.close()

        # R adds a header row, 'V1' we chop off.
        print('read u in from {}'.format(u_path))
        print('read v in from {}'.format(v_path))
        u = np.genfromtxt(u_path, delimiter='\t', skip_header=1)
        v = np.genfromtxt(v_path, delimiter='\t', skip_header=1)
        if verbose:
            print(u.shape)
            print(v.shape)

        # todo: assert some shape constraints.

        return x, z, u, v

    def associate_weights_with_gene_names(self):
        x_gene_df = self.x_genes
        u_weight_df = pd.DataFrame({'weight':self.u,
                                  'abs(weight)':np.abs(self.u)})
        u_info = pd.concat([x_gene_df, u_weight_df], axis=1)
        self.u_with_names = u_info

        # not DRY
        z_gene_df = self.z_genes
        v_weight_df = pd.DataFrame({'weight':self.v,
                                  'abs(weight)':np.abs(self.v)})
        v_info = pd.concat([z_gene_df, v_weight_df], axis=1)
        self.v_with_names = v_info

    def sorted_weights(self, vector='u'):
        if self.u_with_names is None or self.v_with_names is None:
            self.associate_weights_with_gene_names()

        if vector=='u':
            return(self.u_with_names.sort_values(by='abs(weight)', ascending=False))
        elif vector=='v':
            return(self.v_with_names.sort_values(by='abs(weight)', ascending=False))
        else:
            print('expected vector argument to be "u" or "v"')

