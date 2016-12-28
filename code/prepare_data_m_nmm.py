import os
import pandas as pd

from utils import split_df, aggregate_df

DATA_DIR = '../data/'
GENE_EXP_DATA = DATA_DIR + 'raw_data/3_summary_rpkm.xls'
GENOMES = DATA_DIR + 'genomes_curated.tsv'
OUTPUT_DIR = DATA_DIR + 'm_nmm_summed_on_gene/'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

isolate_expression = pd.read_csv(GENE_EXP_DATA, sep='\t')
genomes = pd.read_csv(GENOMES, sep='\t')

data = isolate_expression.merge(genomes)

split_by_type = split_df(data, 'type')

# m for methanotroph, nmm for methylotroph
m_exp = split_by_type['m']
nmm_exp = split_by_type['nmm']

# collapse on gene product (name)
m_exp = aggregate_df(m_exp, 'product', colnorm=False)
nmm_exp = aggregate_df(nmm_exp, 'product', colnorm=False)

# record num rows before trimming out zero variance rows
m_rows = m_exp.shape[0]
nmm_rows = nmm_exp.shape[0]

# Remove rows with zero variance. R's CCA function won't tolerate them.
m_exp = m_exp.loc[m_exp.std(axis=1) > 0.001]
nmm_exp = nmm_exp.loc[nmm_exp.std(axis=1) > 0.001]

print("Trimming zero-variance rows: {} --> {} rows".format(
    m_rows, m_exp.shape[0]))
print("Trimming zero-variance rows: {} --> {} rows".format(
    nmm_rows, nmm_exp.shape[0]))

# save data
m_exp.T.to_csv(OUTPUT_DIR + 'm.tsv')
nmm_exp.T.to_csv(OUTPUT_DIR + 'nmm.tsv')

# Save the gene names
m_names = m_exp.index.to_series()
nmm_names = nmm_exp.index.to_series()
m_names.to_csv(OUTPUT_DIR + 'm_genes.tsv')
nmm_names.to_csv(OUTPUT_DIR + 'nmm_genes.tsv')


