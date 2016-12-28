# Background:
# CCA with sparsity http://biorxiv.org/content/biorxiv/early/2016/04/06/047217.full.pdf
# cites https://faculty.washington.edu/dwitten/Papers/pmd.pdf
# R package: https://cran.r-project.org/web/packages/PMA/PMA.pdf
# R package cites:
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2861323/pdf/sagmb1470.pdf
# See /ML_meta-omics/analysis/CCA_methanotroph_expression_methylotroph_expression/sparse_CCA_X_m_expression_Y_nmm_abundance.R
# for work that led to this script.

# Inputs: csv filename for gene expression vectors, penalty_x, penalty_z
# Outputs: u and v vectors from CCA


library("PMA")
library("ggplot2")

args = commandArgs(trailingOnly=TRUE)
x_filepath = args[1]
print(paste("x file path:", x_filepath))
z_filepath = args[2]
print(paste("z file path:", z_filepath))

u_filepath = args[3]
v_filepath = args[4]

penaltyx = as.numeric(args[5])
penaltyz = as.numeric(args[6])
print(paste('penaltyx:', penaltyx))
print(paste('penaltyz:', penaltyz))

x <- read.csv(x_filepath, sep = '\t', header = FALSE, stringsAsFactors = FALSE)
z <- read.csv(z_filepath, sep = '\t', header = FALSE, stringsAsFactors = FALSE)

# Note: R is modifying the gene (column) names:
# http://stackoverflow.com/questions/10441437/x-in-my-column-names-of-an-r-data-frame
# (1->4)-alpha-D-glucan 1-alpha-D-glucosylmutase
# becomes "X.1..4..alpha.D.glucan.1.alpha.D.glucosylmutase"

dim(x) # [1]   83 4593
dim(z) # [1]   83 7355

# Cannot have NAs in x or z
# Both should have 0 rows.  (Not true for nmm if read.table is used instaed of read.csv)
stopifnot(nrow(x[rowSums(is.na(x)) > 0,]) == 0)
stopifnot(nrow(z[rowSums(is.na(z)) > 0,]) == 0)

# get rid of the 'product' column, and transpose.
#x = x[, -1]
#z = z[, -1]

# train model
print('train model')
model <- CCA(x, z,
             #standardize=FALSE, # don't standardize or we can't transform
             typex="standard", typez="standard", K=1,
             niter=1000,
             penaltyx=penaltyx, penaltyz=penaltyz)

model_stats <- function(CCA_obj){
        x_penalty = CCA_obj$penaltyx
        u = CCA_obj$u[,1]
        u_len = length(u)
        u_zeros = sum(u == 0)
        u_coeffs = sum(u != 0)
        u_frac_zeros = u_zeros/u_len

        z_penalty = CCA_obj$penaltyz
        v = CCA_obj$v[,1]
        v_len = length(v)
        v_zeros = sum(v == 0)
        v_coeffs = sum(v != 0)
        v_frac_zeros = v_zeros/v_len
        return(data.frame(x_penalty=x_penalty,
                          u_len=u_len, u_zeros=u_zeros, u_coeffs=u_coeffs, u_frac_zeros=u_frac_zeros,
                          z_penalty=z_penalty,
                          v_len=v_len, v_zeros=v_zeros, v_coeffs=v_coeffs, v_frac_zeros=v_frac_zeros))
}

print(model_stats(model))


u <- model$u
v <- model$v

print(paste('save output files to:', u_filepath, v_filepath))
write.table(u, file = u_filepath, row.names = FALSE)
write.table(v, file = v_filepath, row.names = FALSE)

