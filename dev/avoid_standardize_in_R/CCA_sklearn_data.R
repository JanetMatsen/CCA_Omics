library(PMA)

standard_scalar=TRUE
#standard_scalar=FALSE

if (standard_scalar){
        x = read.csv('./sklearn_toy_data/X_train_SS.tsv', sep = '\t', header = FALSE)
        z = read.csv('./sklearn_toy_data/Y_train_SS.tsv', sep = '\t', header = FALSE)
} else {
        x = read.csv('./sklearn_toy_data/X_train.tsv', sep = '\t', header = FALSE)
        z = read.csv('./sklearn_toy_data/Y_train.tsv', sep = '\t', header = FALSE)
}

x[0:4, 0:4]

# Check whether it is zero-mean and unit variance:
colMeans(x)
colMeans(z)
# standard deviations
apply(x, 2, sd)
apply(z, 2, sd)

cca = CCA(x = x, z = z, K = 1,
          typex = 'standard', typez = 'standard',
          standardize = FALSE
          #standardize = TRUE
)

ss_version <- function(v){
        sd <- apply(v, 2, sd)
        if (min(sd) == 0)
                stop("Cannot standardize because some of the columns of x have std. dev. 0")
        return(scale(v, TRUE, sd))
}

x_ss_R <- ss_version(x)

class(x) # data.frame
class(x_ss_R)  # matrix

x[0:4, 0:4]
x_ss_R[0:4, 0:4]

# Convert the data.frame to matrix and the SVD error disappears.
x_matrix <- data.matrix(x)
z_matrix <- data.matrix(z)

# now this works!
cca = CCA(x = x_matrix, z = z_matrix, K = 1,
          typex = 'standard', typez = 'standard',
          standardize = FALSE)

# Conclusion:  there is a bug in the PMA package.
# It should convert data.frame instances to matrices or at least require you to provide a matrix.

