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

