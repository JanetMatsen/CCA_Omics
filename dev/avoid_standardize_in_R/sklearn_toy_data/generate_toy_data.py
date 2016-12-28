import numpy as np
from sklearn.preprocessing import StandardScaler


def prepare_data():
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
    #return X_train, Y_train, X_test, Y_test
    np.savetxt('X_train.tsv', X_train, delimiter='\t')
    np.savetxt('Y_train.tsv', Y_train, delimiter='\t')
    np.savetxt('X_test.tsv', X_test, delimiter='\t')
    np.savetxt('Y_test.tsv', Y_test, delimiter='\t')

    # Make a standard scalar version
    ssX = StandardScaler()
    X_train_ss = ssX.fit_transform(X_train)
    ssY = StandardScaler()
    Y_train_ss = ssY.fit_transform(Y_train)

    # print means
    print('column mean of X_train_ss: {}'.format(np.mean(X_train_ss, axis=0)))
    print('column mean of Y_train_ss: {}'.format(np.mean(Y_train_ss, axis=0)))
    # print SDs
    print('column SD of X_train_ss: {}'.format(np.std(X_train_ss, axis=0)))
    print('column SD of Y_train_ss: {}'.format(np.std(Y_train_ss, axis=0)))

    # Also prep the standard scalar versions.
    np.savetxt('X_train_SS.tsv', X_train_ss, delimiter='\t')
    np.savetxt('Y_train_SS.tsv', Y_train_ss, delimiter='\t')
    #np.savetxt('X_test_SS.tsv', X_test_ss, delimiter='\t')
    #np.savetxt('Y_test_SS.tsv', Y_test_ss, delimiter='\t')


if __name__ == "__main__":
    print('save tsv files')
    prepare_data()
