"""
CONTRAINED LEAST SQUARES IN PYTHON

Harry Michell, November 2023

Overview:
The file cls.py contains a set of functions and a main program for performing 
Constrained Least Squares (CLS) linear fitting on a dataset. 

Dependencies:
- numpy
- matplotlib
- scipy

Notes:
 - The script assumes that the input data is generated using the clsdata function.
 - Adjust the threshold (theta) as needed for CLS.
 - The script uses random number generation for k-fold cross-validation; 
   you may adjust the seed for reproducibility (np.random.seed()).

   Enjoy!
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

"""
Generate data, perform regression, compute 5-fold cross validation
"""

def main():
    # Get data
    xvec, yvec = clsdata()

    # Append 1's to create the design matrix
    xmat = np.column_stack((xvec, np.ones_like(xvec)))

    # Hyper-parameter theta is the maximum norm allowed of the solution vector w*. 
    # Changing this will drastically change the performance of the model.
    theta = 8

    #
    # Compare OLS and CLS on all of the data
    #

    # Compute the ordinary least squares from the normal equation
    w_ols = np.linalg.inv(xmat.T @ xmat) @ xmat.T @ yvec

    # Compute the constrained least squares solution
    w_cls, _ = cls(xmat, yvec, theta)

    # Plot: data, OLS fit, CLS fit
    plt.plot(xvec, yvec, 'k*', label='Data')
    plt.plot(xvec, np.polyval(w_ols, xvec), 'r-', linewidth=1.5, label='OLS Fit')
    plt.plot(xvec, np.polyval(w_cls, xvec), 'b-', linewidth=1.5, label='CLS Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title(f'CLS fit: $\\|\\vec{{w}}\\|^2 = {np.linalg.norm(w_cls)**2:.2f}$', fontsize=14)
    plt.show()

    # Part (B): compare 10 sets of 5-fold validation for OLS and CLS

    # Set the number of repetitions and number of folds
    nreps = 10
    k = 5

    # Set up vectors to collect results
    ols_train_vec = np.zeros(nreps)
    ols_test_vec = np.zeros(nreps)
    cls_train_vec = np.zeros(nreps)
    cls_test_vec = np.zeros(nreps)

    # Set the Random Number Generator for debugging
    np.random.seed(42)

    # Run k-fold on OLS by setting theta=0
    for i in range(nreps):
        ols_train_vec[i], ols_test_vec[i] = clskfold(xmat, yvec, 0, 5)

    # Re-set the RNG and run k-fold on CLS
    np.random.seed(42)
    for i in range(nreps):
        cls_train_vec[i], cls_test_vec[i] = clskfold(xmat, yvec, theta, 5)

    # Compute means and standard deviations of results (terse code?)
    ols_train_mean, ols_train_std_dev = np.mean(ols_train_vec), np.std(ols_train_vec)
    ols_test_mean, ols_test_std_dev = np.mean(ols_test_vec), np.std(ols_test_vec)
    cls_train_mean, cls_train_std_dev = np.mean(cls_train_vec), np.std(cls_train_vec)
    cls_test_mean, cls_test_std_dev = np.mean(cls_test_vec), np.std(cls_test_vec)

    # Display the results
    print('   OLS results are\n     TRAIN     TEST')
    print(np.column_stack((ols_train_vec, ols_test_vec)))
    print(f'   OLS means and std. dev. are\n    {ols_train_mean:.4f}    {ols_test_mean:.4f}\n'
          f'    {ols_train_std_dev:.4f}    {ols_test_std_dev:.4f}\n\n'
          f'   CLS results are\n     TRAIN     TEST')
    print(np.column_stack((cls_train_vec, cls_test_vec)))
    print(f'   CLS means and std. dev. are\n    {cls_train_mean:.4f}    {cls_test_mean:.4f}\n'
          f'    {cls_train_std_dev:.4f}    {cls_test_std_dev:.4f}')

"""
Custom k-fold cross-validation function
"""
def clskfold(xmat, yvec, theta, k_in):

    # Problem size
    M = xmat.shape[0]

    # Set the number of folds; must be 1 < k < M
    if k_in is not None:
        k = max(min(round(k_in), M - 1), 2)
    else:
        k = 5

    # Randomly assign the data into k folds; discard any remainders
    one_to_M = np.arange(1, M + 1)
    Mk = M // k
    ndxmat = np.reshape(np.random.permutation(M)[:Mk * k], (k, Mk))

    # To compute RMS of fit and prediction, we will sum the variances
    var_train = 0.0
    var_test = 0.0

    # Process each fold
    for ix in range(k):
        ndxtrain = np.reshape(ndxmat[np.nonzero(np.arange(1, k + 1) != ix + 1)[0], :], (Mk * (k - 1),))
        ndxtest = np.reshape(ndxmat[ix, :], (Mk,))
        xmat_train, yvec_train = xmat[ndxtrain, :], yvec[ndxtrain]
        xmat_test, yvec_test = xmat[ndxtest, :], yvec[ndxtest]
        w_cls, _ = cls(xmat_train, yvec_train, theta)
        var_train += np.mean((xmat_train @ w_cls - yvec_train) ** 2)
        var_test += np.mean((xmat_test @ w_cls - yvec_test) ** 2)

    rmstrain = np.sqrt(var_train / k)
    rmstest = np.sqrt(var_test / k)
    return rmstrain, rmstest

"""
Constrained Least Squares (CLS) Function: Performs CLS computations
"""
def cls(xmat, yvec, theta):

    # Return immediately if the threshold is invalid
    if theta < 0:
        return None, None

    # Set up the problem as xmat*w=yvec
    Im = np.eye(xmat.shape[1])

    # Set up w and g functions for CLS computations
    wfun = lambda lval: np.linalg.inv(xmat.T @ xmat + lval * Im) @ xmat.T @ yvec
    gfun = lambda lval: np.linalg.norm(wfun(lval))**2 - theta

    # OLS solution: use pseudo-inverse for ill conditioned matrix
    wls = np.linalg.pinv(xmat) @ yvec

    # The OLS solution is used if it is within the user's threshold
    if np.linalg.norm(wls)**2 <= theta or theta <= 0:
        return wls, 0
    else:

        # Estimate lambda using gfun, wfun, and fzero.
        lambda_val = fsolve(gfun, 0)
        w_cls = wfun(lambda_val)
        return w_cls, lambda_val


def clsdata():
    # x values are equally spaced
    xvec = np.linspace(0, 9, 10)
    ylin = np.exp(1) * xvec + np.pi
    
    # Y are linear, deviating first and last
    # Using 1D arrays for concatenation
    yvec = np.concatenate(([ylin[0] - 5], ylin[1:-1], [ylin[-1] + 3]))
    return xvec, yvec


if __name__ == "__main__":
    main()
