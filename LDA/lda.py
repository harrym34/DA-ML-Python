"""
LINEAR DISCRIMINANT ANALYSIS IN PYTHON

Harry Michell, November 2023

Overview:
The file lda.py contains a set of functions and a main program for performing 
Linear Discriminant Analysis (LDA) on two datasets related to diabetes (DM) 
and obesity (OB). The script reads data from a CSV file, preprocesses it, 
performs LDA, plots LDA axes with data, computes and plots ROC curves, 
and displays confusion matrices with optimal thresholds and classification accuracy.

Dependencies:
- numpy
- matplotlib
- scikit-learn

Notes:
 - The script assumes that the input data is in the file 'dmrisk.csv'.
 - Adjustments to the pause time (plt.pause()) may be necessary based 
   on the system's rendering speed.
 - The script uses PCA for dimensionality reduction before applying LDA. 
   Adjust the number of components as needed (n_components in PCA).
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix


"""
Computes the LDA axis and scores for each dataset using the BinaryLDA
function.

Parameters:
- Xmat1: Data matrix for the first dataset.
- yvec1: Labels for the first dataset.
- Xmat2: Data matrix for the second dataset.
- yvec2: Labels for the second dataset.

Returns:
- q1: LDA axis for the first dataset.
- z1: Scores for the first dataset along the LDA axis.
- q2: LDA axis for the second dataset.
- z2: Scores for the second dataset along the LDA axis.
"""
def LDAAxis(Xmat1, yvec1, Xmat2, yvec2):
    q1 = np.array([])
    z1 = np.array([])
    q2 = np.array([])
    z2 = np.array([])

    # Compute the LDA axis for each data set
    q1 = BinaryLDA(Xmat1[yvec1 == 1, :], Xmat1[yvec1 != 1, :])
    q2 = BinaryLDA(Xmat2[yvec2 == 1, :], Xmat2[yvec2 != 1, :])

    # Score data using LDA axes
    z1 = np.dot(Xmat1 - np.mean(Xmat1, axis=0), q1)
    z2 = np.dot(Xmat2 - np.mean(Xmat2, axis=0), q2)

    return q1, z1, q2, z2


"""
Performs binary LDA to compute the LDA axis for given class data.

Parameters:
- X1: Data matrix for one class.
- X2: Data matrix for another class.

Returns:
- qvec: LDA axis.
"""
def BinaryLDA(X1, X2):
    # Initialize return variable
    qvec = np.ones((X1.shape[1], 1))

    # Means for data
    xbar1 = np.mean(X1, axis=0)
    xbar2 = np.mean(X2, axis=0)

    # Compute within-class means and scatter matrices
    M1 = X1 - xbar1
    M2 = X2 - xbar2

    # Scatter matrices
    S1 = M1.T @ M1
    S2 = M2.T @ M2

    # Within-label scatter
    SW = S1 + S2

    # Means for both classes stored in one vector
    Mb = np.vstack((xbar1, xbar2))

    # Compute between-class scatter matrix
    SB = (Mb - np.mean(Mb, axis=0)).T @ (Mb - np.mean(Mb, axis=0))

    # Fisher's linear discriminant is the largest
    # eigenvector of the Rayleigh quotient
    l, E = np.linalg.eig(np.linalg.inv(SW) @ SB)

    # Sort eigenvalues
    ndx = np.argsort(l)[::-1]

    # Find qvec
    Emat = E[:, ndx]
    qvec = Emat[:, 0]

    # May need to correct the sign of qvec to point towards
    # the mean of X1
    if np.dot((xbar1 - xbar2), qvec) < 0:
        qvec = -qvec

    return qvec


"""
Displays a scatter plot with LDA axes.

Parameters:
- data: Data matrix.
- lda_axis: LDA axis.
- labels: Data labels.
- title: Title for the plot.
"""
def plot_lda_axes(data, lda_axis, labels, title):
    plt.figure()
    cmap = plt.get_cmap("Set1")  # Choose a colormap, e.g., "Set1"
    normalize = plt.Normalize(labels.min(), labels.max())
    colors = cmap(normalize(labels))

    plt.scatter(data[:, 0], data[:, 1], c=colors, cmap=cmap, label="Data", alpha=0.7)

    # Plot the LDA axis
    origin = np.mean(data, axis=0)
    plt.quiver(*origin, lda_axis[0], lda_axis[1], color='red', scale=3, label="LDA Axis")

    plt.title(title)
    plt.legend()

"""
Computes the optimal threshold using the Youden's Index.

Parameters:
- y_true: True class labels.
- scores: Scores obtained from the classifier.

Returns:
- optimal_threshold: Optimal threshold for binary classification.
"""
def find_optimal_threshold(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    youden_index = tpr - fpr
    optimal_threshold_index = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_threshold_index]
    return optimal_threshold



"""
Reads data from a CSV file, preprocesses it, performs PCA for dimensionality 
reduction, computes LDA axes, plots LDA axes with data, computes and plots 
ROC curves, and displays confusion matrices with optimal thresholds and 
classification accuracy.
"""
def main():
    # Read the test data from a CSV file
    dmrisk = np.genfromtxt('dmrisk.csv', delimiter=',', skip_header=1)

    # Columns for the data and labels; DM is diabetes, OB is obesity
    jDM = 16  # Subtract 1 to make it 0-based
    jOB = 15  # Subtract 1 to make it 0-based

    # Extract the data matrices and labels
    XDM = np.delete(dmrisk, jDM, axis=1)
    yDM = (dmrisk[:, jDM] == 1).astype(int)  # Convert to binary

    XOB = np.delete(dmrisk, jOB, axis=1)
    yOB = (dmrisk[:, jOB] == 1).astype(int)  # Convert to binary

    # Reduce the dimensionality to 2D using PCA
    scaler = StandardScaler()
    XDM_scaled = scaler.fit_transform(XDM)
    rDM = PCA(n_components=2).fit_transform(XDM_scaled)

    XOB_scaled = scaler.fit_transform(XOB)
    rOB = PCA(n_components=2).fit_transform(XOB_scaled)

    # Find the LDA vectors and scores for each data set
    qDM, zDM, qOB, zOB = LDAAxis(rDM, yDM, rOB, yOB)

    # Plot LDA axes with the data
    plot_lda_axes(rDM, qDM, yDM, "LDA Axis and Data (Diabetes)")
    plot_lda_axes(rOB, qOB, yOB, "LDA Axis and Data (Obesity)")

    # Display all figures at once
    plt.show(block=False)
    
    # Adjust the pause time as needed
    plt.pause(1)  

    # Compute the ROC curve and its AUC
    fprDM, tprDM, _ = roc_curve(yDM, zDM)
    aucDM = auc(fprDM, tprDM)

    fprOB, tprOB, _ = roc_curve(yOB, zOB)
    aucOB = auc(fprOB, tprOB)

    # Plot ROC curves
    plt.figure()
    plt.plot(fprDM, tprDM, 'b', [0, 1], [0, 1], 'r')
    plt.title("ROC Curve for Diabetes Data")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.text(0.6, 0.3, f'AUC = {aucDM:.4f}')
    plt.show()

    plt.figure()
    plt.plot(fprOB, tprOB, 'b', [0, 1], [0, 1], 'r')
    plt.title("ROC Curve for Obesity Data")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.text(0.6, 0.3, f'AUC = {aucOB:.4f}')
    plt.show()

    # Display confusion matrices, optimal thresholds, and accuracy
    threshold_DM = find_optimal_threshold(yDM, zDM)
    threshold_OB = find_optimal_threshold(yOB, zOB)

    confDM = confusion_matrix(yDM, zDM > threshold_DM)
    confOB = confusion_matrix(yOB, zOB > threshold_OB)
    
    accuracy_DM = (confDM[0, 0] + confDM[1, 1]) / np.sum(confDM)
    accuracy_OB = (confOB[0, 0] + confOB[1, 1]) / np.sum(confOB)

    print("Diabetes Confusion Matrix:")
    print(confDM)
    print(f"Optimal Threshold for Diabetes: {threshold_DM}")
    print(f"Diabetes Classification Accuracy: {accuracy_DM}")

    print("\nObesity Confusion Matrix:")
    print(confOB)
    print(f"Optimal Threshold for Obesity: {threshold_OB}")
    print(f"Obesity Classification Accuracy: {accuracy_OB}")


if __name__ == "__main__":
    main()

