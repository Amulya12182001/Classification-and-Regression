import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat(r"C:\Users\ideal\Downloads\Project 4\Project 4\basecode\mnist_all.mat")  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    # Add bias term to the input data
    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    train_data = np.hstack((np.ones((n_data, 1)), train_data))

    # Reshape initialWeights to a column vector
    initialWeights = initialWeights.reshape((n_features + 1, 1))

    # Compute the sigmoid of the linear combination
    z = np.dot(train_data, initialWeights)
    theta = sigmoid(z)
    epsilon = 1e-5
    theta = np.clip(theta, epsilon, 1 - epsilon)

    # Error computation using the negative log-likelihood
    error = -np.sum(labeli * np.log(theta) + (1 - labeli) * np.log(1 - theta))

    # Error Gradiant computation
    error_grad = np.dot(train_data.T, (theta - labeli)).flatten()

    return error, error_grad


def blrPredict(W, data):
    # Add bias term to the input data
    n_data = data.shape[0]
    data = np.hstack((np.ones((n_data, 1)), data))

    # Computation of the class probabilities
    probabilities = sigmoid(np.dot(data, W))
    label = np.argmax(probabilities, axis=1).reshape((n_data, 1))

    return label

def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]

    # Add bias term to the input data
    train_data = np.hstack((np.ones((n_data, 1)), train_data))

    # Reshape params to a weight matrix
    params = params.reshape((n_feature + 1, n_class))

    # Computation of the class probabilities using softmax
    z = np.dot(train_data, params)
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # for numerical stability
    softmax_probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # Error computation using cross-entropy loss
    error = -np.sum(labeli * np.log(softmax_probs))

    # Gradient of the error computation
    error_grad = np.dot(train_data.T, (softmax_probs - labeli)).flatten()

    return error, error_grad


def mlrPredict(W, data):
    n_data = data.shape[0]
    data = np.hstack((np.ones((n_data, 1)), data))

    # Computation of the class probabilities using softmax
    z = np.dot(data, W)
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # for numerical stability
    softmax_probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    label = np.argmax(softmax_probs, axis=1).reshape((n_data, 1))

    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights.flatten(), jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

# Calculate and record total error for each category in training data
train_errors_ova = np.zeros(n_class)
predicted_label_train = blrPredict(W, train_data)
for i in range(n_class):
    train_errors_ova[i] = np.sum(predicted_label_train[train_label == i] != i)

# Calculate and record total error for each category in test data
test_errors_ova = np.zeros(n_class)
predicted_label_test = blrPredict(W, test_data)
for i in range(n_class):
    test_errors_ova[i] = np.sum(predicted_label_test[test_label == i] != i)

# Total number of samples in training and test datasets
n_train_samples = train_data.shape[0]
n_test_samples = test_data.shape[0]

# Calculate errors in percentage
train_errors_percentage = (train_errors_ova / n_train_samples) * 100
test_errors_percentage = (test_errors_ova / n_test_samples) * 100

# Print errors in percentage
print(f'\n Logistic Regression - Training Errors per Class (%): {train_errors_percentage}')
print(f'Logistic Regression - Testing Errors per Class (%): {test_errors_percentage}')

# Calculate overall training and testing errors for one-vs-all logistic regression
overall_train_error_ova = np.sum(train_errors_ova) / n_train_samples * 100
overall_test_error_ova = np.sum(test_errors_ova) / n_test_samples * 100

print(f'\n Logistic Regression (One-vs-All) - Overall Training Error: {overall_train_error_ova:.2f}%')
print(f'Logistic Regression (One-vs-All) - Overall Testing Error: {overall_test_error_ova:.2f}%')

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################

# Linear Kernel
svm_linear = SVC(kernel='linear')
svm_linear.fit(train_data, train_label.ravel())
train_acc_linear = svm_linear.score(train_data, train_label) * 100
val_acc_linear = svm_linear.score(validation_data, validation_label) * 100
test_acc_linear = svm_linear.score(test_data, test_label) * 100
print(f'Linear Kernel - Training Accuracy: {train_acc_linear:.2f}%')
print(f'Linear Kernel - Validation Accuracy: {val_acc_linear:.2f}%')
print(f'Linear Kernel - Testing Accuracy: {test_acc_linear:.2f}%')

# Radial Basis Kernel with Gamma = 1
svm_rbf_gamma1 = SVC(kernel='rbf', gamma=1)
svm_rbf_gamma1.fit(train_data, train_label.ravel())
train_acc_rbf_gamma1 = svm_rbf_gamma1.score(train_data, train_label) * 100
val_acc_rbf_gamma1 = svm_rbf_gamma1.score(validation_data, validation_label) * 100
test_acc_rbf_gamma1 = svm_rbf_gamma1.score(test_data, test_label) * 100
print(f'RBF Kernel when gamma=1 - Training Accuracy: {train_acc_rbf_gamma1:.2f}%')
print(f'RBF Kernel when gamma=1 - Validation Accuracy: {val_acc_rbf_gamma1:.2f}%')
print(f'RBF Kernel when gamma=1 - Testing Accuracy: {test_acc_rbf_gamma1:.2f}%')

# Radial Basis Kernel with Default Gamma
svm_rbf_default = SVC(kernel='rbf')
svm_rbf_default.fit(train_data, train_label.ravel())
train_acc_rbf_default = svm_rbf_default.score(train_data, train_label) * 100
val_acc_rbf_default = svm_rbf_default.score(validation_data, validation_label) * 100
test_acc_rbf_default = svm_rbf_default.score(test_data, test_label) * 100
print(f'RBF Kernel with default gamma - Training Accuracy: {train_acc_rbf_default:.2f}%')
print(f'RBF Kernel with default gamma - Validation Accuracy: {val_acc_rbf_default:.2f}%')
print(f'RBF Kernel with default gamma - Testing Accuracy: {test_acc_rbf_default:.2f}%')

# Radial Basis Kernel with Varying C
C_values = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
train_accuracies = []
val_accuracies = []
test_accuracies = []

for C in C_values:
    svm_rbf_varying_C = SVC(kernel='rbf', C=C)
    svm_rbf_varying_C.fit(train_data, train_label.ravel())
    train_accuracies.append(svm_rbf_varying_C.score(train_data, train_label) * 100)
    val_accuracies.append(svm_rbf_varying_C.score(validation_data, validation_label) * 100)
    test_accuracies.append(svm_rbf_varying_C.score(test_data, test_label) * 100)

# Plotting the accuracies
plt.figure(figsize=(10, 6))
plt.plot(C_values, train_accuracies, label='Training Accuracy')
plt.plot(C_values, val_accuracies, label='Validation Accuracy')
plt.plot(C_values, test_accuracies, label='Testing Accuracy')
plt.xlabel('C Value')
plt.ylabel('Accuracy (%)')
plt.title('SVM Accuracy with Varying C Values')
plt.legend()
plt.grid(True)
plt.show()

"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
initialWeights_b = np.array(initialWeights_b).flatten()
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')

# Calculate and print total error for multi-class logistic regression
train_errors_mlr = np.zeros(n_class)
test_errors_mlr = np.zeros(n_class)

# Use the correct predicted_label for training data
predicted_label_train_b = mlrPredict(W_b, train_data)

for i in range(n_class):
    train_errors_mlr[i] = np.sum(predicted_label_train_b[train_label == i] != i)
    test_errors_mlr[i] = np.sum(predicted_label_b[test_label == i] != i)

# Total number of samples in training and test datasets
n_train_samples = train_data.shape[0]
n_test_samples = test_data.shape[0]

# Calculate errors in percentage
train_errors_mlr_percentage = (train_errors_mlr / n_train_samples) * 100
test_errors_mlr_percentage = (test_errors_mlr / n_test_samples) * 100

# Print errors in percentage
print(f'\n Multi-class Logistic Regression - Training Errors per Class (%): {train_errors_mlr_percentage}')
print(f'Multi-class Logistic Regression - Testing Errors per Class (%): {test_errors_mlr_percentage}')

# Calculate overall training and testing errors for multi-class logistic regression
overall_train_error_mlr = np.sum(train_errors_mlr) / n_train_samples * 100
overall_test_error_mlr = np.sum(test_errors_mlr) / n_test_samples * 100

print(f'\n Multi-class Logistic Regression - Overall Training Error: {overall_train_error_mlr:.2f}%')
print(f'Multi-class Logistic Regression - Overall Testing Error: {overall_test_error_mlr:.2f}%')

# Compare performance
print('\nPerformance Comparison:')
for i in range(n_class):
    print(f'Class {i}:')
    print(f'  One-vs-All Training Error: {train_errors_percentage[i]}, Multi-class Training Error: {train_errors_mlr_percentage[i]}')
    print(f'  One-vs-All Testing Error: {test_errors_percentage[i]}, Multi-class Testing Error: {test_errors_mlr_percentage[i]}')
