import numpy as np
import matplotlib.pyplot as plt 

X, y = np.loadtxt('Linear Regression/data.csv', delimiter=',', skiprows=1, unpack=True)

# ensure X is 2D - (number of samples, number of features)
X = X.reshape(-1, 1)

def display_dataset_graph():
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, color='blue', alpha=0.6, edgecolors='k', label='Data points')
    plt.title('Dataset: X vs Y')
    plt.xlabel('X (Feature)')
    plt.ylabel('Y (Target)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

# split data into training and testing
# shuffle data before splitting
np.random.seed(42)
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]
# 80% train, 20% test
split_index = int(0.8 * len(X))

X_train = X[:split_index]
y_train = y[:split_index]

X_test = X[split_index:] 
y_test = y[split_index:] 

#print(f"training samples: {len(X_train)}") #560
#print(f"testing samples: {len(X_test)}") #140

def initialise_params(X_train):
    """
    initialises parameters for hypothesis function
    input
    X: training data
    output
    params: dictionary containing optimised coefficients
    """
    params = {}
    _, num_features = X_train.shape

    # initialising coefficients to 0
    params['W'] = np.zeros(num_features)
    params['b'] = 0

    return params

def gradient_descent(X_train, y_train , params, alpha, num_iterations):
    """
    batch gradient descent
    output
    params: dictionary containing optimised coefficients
    """
    W = params['W']
    b = params['b']
    num_samples = X_train.shape[0]

    for _ in range(num_iterations):
        # make predictions
        y_pred = np.dot(X_train, W) + b 
        # residual error for each sample
        residuals = y_pred - y_train
        # compute cost (mean squared error)
        #cost = (1 / (2 * num_samples)) * np.sum(residuals**2)
        # compute partial derivatives
        # dW shape: (num_features, 1), db: scalar
        dW = (1 / num_samples) * np.dot(X_train.T, residuals)
        db = (1 / num_samples) * np.sum(residuals)
        # update parameters
        W = W - (alpha * dW)
        b = b - (alpha * db)

    # store updated values
    params['W'] = W
    params['b'] = b
    
    return params

def train(X_train, y_train, alpha=0.0001, num_iterations=1000):
    """
    output
    params: dictionary containing optimised coefficients
    """
    init_params = initialise_params(X_train)
    params = gradient_descent(X_train, y_train, init_params, alpha, num_iterations)
    return params

def predict(X_test, params):
    """
    inputs
    X: unseen data
    params: dictionary contianing optimised weights from training

    outputs
    y_preds: predictions of model
    """  
    y_preds = np.dot(X_test, params["W"]) + params["b"]
    return y_preds


params = train(X_train, y_train) # train model
y_preds = predict(X_test, params) # inference

# testing
print("weights:", params['W'])
print("bias:", params['b'])

# visualise predictions
plt.scatter(X_test[:, 0], y_test, label='Actual data')
sort_idx = np.argsort(X_test[:, 0])
plt.plot(X_test[sort_idx, 0], y_preds[sort_idx], color='red', label='Regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()