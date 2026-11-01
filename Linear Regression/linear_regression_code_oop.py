class LinearRegression:

  def __init__(self, dataset):
    self.X = dataset[:, :-1] # all rows and columns except the last column which is the target/label
    # : selects all rows, :-1 selects all columns except last one
    self.y = dataset[:, -1] # last column
    # : selects all rows, -1 selects only the last item in each row

    # split data into training and testing
    np.random.seed(42) # shuffle data before splitting 
    indices = np.random.permutation(len(self.X))
    self.X = self.X[indices]
    self.y = self.y[indices]
    split_index = int(0.8 * len(self.X)) # 80% train, 20% test

    self.X_train = self.X[:split_index]
    self.y_train = self.y[:split_index]
    self.X_test = self.X[split_index:]
    self.y_test = self.y[split_index:]

    self.alpha = 0.0001
    self.num_iterations = 1000

    self.params = {}
  
  def initialise_params(self,X_train):
    """
    initialises parameters for hypothesis function
    input
    X: training data
    output
    params: dictionary containing the initialised coefficients
    """
    _, num_features = X_train.shape

    # initialise coefficients to 0
    self.params['W'] = np.zeros(num_features)
    self.params['b'] = 0

    return self.params

  def gradient_descent(self, X_train, y_train):
    """
    batch gradient descent
    output
    params: dictionary containing optimised coefficients
    """
    W = self.params['W']
    b = self.params['b']
    num_samples = X_train.shape[0]

    for _ in range(self.num_iterations):
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
        W = W - (self.alpha * dW)
        b = b - (self.alpha * db)

    # store updated values
    self.params['W'] = W
    self.params['b'] = b

    return self.params

  def train(self, X_train, y_train):
    self.initialise_params(X_train)
    self.gradient_descent(X_train, y_train)
    return self.params

  def predict(self, X_test):
    """
    inputs
    X_test: unseen data, params: optimised weights from training
    outputs
    y_preds: predictions of model
    """
    y_preds = np.dot(X_test, self.params['W']) + self.params["b"] # y = mx + c
    return y_preds

  def visualise_predictions(self):
    y_preds = self.predict(self.X_test)
    plt.scatter(self.y_test, y_preds)

    min_val = min(self.y_test.min(), y_preds.min())
    max_val = max(self.y_test.max(), y_preds.max())

    plt.plot([min_val, max_val], [min_val, max_val], color='red')

    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.title("Actual vs Predicted Values")
    plt.show()

def linear_regression(dataset):
  model = LinearRegression(dataset)

  params = model.train(model.X_train, model.y_train) # train model
  y_preds = model.predict(model.X_test) # inference
  mse = np.mean((model.y_test - y_preds) ** 2) # cost
  r_squared = 1 - (
      np.sum((model.y_test - y_preds)**2) /  
      np.sum((model.y_test - np.mean(model.y_test))**2)
      )

  print("weights:", params['W'])
  print("bias:", params['b'])
  print("MSE:", mse)
  print("R^2 score", r_squared)

linear_regression(dataset)

