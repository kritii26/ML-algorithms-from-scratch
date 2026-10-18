Linear Regression

Implementation
- Implemented linear regression using NumPy and batch gradient descent.
- Calculated gradients for the weight and bias to minimise mean squared error.

What I have learned
- A learning rate of 0.01 caused gradient descent to diverge on this dataset, producing NaN parameters.
- Experimenting with smaller learning rates showed that excessively large updates can cause divergence, while excessively small updates lead to slow convergence.

Experiments
- Tested different learning rates and observed their effects on convergence and the fitted regression line.

Future Improvements
- Implement feature scaling to improve numerical stability and enable the model to converge faster.