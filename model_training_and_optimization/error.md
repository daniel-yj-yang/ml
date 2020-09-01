Cost function measures the difference between y and y_pred (= y_hat = h<sub>θ</sub>(x))

Algorithm | y_pred | General idea of the cost function, J(θ)<br/>(y_pred - y)<sup>2</sup> | Implementation of J(θ)
--- | --- | --- | ---
Linear Regression | <img src="./images/y_hat_linear_regression.png" width="50px"> | (y_pred - y)<sup>2</sup> | <img src="./images/cost_function_linear_regression.png" width="180px">
Logistic Regression | <img src="./images/y_hat_logistic_regression.png" width="350px"> | We cannot just use (y_pred - y)<sup>2</sup> because it's wavy and non-convex.<br/><br/>Instead, we use cross-entropy, or log loss:<br/><img src="./images/cost_function_logistic_regression_idea.png" width="200px"> | <img src="./images/cost_function_logistic_regression_implementation.png" width="400px">
Neural Networks | --- | --- | ---
