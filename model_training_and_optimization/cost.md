Importantly, cost function **connects the algorithm's θ with its prediction loss (error)**, just like how a person (the optimization method) trying to walk down a path (θ's) in an error mountain/surface* (the higher the altitude the greater the error) could translate the location (θ) in a path (θ's) to the corresponding altitude (the loss/error).

\*The error mountain/surface describes the relationship between an algorithm and the ground truth.

In other words, cost function measures the difference between y and y_pred (= y_hat = h<sub>θ</sub>(x)).

Algorithm | y_pred | Implementation of the cost function, J(θ) = loss<br/>Generally, the idea is (y_pred - y)<sup>2</sup> | Implementation of the gradient<br/><img src="./images/partial_derivative.png" width="50px">
--- | --- | --- | ---
Linear Regression | <img src="./images/y_hat_linear_regression.png" width="50px"> | <img src="./images/cost_function_linear_regression.png" width="180px"> | <img src="./images/gradient_of_cost_function_linear_or_logistic_regression.png" width="180px">
Logistic Regression | <img src="./images/y_hat_logistic_regression.png" width="200px"> | Avoid using (y_pred - y)<sup>2</sup>directly<br/>because it tends to be wavy and non-convex.<br/><br/>Instead, we use cross-entropy, or log loss:<br/><img src="./images/cost_function_logistic_regression_idea.png" width="200px"><br/><br/>or,<br/><img src="./images/cost_function_logistic_regression_implementation.png" width="350px"><br/><br/>or,<br/><img src="./images/cost_function_logistic_regression_implementation_vectorized.png" width="350px"> | <img src="./images/gradient_of_cost_function_linear_or_logistic_regression.png" width="180px"><br/>or,<br/><img src="./images/gradient_of_cost_function_logistic_regression.png" width="180px"><br/>, where g(z) is the sigmoid function
Neural Networks | --- | --- | ---

