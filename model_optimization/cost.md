Importantly, a cost function **connects the algorithm's θ with its prediction loss (error)**, just like how a person (the optimization method) trying to walk down to the bottom in an error mountain/surface* (the higher the altitude the greater the error) could translate the current thinking (θ) in a thought path (θ's) to the corresponding altitude (the loss/error).

\*The error mountain/surface describes the difference between the ground truth (y) and an algorithm (h<sub>θ</sub>(x)).

In other words, a cost function connects the algorithm's θ with the difference between y and h<sub>θ</sub>(x) (= y_hat = y_pred).

Algorithm | y_pred | Implementation of the cost function, J(θ) = loss<br/>Generally, the idea is (h<sub>θ</sub>(x) - y)<sup>2</sup> | Implementation of the gradient<br/><img src="./images/partial_derivative.png" width="50px">
--- | --- | --- | ---
Linear Regression | <img src="./images/y_hat_linear_regression.png" width="50px"> | <img src="./images/cost_function_linear_regression.png" width="180px"> | <img src="./images/gradient_of_cost_function_linear_or_logistic_regression.png" width="180px">
Ridge Regression;<br>Regularized Linear Regression | --- | --- | ---
Logistic Regression | <img src="./images/y_hat_logistic_regression.png" width="200px"> | Avoid using (h<sub>θ</sub>(x) - y)<sup>2</sup>directly<br/>because it tends to be wavy and non-convex.<br/><br/>Instead, we use cross-entropy, or log loss:<br/><img src="./images/cost_function_logistic_regression_idea.png" width="200px"><br/><br/>or,<br/><img src="./images/cost_function_logistic_regression_implementation.png" width="350px"><br/><br/>or,<br/><img src="./images/cost_function_logistic_regression_implementation_vectorized.png" width="350px"> | <img src="./images/gradient_of_cost_function_linear_or_logistic_regression.png" width="180px"><br/>or,<br/><img src="./images/gradient_of_cost_function_logistic_regression.png" width="180px"><br/>, where g(z) is the sigmoid function
Neural Networks | --- | --- | ---

