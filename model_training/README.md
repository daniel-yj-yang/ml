## 1. ROC curve

<p align="center"><img src="./images/prob_distribution_and_ROC.gif" width="600px"><br/>Ideally, a ML classification algorithm would improve over time via training, resulting in greater predictive power, namely, a cleaner separation of the y probability distributions of True Positive vs. True Negative, given X's (<a href="http://arogozhnikov.github.io/2015/10/05/roc-curve.html">website reference</a>)</p>

<hr>

## 2. <a href="https://en.wikipedia.org/wiki/Gradient_descent">Gradient Descent</a>

A **first-order** (namely, an algorithm that requires at least one first-derivative/gradient) iterative optimization algorithm that finds the weights or coefficients that reach a local minimum of a differentiable function.

#### Step#1: The model make predictions on training data.
#### Step#2: Using the error on the predictions to update the model in the direction of reducing the error.

Analogy | Gradient Descent
--- | ---
1.&nbsp;A person, who is stuck in the mountain and trying to get down | the algorithm, which is somewhere in the error surface and trying to find the global minimum
2.&nbsp;Path taken down the mountain | The sequence of parameter settings that the algorithm will explore
3.&nbsp;The steepness of the hill (the direction to travel is the steepest descent) | the slope/gradient of the error surface at that point
4.&nbsp;The instrument used to measure steepness | differentiation (derivative) of the cost function
5.&nbsp;The amount of time they travel before taking another measurement | the learning rate of the algorithm
6.&nbsp;The altitude at the person's location | the error the model make when predicting on training data

<p align="center"><img src="./images/gradient_descent.png" width="500px"></p>

<hr>

Approach | ---
--- | ---
<a href="./stochastic_gradient_descent">Stochastic Gradient Descent</a> (SGD) | Use a randomly selected subset of the data (rather than the entire data set) to estimate the gradient. Doing so helps reduce computational burden and achieve faster iterations, especially in high-dimension problems, although it may converge more slowly.
<a href="./batch_gradient_descent">Batch Gradient Descent</a> | ---
<a href="./mini_batch_gradient_descent">Mini-Batch Gradient Descent</a> | ---

