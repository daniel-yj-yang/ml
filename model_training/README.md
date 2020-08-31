## ROC curve

<p align="center"><img src="./images/prob_distribution_and_ROC.gif" width="600px"><br/>Ideally, a ML classification algorithm would improve over time via training, resulting in greater predictive power, namely, a cleaner separation of the y probability distributions of True Positive vs. True Negative, given X's (<a href="http://arogozhnikov.github.io/2015/10/05/roc-curve.html">website reference</a>)</p>

<hr>

## <a href="./gradient_descent">Gradient Descent</a>

A **first-order** (namely, an algorithm that requires at least one first-derivative/gradient) iterative optimization algorithm that finds a local minimum of a differentiable function.

Analogy | Gradient Descent
--- | ---
A person, who is stuck in the mountain and trying to get down | the algorithm, which is somewhere in the error surface and trying to find the global minimum
Path taken down the mountain | The sequence of parameter settings that the algorithm will explore
The steepness of the hill (the direction to travel is the steepest descent) | the slope/gradient of the error surface at that point
The instrument used to measure steepness | differentiation of the error function
The amount of time they travel before taking another measurement | the learning rate of the algorithm

<p align="center"><img src="./images/gradient_descent.png" width="500px"></p>
