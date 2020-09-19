# Decision tree
To predict consumer's purchasing decision via asking a series of questions honing in the final answer

<hr>

## Online Demo

<a href="https://github.com/daniel-yj-yang/ml/blob/master/decision_trees/DT_Purchasing.ipynb">Classification of user purchasing (yes/no) using age and estimated salary</a>

<hr>

## Concepts

An analogy of decision trees is the <a href="https://en.wikipedia.org/wiki/Twenty_Questions">20 Questions</a> game, where a questioner can ask an answerer 20 yes/no questions to guess what the answerer is thinking of. A simplified version is to guess a number between 1-100. The questioner would ask a series of questions: is it larger than 50? is it larger than 75? is it larger than 87.5? and so on, until the questioner can tell the answer with increased certainty.

In ML, the process of a decision tree is to <b><i>recursively</i></b> split down from the root node to one of the leaf/terminal nodes, with every node acting as a test case for making <b><i>increasingly more certain/accurate</i></b> decisions based on the values of predictors.

<p align="center"><img src="https://github.com/daniel-yj-yang/ML-retail-analytics/blob/master/decision_trees/concept-1.jpg" width="600px"></p>

Criterion | Equation | Descriptions | Comparison
--- | --- | --- | ---
Gini index of impurity | <p><img src="https://github.com/daniel-yj-yang/ML-retail-analytics/blob/master/decision_trees/Gini_index_formula.png" width="200px"></p><i>p</i><sub>i</sub> is the proportion of samples belonging to class <i>c</i> for a particular node | * Gini impurity, the amount of probability of a specific feature that would be classified incorrectly if randomly selected, and 1 ≥ Gini index ≥ 0;<br />* Used in <a href="https://en.wikipedia.org/wiki/Decision_tree_learning">CART (Classification and Regression Tree)</a> algorithm | * Favors larger partitions<br/>* Used in classification
Entropy | <p><img src="https://github.com/daniel-yj-yang/ML-retail-analytics/blob/master/decision_trees/Entropy_formula.png" width="300px"></p> | * Entropy is how much randomness/guessing it'd require to determine the class, and 1 ≥ Entropy ≥ 0;<br />* Entropy decrease is information gain; how well the value of predictor can separate the training samples with respect to the target class | * Favors smaller partitions with distinct values<br/>* Used in classification
Variance Reduction | <p><img src="https://github.com/daniel-yj-yang/ML-retail-analytics/blob/master/decision_trees/Variance_formula.png" width="300px"></p> | * Among all possible splits to split the population, the one with lower variance is selected | * Used in regression

<hr> 

## Process of generating a tree

The root node is the entire training set. The values of the predictor is binarized. The node will keep splitting until its Gini index or Entropy ≈ 0, that is, requiring little or no uncertainty to determine the class.

Here is a max_depth=2 decision tree example of predicting Purchased by Age and Estimated Salary using the Gini index:
<p align="center"><img src="https://github.com/daniel-yj-yang/ML-retail-analytics/blob/master/decision_trees/DT_Purchase_maxdepth=2.png" width="600px"></p>
<p align="center">Note 1: A node of lighter color indicates higher impurity, which requires further splitting</p>
<p align="center">Note 2: <i>Value</i> indicates how the sample would be split if the node were the last node</p>

<hr>

## Reducing over-fitting in decision trees

Method | Descriptions
--- | ---
<a href="https://en.wikipedia.org/wiki/Random_forest">Random Forest</a> | * An ensemble method.<br />* The training set is randomly sampled, and the features used to split nodes are also randomly sampled.<br />* The sampling method includes <b>bagging</b> and <b>boosting</b>.<p><img src="https://miro.medium.com/max/620/1*WcgEmCuaFr6DsJhHzKi30Q.png"></p>
Pruning | * Split the actual training set into two sets, T<sub>1</sub> and T<sub>2</sub>.<br />* Prepare the tree using T<sub>1</sub> and prune the tree to optimize the accuracy of the validating set T<sub>2</sub>
AdaBoost | <a href="https://en.wikipedia.org/wiki/AdaBoost#:~:text=AdaBoost%2C%20short%20for%20Adaptive%20Boosting,learning%20algorithms%20to%20improve%20performance.">Adaptive Boosting</a>

<hr>
