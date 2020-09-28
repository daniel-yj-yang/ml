# Machine Learning
<b>Learning the (hidden or obvious) <a href="./regularity/">regularity of the world</a> via computer algorithms that <a href="./model_optimization/">optimize parameters through training</a></b>. In supervised ML, the goal is to predict, namely, identifying data signals relevant for the new, unseen.

<a href="https://www.businessnewsdaily.com/10352-machine-learning-vs-automation.html#:~:text=The%20biggest%20difference%20is%20that,is%20frequently%20confused%20with%20AI.&text=But%20the%20difference%20is%20that,and%20then%20thinks%20no%20further.">It is different from automation.</a>

<hr>

## Algorithms / models:

Algorithm / model | Type | Use case | Online demo / example
--- | --- | --- | --
<a href="association_rules">Association Rules</a> | <a href="./glossary">Unsupervised</a>/<a href="./glossary">Supervised</a> | To identify items frequently bought together in transactional data; to perform <a href="https://en.wikipedia.org/wiki/Affinity_analysis">market basket / affinity</a> analysis | <a href="https://danielyang.shinyapps.io/association_rules/">Demo: Generating association rules with transactions data</a> (\*<b>interactive</b>\*)
<a href="neural_network">Neural Network</a> | Unsupervised/Supervised | To understand how similar products are in order to design a campaign | <a href="neural_network">Example: R</a>
<a href="DNN-softmax">Deep Neural Network: Softmax</a> | Unsupervised/Supervised | To capture personalized preferences for a latent factor model for <b><a href="./recommendation_system">recommendations</a></b>;<br/><a href="https://en.wikipedia.org/wiki/Deep_learning#Financial_fraud_detection">To detect fraud transactions</a> | Example: see <a href="collaborative_filtering">collaborative filtering</a>
<a href="collaborative_filtering">Collaborative Filtering</a> | Unsupervised | To <b><a href="./recommendation_system">recommend</a></b> an item to a buyer because (a) similar buyers purchased it and (b) the user purchased similar item(s) | <a href="collaborative_filtering">Examples: Python, R</a>
<a href="content-based_filtering">Content-Based Filtering</a> | Unsupervised | To <b><a href="./recommendation_system">recommend</a></b> an item to a buyer because the item strongly fits the user's preference | <a href="content-based_filtering">Example: Illustration</a>
<a href="clustering">Clustering</a> | Unsupervised | To understand the grouping of consumers with respect to their purchase habits | <a href="clustering">Examples: Python, R</a>
<a href="PCA">PCA</a> | Unsupervised | (a) To summarize data on a 2D map;<br/>(b) To reconstruct data using PCs | <a href="PCA">Examples: Clojure, Python, R</a>
<a href="t-SNE">t-SNE</a> | Unsupervised | To visualize data consisting of legitimate and fraudulent transactions on a 2D map | <a href="t-SNE">Examples: Python, R</a>
<a href="UMAP">UMAP</a> | Unsupervised | To visualize higher dimensional data on a 2D map | <a href="UMAP">Examples: Python, R</a>
<a href="network_analysis">Network Analysis</a> | Unsupervised | To understand the dynamics of how purchasing one item may affect purchasing another | <a href="network_analysis">Example: R</a>
<a href="bayesian_networks">Bayesian/Probabilities Networks</a> | Supervised | To predict the chain of events linking to greater likelihood of consumer purchasing | <a href="bayesian_networks">Example: R</a>
<a href="kNN">k-Nearest Neighbors</a> | Supervised | To predict what product a new customer may like, given the customer's characteristics | <a href="kNN">Examples: Python, R</a>
<a href="SVM">Support Vector Machine (SVM)</a> | Supervised | To predict consumer's dichotomous purchasing decision | <a href="SVM">Examples: Python, R</a>
<a href="naive_bayes">Naive Bayes</a> | Supervised | To predict consumer's dichotomous purchasing decision | <a href="naive_bayes#examples">Examples: Python</a>
<a href="./linear_regression">Linear Regression</a> | Supervised | To explain sales via advertising budget | <a href="./linear_regression/multiple_regression.md">Examples: Python, R</a>
<a href="logistic_regression">Logistic Regression</a> | Supervised | To predict consumer's dichotomous purchasing decision | (1) <a href="logistic_regression">Example: Python</a>; (2) <a href="https://danielyang.shinyapps.io/Logistic_Regression/">Demo: Running logistic regression with retail data</a> (\*<b>interactive</b>\*)
<a href="./decision_tree">Decision Tree</a> | Supervised | To predict consumer's decision of purchasing | <a href="./decision_tree/DT_Purchasing.ipynb">Example: Decision trees of consumer purchasing</a>

<hr>

## Assumptions:

Algorithm / model (selected) | Assumptions
--- | ---
Associate Rules (e.g., apriori) | 1. All subsets of a frequent itemset are frequent.
Decision Trees | 1. The data can be described by features.<br/>2. The class label can be predicted using the logic set of decisions in a decision tree.<br/>3. Effectiveness can be achieved by finding a smaller tree with lower error.
<a href="https://medium.com/analytics-vidhya/assumptions-which-makes-artificial-neural-network-simple-81ba7f46abbc">Neural Networks</a> | As opposed to real neurons:<br/>1. Nodes connect to each other sequentially via distinct layers.<br/>2. Nodes within the same layer do not communicate with each other.<br/>3. Nodes of the same layer have the same activation functions.<br/>4. Input nodes only communicate indirectly with output nodes via the hidden layer.
K-means clustering | 1. The clusters are spherical.<br/>2. The clusters are of similar size.
Naive Bayes | 1. Every pair of feature variables is independent of each other.<br/>2. The contribution each feature makes to the target variable is equal.
Logistic Regression | 1. DV is binary or ordinal.<br/>2. Observations are independent of each other.<br/>3. Little or no multicollinearity among the IV.<br/>4. Linearity of IV (the X) and log odds (the z).<br/>5. A large sample size. It needs at minimum of 10 cases with the least frequent DV for each IV.<br/>6. There is no influential values (extreme values or outliers) in the continuous IV.
Linear Regression | 1. Linearity: The relationship between X and Y is linear.<br/>2. Independence: Residual -- Y is independent of the residuals.<br/>3. Homoscedasticity: Residual -- variance of the residuals is the same for all values of X.<br/>4. Normality: Residual -- residual is normally distributed.<br/>(2-4 are also known as <b>IID</b>: residuals are Independently, Identically Distributed as normal).<br/>5. No or little multicollinearity among X's (for Multiple Linear Regression).

<hr>

## Algorithm Selection:

It dependes on several factors, including (a) the nature of the data, (b) the goal of the analysis, (c) the <a href="./model_evaluation/">relative performance of the algorithm</a>, and (d) the possibility to integrate with business & operations.

Factors | Details
--- | ---
Nature of the data | Categorical, continuous, etc.
Goal of analysis | * To describe, estimate, predict, cluster, classify, associate, explain, etc.<br/>* For example, decision trees are more readily interpretable than neural networks
<a href="./model_evaluation">Algorithm performance/<br/>Model evaluation</a> | * For classification, predictive power can be assessed via <a href="./model_evaluation/">the area under ROC</a><br/>* For regression, there are a variety of choices, including <a href="./model_evaluation/">R<sup>2</sup>, AIC, RMSE</a>
<a href="http://ucanalytics.com/blogs/model-selection-retail-case-study-example-part-7/">Business integration</a> | * Data availability<br/>* Model tuning vs. new model<br/>* Thinking through IT integration at the beginning of the project<br/>* Business end users' actual uses

