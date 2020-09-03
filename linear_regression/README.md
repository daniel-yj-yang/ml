# Linear Regression
To explain quarterly sales via other explanatory variables

<hr>

## Examples

Example | Details
--- | ---
Simple Linear Regression | To explain sales via the budget of youtube and facebook advertising, respectively:<br/><table><tr><td><img src="./images/explain_sales_via_youtube_ads.png" width="400px"></td><td><img src="./images/explain_sales_via_facebook_ads.png" width="400px"></td></tr></table>
Multiple Linear Regression | Here are the comprehensive <a href="./multiple_regression.md">results</a> of my <a href="./linear_regression.R">R code</a> that run a multiple linear regression of sales on the budget of three advertising medias (youtube, facebook and newspaper)

<hr>

## Models

Model | Hypothesis, h<sub>θ</sub>(x) | Notes
--- | --- | ---
Simple Linear Regression | θ<sup>T</sup>x = θ<sub>0</sub>x<sub>0</sub> + θ<sub>1</sub>x<sub>1</sub><br/>Conventionally, x<sub>0</sub> = 1 | assumes linearity in the relationship between x and y, and <a href="https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables">I.I.D.</a> in residuals
Multiple Linear Regression | θ<sub>T</sub>x = θ<sub>0</sub>x<sub>0</sub> + θ<sub>1</sub>x<sub>1</sub> + θ<sub>2</sub>x<sub>2</sub> + ⋯ + θ<sub>n</sub>x<sub>n</sub><br/>Conventionally, x<sub>0</sub> = 1 | all assumptions in simple linear regression, while also assuming little or no multicollinearity
Polynomial Regression | <table><tr><th>Function</th><th>Math Expression, y ~ f(x,θ)</th></tr><tr><td>Quadratic</td><td>θ<sub>0</sub>x<sub>0</sub> + θ<sub>1</sub>x<sub>1</sub> + θ<sub>2</sub>〖x<sub>1</sub>〗<sup>2</sup></td></tr><tr><td>A "circle"</td><td>θ<sub>0</sub>x<sub>0</sub> + θ<sub>1</sub>x<sub>1</sub> + θ<sub>2</sub>x<sub>2</sub> + θ<sub>3</sub>〖x<sub>1</sub>〗<sup>2</sup> + θ<sub>4</sub>〖x<sub>2</sub>〗<sup>2</sup></td></tr><tr><td>Cubic</td><td>θ<sub>0</sub>x<sub>0</sub> + θ<sub>1</sub>x<sub>1</sub> + θ<sub>2</sub>〖x<sub>1</sub>〗<sup>2</sup> + θ<sub>3</sub>〖x<sub>1</sub>〗<sup>3</sup></td><tr><td>Square root</td><td>θ<sub>0</sub>x<sub>0</sub> + θ<sub>1</sub>x<sub>1</sub> + θ<sub>2</sub>〖x<sub>1</sub>〗<sup>0.5</sup></td></tr><tr><td>Other higher-order polynomial func</td><td>θ<sub>0</sub>x<sub>0</sub> + θ<sub>1</sub>x<sub>1</sub> + θ<sub>2</sub>x<sub>2</sub> + θ<sub>3</sub>x<sub>1</sub>x<sub>2</sub> <i>etc.</i></td></tr></table>Conventionally, x<sub>0</sub> = 1 | considered linear regression since it is linear in the regression coefficients, although the decision boundary is non-linear

<hr>

## Estimation of coefficients/parameters

Method | Details
--- | ---
To minimize cost function | <img src="./images/minimize_cost_function.png" width="600px">
To compute analytically with <a href="http://mlwiki.org/index.php/Normal_Equation">normal equation</a> | <img src="./images/normal_equation.png" width="200px"><br/>X transpose times X inverse times X transpose times Y

<hr>

## Interpretation of coefficients

Coefficient | Interpretation
--- | ---
Unstandardized | It represents the amount by which dependent variable changes if we change independent variable by one unit keeping other independent variables constant.
Standardized | The standardized coefficient is measured in units of standard deviation (both X and Y). A beta value of 1.25 indicates that a change of one standard deviation in the independent variable results in a 1.25 standard deviations increase in the dependent variable.

<hr>

## Regularized linear regression

Model | Penalty | Description
--- | --- | ---
<a href="https://en.wikipedia.org/wiki/Lasso_(statistics)">Lasso regression</a><br/>(To simplify model) | L1 norm:<br/><img src="./images/L1_constraint.png" width="300px"> | * To shrink some of the coefficients to exactly 0;<br/>* This leads to a sparse model (having many β's = 0), helping with interpretability
<a href="https://en.wikipedia.org/wiki/Tikhonov_regularization">Ridge regression</a><br/>(To handle multicollinearity)<br/><br/> it is named "ridge" because the matrix form of <i>I<sub>p</sub></i> looks like a ridge | Using **L2 Regularization Technique:**<br/><br/>Adding “squared magnitude” of coefficient (squared L2 norm) as a penalty term to the cost function when there is multicollinearity among X's (which leads to <a href="https://en.wikipedia.org/wiki/Multicollinearity">overfitting of the data</a>):<br/><p align="center"><img src="./images/ridge_regression_cost_function.png" width="250px"></p><br/>Reference: L2 norm:<br/><p align="center"><img src="./images/ridge_regression_L2_norm.png" width="75px"></p><br/><a href="https://stats.stackexchange.com/questions/160096/what-are-the-differences-between-ridge-regression-using-rs-glmnet-and-pythons">R's glmnet(alpha=0)</a> and minimize this objective function:<br/><img src="./images/glmnet_objective_function.png" width="300px"><br/><br/><a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html">sklearn's Ridge()</a> and minimize this objective function:<br/>\|\|y - Xw\|\|^2_2 + alpha * \|\|w\|\|^2_2<br/><br/><a href="https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.fit_regularized.html">statsmodels' OLS.fit_regularized(L1_wt=0.0)</a> and minimize this insteead:<br/>0.5\*RSS/n + 0.5\*alpha\*\|params\|^2_2<br/> | * **Multicollinearity** may give rise to large variances of the coefficient estimates, which can be reduced by ridge regression;<br/><br/>* When (X<sup>T</sup>X)<sup>-1</sup> does not exist, (X<sup>T</sup>X) being singular;<br/>* Recall, the LS estimator:<br/><p align="center"><img src="./images/LS_estimator.png" width="150px"></p>* Such <b>non-invertibility</b> is due to (a) multicollinearity, or (b) # predictors > # observations;<br/>* To fix, use:<br/><p align="center"><img src="./images/ridge_regression_estimator.png" width="200px"></p>
<a href="https://en.wikipedia.org/wiki/Elastic_net_regularization">Elastic net</a> | Lasso L1 penalty + Ridge L2 penalty | ---

<hr>

### Example: Ridge Regression

λ is the only parameter to adjust. When λ increases, coefficients tend to shrink but MSE tends to increase (see graphs below). Thus, there exists a sweet spot where coefficients shrink and MSE is also the lowest. Because the implementations in R and Python are different, the coefficients across different implementations may not be directly comparable; however, the MSE/RMSE/R<sup>2</sup> are comparable.

<img src="./images/ridge_Hitter_coef_vs_log_lambda.png" width="500px"> <img src="./images/ridge_Hitter_MSE_vs_log_lambda.png" width="500px">

Model performance with <a href="./regularized_linear_regression.R">the testing set</a> | Linear regression | Ridge regression | Lasso
--- | --- | --- | ---
RMSE | 418.3987| 374.5406 | 380.2771
R<sup>2</sup> | 0.2209 | 0.3757 | 0.3564
Coefficient | <img src="./images/linear_Hitter_coef.png" height="400px"> | <img src="./images/ridge_Hitter_coef.png" height="400px"> | <img src="./images/lasso_Hitter_coef.png" height="400px">

Compared to linear regression, both ridge and lasso regression appear to have improved the model performance.

My own codes: <a href="./regularized_linear_regression.R">R</a> and <a href="./regularized_linear_regression.py">Python</a>
