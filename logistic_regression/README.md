# <a href="https://en.wikipedia.org/wiki/Logistic_regression">Logistic Regression</a>
To predict yes or no in the target variable

<hr>

## <a href="https://en.wikipedia.org/wiki/Logistic_regression#Latent_variable_interpretation">Fundamental interpretation: the odds as an exponential function of the predictor</a>

* DV = Odds of P(Outcome<sub>Positive</sub>) = P(Outcome<sub>Positive</sub>) / P(Outcome<sub>Negative</sub>)
* IV = Predictor in terms of β<sub>0</sub> + β<sub>1</sub>\*x
* Fundamental Relationship: <a href="https://danielyang.shinyapps.io/Logistic_Regression/">DV = <i>e</i><sup>IV</sup></a>

Related considerations:
* Interesting property: the derivative of DV is also <i>e</i><sup>IV</sup></a>, that is, whether DV is already very high, slight increase in IV will exponentially make DV much higher. For example, if you've already studied for 5 hours, studying for another 30 minutes will <a href="https://danielyang.shinyapps.io/Logistic_Regression/">make the Odds of P(passing) much, much higher</a>.
* Important value to consider: what IV will lead to DV≥1? Ans: When x≥(-β<sub>0</sub>/β<sub>1</sub>)

<hr>

No. | Assumptions
--- | ---
1 | Dependent Variable is binary or ordinal
2 | Observations are independent of each other
3 | Little or no multicollinearity among the independent variables
4 | **Linearity of independent variables (the x) and log odds (the z)**
5 | A large sample size. It needs at minimum of 10 cases with the least frequent outcome for each independent variable in your model.


No. | It does not require the following
--- | ---
1 | It does not need a linear relationship between the dependent and independent variables.
2 | The error terms (residuals) do not need to be normally distributed.
3 | Homoscedasticity is not required.
4 | The dependent variable is not measured on an interval or ratio scale.

<hr>

Example:

### 1. Passing exam

* Raw Data: <a href="./Examples/Passing-exam/Passing-exam.csv">CSV</a>; <a href="./Examples/Passing-exam/Passing-exam.sav">SPSS SAV</a>
* Python: <a href="./Examples/Passing-exam/Passing-exam.py">Python Code</a>; <a href="./Examples/Passing-exam/Passing-exam.py.output">Python Output</a>
* R: <a href="./Examples/Passing-exam/Passing-exam.R">R Code</a>; <a href="./Examples/Passing-exam/Passing-exam.R.output">R Output</a>

<img src="./Examples/Passing-exam/Passing-exam.py.png" width="500" />
<img src="./Examples/Passing-exam/Passing-exam.R.png" width="500" />

The `sigmoid function` or `logistic function`: y = 1/(1+e<sup>`-z`</sup>), where z=β<sub>0</sub>+β<sub>1</sub>\*x; Notice that there is a **negative** sign before z, which is very important.
For example, let's say β<sub>0</sub> = -4.077, β<sub>1</sub> = 1.5046

Studying hours<br>(x) | z=β<sub>0</sub>+β<sub>1</sub>\*x | Prob<sub>pass</sub><br>=y=p=1/(1+e<sup>`-z`</sup>) | Prob<sub>fail</sub><br>=q=(1-y)=(1-p) | odds(y)<br>=(prob<sub>pass</sub>)/(prob<sub>fail</sub>)<br>=y/(1-y)=p/q=e<sup>z</sup>
--- | --- | --- | --- | ---
1 | **-2.5724** | 0.07 | 1-0.07 = 0.93 | 0.07/0.93 = 0.08 
2 | **-1.0678** | 0.26 | 1-0.26 = 0.74 | 0.26/0.74 = 0.34
3 | **0.4368** | 0.61 | 1-0.61 = 0.39 | 0.61/0.39 = 1.55
4 | **1.9414** | 0.87 | 1-0.87 = 0.13 | 0.87/0.13 = 6.97
5 | **3.446** | 0.97 | 1-0.97 = 0.03 | 0.97/0.03 = 31.37

odds(y)<br>=(prob<sub>pass</sub>)/(prob<sub>fail</sub>)<br>=y/(1-y)=p/q=e<sup>z</sup> | log odds = logit func<br>ln(odds(y))=ln(p/q)=z | odds ratio (OR)<br> = e<sup>β<sub>1</sub></sup>
--- | --- | ---
0.08 | **-2.5724** | 0.08/0.02=4.50, where 0.02=e<sup>β<sub>0</sub></sup>
0.34 | **-1.0678** | 0.34/0.08=4.50
1.55 | **0.4368** | 1.55/0.34=4.50
6.97 | **1.9414** | 6.97/1.55=4.50
31.37 | **3.446** | 31.37/6.97=4.50

odds(y)=p/q=e<sup>z</sup>=e<sup>β<sub>0</sub>+β<sub>1</sub>\*x</sup>

## Explaining odds ratio

**odds ratio** = odds(y with x+1) / odds(y with x) = e<sup>β<sub>1</sub></sup>, that is, for every 1-unit increase in x, the odds of y will be multiplied by e<sup>β<sub>1</sub></sup>.

For example, given β<sub>1</sub> = 1.5046, e<sup>1.5046</sup>=4.50, it means that every 1 more hr of studying will increase the odds of passing versus failing by a **constant** multiplication factor of 4.50.

Note that <a href="https://www.quora.com/Why-is-the-exponential-function-always-positive">the exponential function is always positive</a>, but if β<sub>1</sub><0, 0<e<sup>β<sub>1</sub></sup><1, it means every 1-unit increase in x, the odds of y will be multipled by a factor<1

For example, if β<sub>1</sub>=-0.05, it means odds ratio = e<sup>-0.05</sup>=0.95, so for every 1-unit increase in x, the odds of P(Outcome<sub>positive</sub>) versus P(Outcome<sub>negative</sub>) will decrease by 5%. Another example, if β<sub>1</sub>=-0.17, it means odds ratio = e<sup>-0.17</sup>=0.84.

<img src="./images/exponential_func.png" width="70%" />
<img src="./images/exponential_func_interesting_property.png" width="50%" />

Note. If X (the predictor) is binary, the odds ratio = ad/bc in a contingency table. For example:

Actual \ Study | Having studied (1) | Didn't study (0)
--- | --- | ---
Actual Pass (1) | a | b
Actual Fail (0) | c | d

β<sub>1</sub> = ln(ad/bc), odds ratio (OR) = e<sup>β<sub>1</sub></sup> = ad/bc.

## Explaining y (prob<sub>pass</sub>) as a function of z (log odds), the logistic/sigmoid function:

<img src="./images/logistic_func.png" width="70%" />

## Explaining z (log odds) as a function of y (prob<sub>pass</sub>), the logit function:

<img src="./images/logit_func.png" width="70%" />

<hr>

No. | <a href="https://www.quora.com/What-are-alternatives-to-logistic-regression">Alternatives to logistic regression</a>
--- | ---
1 | Support Vector Machine
2 | Decision Trees and Ensembled-based classifier
3 | Naive Bayesian

<hr>

Reference: https://www.quora.com/What-are-the-pros-and-cons-of-using-logistic-regression-with-one-binary-outcome-and-several-binary-predictors

Pros | Logistic regression
--- | ---
1 | Directly interpretable (conditional probabilities)
2 | Easy to implement
3 | Very efficient/fast to train
4 | Can serve as a benchmark to compare to more complex algorithms

Cons | Logistic regression
--- | ---
1 | As a generalized linear model, it cannot solve non-linear hypothesis-space/decision-boundary problems (a decision tree would be better here)
2 | Cannot solve non-categorical dependent variables

<hr>

## Regularized logistic regression

- https://machinelearningmedium.com/2017/09/15/regularized-logistic-regression/
 
