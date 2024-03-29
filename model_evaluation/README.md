# Model Evaluation
Covering the basics of how to evaluate model performance

<hr> 

## Regression

Metrics | Measuring | Features or interpretation
--- | --- | ---
RMSE (Root Mean Square prediction Error) | Error | <a href="https://medium.com/human-in-a-machine-world/mae-and-rmse-which-metric-is-better-e60ac3bde13d">Penalizing large errors</a>
MSE (Mean Squared Error) | Error<br/><img src="./images/MSE_formula.png"> | Penalizing large errors, as in RMSE
MAE (Mean Absolute Error) | Error | Easy to interpret
η<sup>2</sup>, Ω<sup>2</sup>, Cohen's *f*<sup>2</sup> | effect size | ---
R<sup>2</sup> | goodness-of-fit | ---
Adj. R<sup>2</sup> | R<sup>2</sup> that has been adjusted for the number of predictors in the model | ---
<a href="https://methodology.psu.edu/AIC-vs-BIC">AIC (Akaike Information Criterion), BIC (Bayesian Information Criterion)</a>  | Trade-off between goodness of fit and model simplicit | The lower, the better
Outliers | Diagnostic | ---
Residual plot | Diagnostic | y = residual, x = predictor, to see non-random pattern, which may indicate a non-linear relationship, heteroscedasticity, or left over from series correlation
QQ plot | Diagnostic | How well the distribution of residuals fit the normal distribution
Multicollinearity test | Diagnostic	| ---

<hr>

## <a href="https://stats.stackexchange.com/questions/34193/how-to-choose-an-error-metric-when-evaluating-a-classifier">Classification</a>

### ROC (Receiver operating characteristic) curve

axis | name | conditional probability | meaning | sensitive to baseline probabilities, prob(y<sub>actual</sub>=1)
--- | --- | --- | --- | ---
y-axis | True Positive Rate = Recall = Sensitivity | prob(y<sub>pred</sub>=1 \| y<sub>actual</sub>=1) | higher values mean lower FNR | No
x-axis | False Positive Rate | prob(y<sub>pred</sub>=1 \| y<sub>actual</sub>=0) | higher values mean higher FPR | No

<p align="center"><img src="./images/roccomp.jpg" width="40%" /><br/>(<a href="http://gim.unmc.edu/dxtests/roc3.htm">image source</a>)</p>

Note:
- The y-axis and x-axis on the ROC curve are probabilities conditioned on the true class label and <a href="https://stats.stackexchange.com/questions/7207/roc-vs-precision-and-recall-curves">will be the same regardless of what P(Y<sub>actual</sub>=1) is</a> and insenitive to imbalanced sample class size.
-  Since ROC curve is insensitive to different baseline probabilities, ROC curve is suitable for answering the following question:

```
How well can this classifier be expected to perform in general, regardless of different baseline probabilities?
```

<hr>

### Interpretation of the ROC curve

<a href="http://www.dataschool.io/roc-curves-and-auc-explained/">AUC</a> (Area Under Curve) of the ROC curve

- ROC curve is a plot of Power (1-β; Recall) as a function of α (that is, 1-specificity)
- AUC can be interpreted as the **probability** that the model **ranks** a random positive example (a random sample of y<sub>actual</sub>=1) **more highly** than a random negative example (a random sample of y<sub>actual</sub>=0), where "ranks more highly" means prob(y<sub>pred</sub>=1 \| y<sub>actual</sub>=1) > prob(y<sub>pred</sub>=1 \| y<sub>actual</sub>=0), that is, y-axis > x-axis on the ROC curve.
- If prob(y<sub>pred</sub>=1 \| y<sub>actual</sub>=1) = prob(y<sub>pred</sub>=1 \| y<sub>actual</sub>=0), then y-axis = x-axis, then AUC = 50% probability, which is the same as random guessing.

- Accuracy is sensitive to **class imbalance** (that is, the ratio of Actual Positive cases [P] to Actual Negative cases [N]), but the ROC curve is **independent** of the P:N ratio and is therefore suitable for comparing classifiers when this ratio may vary.

- AUC is an aggregate measure of binary classifier's performance **across all possible decision thresholds (the threshold to make the decision that y<sub>pred</sub>=1)**
- Increasing decision threshold (cut-off probability score for predicting y=1 vs. y=0) equals to moving a point on the ROC curve to the left, making it harder to classify y=1 or reducing the size of y<sub>pred</sub>=1.

<p align="center"><img src="./images/decision_threshold.png" width="600px"><br/>(modified from <a href="https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65">image source</a>)</p>
<p align="center"><img src="./images/decision_threshold_ROC.gif" width="600px"><br/>(<a href="http://arogozhnikov.github.io/2015/10/05/roc-curve.html">website reference</a>)</p>

- For a random classifier, it will be right/wrong 50% of the time, thus half predictions would be true/false positives, thus the diagonal line
- AUC can be converted to the Gini index, which is 2\*AUC-1
- AUC can also be interpreted as predictive power

<p align="center"><img src="./images/prob_distribution_and_ROC.gif" width="600px"><br/>When y_preb_prob > the decision threshold, y_pred=1;<br/>Ideally, the performance of a ML classification algorithm would improve over time via training, resulting in a more <b>sensitive</b> detection of the actual y, given X's and α;<br/>that is, given a specific α = p(y_pred=1 | y_actual=0) decreases, recall = p(y_pred=1 | y_actual=1) increases;<br/>From the model's perspective, the y_actual=1 probability distribution (given X's) is gradually more separable from y_actual=0 probability distribution (given X's), as the model becomes more sensitive to the difference of the two y classes (given X's).</p>

- When comparing two models and **their ROC curves cross**, it is possible to have higher AUC scores in one model but the other model <a href="https://stackoverflow.com/questions/38387913/reason-of-having-high-auc-and-low-accuracy-in-a-balanced-dataset">performs better</a> for a majority of the thresholds with which one may actually use the classifier.

<hr>

### <a href="https://www.quora.com/What-is-Precision-Recall-PR-curve">Precision-recall curve</a>
  
axis | name | conditional probability | meaning | sensitive to baseline probabilities, prob(y<sub>actual</sub>=1)
--- | --- | --- | --- | ---
y-axis | Precision | prob(y<sub>actual</sub>=1 \| y<sub>pred</sub>=1) | higher values mean lower FDR | Yes
x-axis | Recall | prob(y<sub>pred</sub>=1 \| y<sub>actual</sub>=1) | higher values mean lower FNR | No

<br>
<p align="center"><img src="./images/precision_recall_curve.png" width="450px" /></p>

Note:
- Since precision (but not recall) is sensitive to different baseline probabilities, PR curve is suitable for answering the following question:

```
How meaningful is a positive result from my classifier, given the specific baseline probabilities of my population?
```
 
<hr>

### <a href="http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/">Confusion matrix</a>

For visualization using conditional probabilities, please see the <a href="./confusion_matrix/">illustrations</a> that I made.

* Say "YES" (Positive) = Identification
* Say "NO" (Negative) = Rejection

<table>
    <tr>
        <td><p align="right">\ Predict</p>Actual</td>
        <td>y_pred=0</td>
        <td>y_pred=1</td>
        <td>Sum</td>
    </tr>
    <tr>
        <td>y_actual=0</td>
        <td>True Negative (TN)<br>Correct Rejection<br><a
                href="https://en.wikipedia.org/wiki/Sensitivity_and_specificity">Specificity</a> = True Negative Rate
            (1-α) = TN/N<br>Note: Negative Predictive Value = TN/(TN+FN)</td>
        <td><b>False Positive</b> (FP)<br>False alarm, Type I error<br><a
                href="https://en.wikipedia.org/wiki/Type_I_and_type_II_errors#Type_I_error">False Positive Rate (FPR;
                α)</a> = FP/N<br>Note: False Discovery rate = FP/(TP+FP)</td>
        <td>N=TN+FP</td>
    </tr>
    <tr>
        <td>y_actual=1</td>
        <td><b>False Negative</b> (FN)<br>Miss, Type II Error<br><a
                href="https://en.wikipedia.org/wiki/Type_I_and_type_II_errors#Type_II_error">False Negative Rate (FNR;
                β)</a> = FN/P<br>Note: False Omission Rate = FN/(TN+FN)</td>
        <td>True Positive (TP)<br>Hit, Correction Identification<br><a
                href="https://en.wikipedia.org/wiki/Statistical_power">Power (1-β)</a> = Recall = Sensitivity =
            TP/P<br>Note: Precision = Positive Predictive Value = TP/(TP+FP)</td>
        <td>P=FN+TP</td>
    </tr>
    <tr>
        <td>Sum</td>
        <td>TN+FN</td>
        <td>FP+TP</td>
        <td>Total</td>
    </tr>
</table>

Derived Index | Direction in the table| Definition | To minimize | Example | Also known as
--- | --- | --- | --- | --- | ---
<b>Accuracy</b> | both | (TP+TN)/Total | --- | --- | ---
**<a href="https://en.wikipedia.org/wiki/Precision_and_recall">Precision</a>** | vertical | <b>p(y_actual=1 \| y_pred=1)</b> = TP/(TP+FP) | FDR;<br>Precision = 1-FDR | --- | <a href="https://en.wikipedia.org/wiki/Confusion_matrix">Positive Predictive Value</a>
**<a href="https://en.wikipedia.org/wiki/Precision_and_recall">Recall</a>**<br>=True Positive Rate (TPR) | horizontal | <b>p(y_pred=1 \| y_actual=1)</b> = TP/(TP+FN) | Type II error, Miss;<br/>Recall = 1-β | High cost associated with missing gold when digging for gold | The y-axis in the ROC curve, **sensitivity**<br/><br/>In hypothesis testing:<br/><a href="https://en.wikipedia.org/wiki/False_positives_and_false_negatives">(1-β)</a>, correctly rejecting H<sub>0</sub>, <a href="https://en.wikipedia.org/wiki/Statistical_power">Power</a><br/><br/>Hit Rate
F<sub>1</sub> score | both | TP/(TP+0.5*(FP+FN)) | FP and FN | --- | Another measure of accuracy<br>the harmonic mean of precision and recall
False Negative Rate (FNR) | horizontal | <b>p(y_pred=0 \| y_actual=1)</b> = FN/P | --- | --- | Miss Rate<br/><br/>In hypothesis testing:<br/>β, Type II error rate 
Specificity=TNR  | horizontal | <b>p(y_pred=0 \| y_actual=0)</b> = TN/N | α;<br/>Specificity = 1-α | --- | Correct rejection rate<br/><br/>In hypothesis testing:<br/>(1-α);<br/>Rejection means saying No;<br/>it actually means correctly accepting H<sub>0</sub>
False Positive Rate (FPR)<br>=(1-Specificity) | horizontal | <b>p(y_pred=1 \| y_actual=0)</b> = FP/N | --- | --- | The x-axis in the ROC curve, False Alarm Rate, Fall-out rate;<br/><br/>In hypothesis testing:<br/><a href="https://en.wikipedia.org/wiki/False_positives_and_false_negatives#:~:text=The%20false%20positive%20rate%20is%20equal%20to%20the%20significance%20level,the%20specificity%20of%20the%20test.">**signifiance level**, α</a>, <a href="https://en.wikipedia.org/wiki/Type_I_and_type_II_errors#Type_I_error">Type I error rate</a>
False Discovery Rate (FDR) | vertical | <b>p(y_actual=0 \| y_pred=1)</b> = FP/(TP+FP) | --- | --- | ---
False Omission Rate (FOR) | vertical | <b>p(y_actual=1 \| y_pred=0)</b> = FN/(TN+FN) | --- | --- | ---
Misclassification Rate | both | (FP+FN)/Total | --- | --- | Error rate
Prevalence | horizontal | P/Total | --- | --- | ---
Negative Predictive Value (NPV) | vertical | <b>p(y_actual=0 \| y_pred=0)</b> = TN/(TN+FN) | FOR;<br/>NPV = 1-FOR | --- | ---
Positive Likelihood Ratio (LR+) | --- | (1-β)/α<br>cf. the ROC curve | --- | (1-β)=.80,α=.05,(1-β)/α=16 | ---
Negative Likelihood Ratio (LR-) | --- | β/(1-α) | --- | β=.20,(1-α)=.95,β/(1-α)=.21 | ---
<a href="https://en.wikipedia.org/wiki/Sensitivity_and_specificity">Diagnostic Odds Ratio (DOR)</a> | --- | (LR+)/(LR-)<br>=(1-β)(1-α)/(αβ) | --- | β=.20,α=.05,DOR=76 | ---
Balanced Error Rate (BER) | --- | 0.5*(αβ) | --- | A balanced measure of accuracy | ---
Geometric Mean (G-Mean) | --- | sqrt(sensitivity * specificity) | --- | measures the balance between classification performances on both the majority and minority classes. Insensitive to imbalance classes | ---
Matthew's Correlation Coefficient | --- | --- | --- | A correlation coefficient between the observed and predicted classifications. Least influenced by imbalanced data | ---

<hr>

### Imbalanced sample classes, that is, prob(y<sub>actual</sub>=1) >> 50% (or << 50%)

Accuracy is <a href="https://datascience.stackexchange.com/questions/806/advantages-of-auc-vs-standard-accuracy">sensitive</a> to class imbalance, but AUC is <a href="http://fastml.com/what-you-wanted-to-know-about-auc/">insensitive</a> to that.

For example, 99% of the cases are in the same class (e.g., non-ASD), and it's easy to achieve 99% accuracy by predicting the majority/average all the time but AUC will be very low.

<hr>

<a href="https://www.researchgate.net/post/In_classification_how_do_i_handle_an_unbalanced_training_set">Ways to deal with imbalanced data</a> | Details | For ...
--- | --- | ---
Use AUC rather than accuracy | AUC is insensitive to imbalanced sample classes | ---
Sampling methods | e.g., post-hoc up-sampling or down-sampling | Confusion matrix
Alternative cutoff | --- | Confusion matrix
Unequal case weights | different weights on individual data points | Logistic regression
Adjusting prior probabilities | --- | Naive Bayes 

<hr>

## Generalizability evaluation

Method | Purpose | Implementation
--- | --- | ---
k-fold cross-validation | To assessing how the results of a statistical analysis will generalize to an independent data set, that is, how accurately a predictive model will perform in practice | In k-fold cross-validation, the original sample is randomly partitioned into k equal sized subsamples. Of the k subsamples, a single subsample is retained as the validation data for testing the model, and the remaining k − 1 subsamples are used as training data. The cross-validation process is then repeated k times, with each of the k subsamples used exactly once as the validation data. The k results can then be averaged to produce a single estimation.
<a href="https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff">Bias-variance tradeoff</a> | <p><img src="./images/bias-variance-tradeoff.png" width="600px;"></p><p><img src="./images/bias_and_variance.png" width="600px;"></p>(<a href="http://scott.fortmann-roe.com/docs/BiasVariance.html">image source</a>) |

<hr>

## Hyperparameter tuning

- To perform <a href="https://scikit-learn.org/stable/modules/grid_search.html">grid search</a>

<hr>

## Reference

- A <a href="https://scikit-learn.org/stable/modules/model_evaluation.html">comprehensive collection</a> of model evaluation functions in Scikit-learn
