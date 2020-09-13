

<table>
  <tr>
    <td><p align="right">\ Predict</p>Actual</td>
    <td><img src="./images/y_pred=0.png" width="150px"></td>
    <td><img src="./images/y_pred=1.png" width="150px"></td>
    <td>Sum</td>
  </tr>
  <tr>
    <td>
      <img src="./images/y_actual=0.png" width="150px">
    </td>
    <td>
      <img src="./images/TN.png" width="150px">
    </td>
    <td>
      <img src="./images/FP.png" width="150px">
    </td>
    <td>
      N = TN + FP
    </td>
  </tr>
  <tr>
    <td>
      <img src="./images/y_actual=1.png" width="150px">
    </td>
    <td>
      <img src="./images/FN.png" width="150px">
    </td>
    <td>
      <img src="./images/TP.png" width="150px">
    </td>
    <td>
      P = FN + TP
    </td>
  </tr>
  <tr>
    <td>Sum</td>
    <td>TN + FN</td>
    <td>TP + FP</td>
    <td>All samples</td>
  </tr>
  </table>

<hr>

Derived Index | Definition | Visualization 1 | Visualization 2
--- | --- | --- | ---
Recall, True Positive Rate, y-axis of the ROC curve | prob(y<sub>pred</sub>=1 \| y<sub>actual</sub>=1) | <img src="./images/recall_1.png" width="150px"> | <img src="./images/recall_2.png" width="150px">
False Positive Rate, x-axis of the ROC curve | prob(y<sub>pred</sub>=1 \| y<sub>actual</sub>=0) | <img src="./images/FPR_1.png" width="150px"> | <img src="./images/FPR_2.png" width="150px">

Note. Raising classification threshold does not decrease recall in Visualization 1, but does not impact recall n Visualization 2.
This illustrates why recall will **always decrease or stay the same** when raising classification threshold.

<hr>

Classification models with varying predictive power:<br/>
<p align="center"><img src="./images/roccomp.jpg" width="300px"><br/>(<a href="http://gim.unmc.edu/dxtests/roc3.htm">image source</a>)<br/><br/><img src="./images/ROC_curve_better_models.png" width="500px"></p>

<hr>

Raising classification decision threshold:<br/>
<p align="center"><img src="./images/decision_threshold.png" width="500px"><br/>(modified from <a href="https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65">image source</a>)<br/><br/><img src="./images/higher_decision_threshold_1.png" width="450px"><br/><br/><img src="./images/higher_decision_threshold_2.png" width="450px"></p>

<hr>

Derived Index | Definition | Visualization 1 | Visualization 2
--- | --- | --- | ---
Precision | prob(y<sub>actual</sub>=1 \| y<sub>pred</sub>=1) | <img src="./images/precision_1.png" width="150px"> | <img src="./images/precision_2.png" width="150px">

Note. Raising classification threshold does not impact precision in Visualization 1, but increases precision in Visualization 2.
This illustrates why raising classification threshold will usually (**but not always**) increase precision.

<hr>

For other illustrations, please see the <a href="./images/visualization_of_confusion_matrix.pptx">PowerPoint deck</a> I made.
