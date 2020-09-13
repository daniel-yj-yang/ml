

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

Derived Index | Definition | Visualization
--- | --- | ---
Recall, True Positive Rate, y-axis of the ROC curve | prob(y<sub>pred</sub>=1 \| y<sub>actual</sub>=1) | <img src="./images/recall.png" width="150px">
False Positive Rate, x-axis of the ROC curve | prob(y<sub>pred</sub>=1 \| y<sub>actual</sub>=0) | <img src="./images/FPR.png" width="150px">

<hr>

Better classification models:<br/>
<p align="center"><img src="./images/roccomp.jpg" width="300px"><br/>(<a href="http://gim.unmc.edu/dxtests/roc3.htm">image source</a>)<br/><br/><img src="./images/ROC_curve_better_models.png" width="500px"></p>

<hr>

Higher decision threshold:<br/>
<p align="center"><img src="./images/decision_threshold.png" width="500px"><br/>(modified from <a href="https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65">image source</a>)<br/><br/><img src="./images/higher_decision_threshold.png" width="450px"></p>

<hr>

Derived Index | Definition | Visualization
--- | --- | ---
Precision | prob(y<sub>pred</sub>=1 \| y<sub>actual</sub>=1) | <img src="./images/precision.png" width="150px">

<hr>

For other illustrations, please see the <a href="./images/visualization_of_confusion_matrix.pptx">PowerPoint deck</a> I made.
