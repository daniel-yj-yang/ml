# Principal Component Analysis
To diagonalize the covariance matrix

### Concepts

<img src="./images/PCA_concepts_covariance.png" width="150px">

Matrix | Meaning
--- | ---
<b>X</b> | the empirical covariance matrix for the original variables
<b>Q</b> | the empirical matrix for the original variables
<b>W</b> | the p-by-p matrix of weights whose columns are the eigenvectors of <b>X<sup>T</sup>X</b>
<b>Λ</b> | the diagonal matrix of eigenvalues λ<sub>(k)</sub> of <b>X<sup>T</sup>X</b>

<hr>

### Example - iris

<hr>

#### Scree plot (1): To see the eigenvalue of each principal component
<img src="./images/PCA_iris_scree_plot_eigenvalue.png" width="600px">

<hr>

#### Scree plot (2): Or, in order words, to see how much variation each principal component captures in the data
<img src="./images/PCA_iris_scree_plot_percentage.png" width="600px">

<hr>

#### Loading plot: To see how much each feature influences a principal component
<img src="./images/PCA_iris_loading_plot.png" width="600px">

<hr>

#### Biplot: PCA score plot + loading plot
<img src="./images/PCA_iris_biplot.png" width="700px">

<hr>

<a href="./PCA.R">R code</a>
