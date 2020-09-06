# Principal Component Analysis
To project complicated original data onto a lower dimensional (e.g., 2D) space

<hr>

### Background Concept

A covariance matrix ```Q``` reflects the variation/variability of ```p``` features (```X```) in terms of diagnoal (variance) and off-diagnoal (covariance, joint variability).

- ```Q = X'X / (n-1)```

Matrix | Meaning
--- | ---
<b>X</b> | the empirical matrix for the original variables, column centered
<b>Q</b> | the empirical covariance matrix for the original variables

<hr>

### Key Concept of PCA

- ***Common-Language Goal***: To find ```m``` principal components to account for most of the variability in ```X```, where ```m``` << ```p```
- ***Technical Goal***: To find the eigenvalues and eigenvectors of the covariance matrix ```Q``` to <a href="http://www.stat.columbia.edu/~fwood/Teaching/w4315/Fall2009/pca.pdf">decompose</a> and reproduce ```Q``` (namely, singular value decomposition): ```Q = WΛW'```

```
Q: Why using the covariance matrix (or correlation matrix if standardized), as opposed to other matrices?
A: To allow retaining of the variation present in the data set, in terms of variances (diagnoal) and covariances (off-diagnoal) among features, by eigenvalues (the sum equals to the sum of the variances) and eigenvectors.
```

- Goal: To find each eigenvalue ```λ``` and the associated eigenvector ```v``` that satisfy ```Qv = λv```
- Importantly, we are interested in the largest few eigenvalues (e.g., ```λ1```), because their associated eigenvectors (e.g., ```v1```) will be dimensions that can retain the most variation present in ```X``` (e.g., via ```Qv1 = λ1v1```)

Matrix | Meaning
--- | ---
<b>W</b> | the p-by-p matrix of weights whose columns are the <a href="https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors">eigenvectors</a> ```v``` of ```Q```
<b>Λ</b> | the <a href="https://en.wikipedia.org/wiki/Diagonal_matrix">diagonal matrix</a> whose diagnoal elements are the <a href="https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors">eigenvalues</a> ```λ``` of ```Q```

Note.
- Detailed explanation of <a href="./eigenvalue_and_eigenvector.md">eigenvalue and eigenvector</a>
- Importantly, ```WW' = W'W = I``` as ```W``` is orthonormal
- Property: ```Λ = W'QW```
- Why the y-axis in the scree plot is labled as percentage of explained **variance**? Please see an interpretation <a href="https://stats.stackexchange.com/questions/22569/pca-and-proportion-of-variance-explained">here</a>
- The first principal component is the direction that maximizes the variation along that direction in the projected data.

See also.
- <a href="https://en.wikipedia.org/wiki/Principal_component_analysis">Dimensionality reduction and principal component regression</a>

<hr>

### Example - the iris dataset

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

#### Code
- <a href="./PCA.R">R code</a>

<hr>

### Reference

- <a href="https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues/140579">Making sense of PCA</a>
