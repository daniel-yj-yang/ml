# Principal Component Analysis
To project the original data on a reduced dimensional (e.g., 2D) space

<hr>

### Background Concept

Covariance matrix ```Q``` reflects how much the features in ```X``` are linearly linked with each other.

- ```Q = X'X / (n-1)```

Matrix | Meaning
--- | ---
<b>X</b> | the empirical matrix for the original variables, column centered
<b>Q</b> | the empirical covariance matrix for the original variables

<hr>

### Key Concept of PCA


Goal: To find the eigenvalues and eigenvectors of the covariance matrix Q


- Goal: to find eigenvalue ```λ``` and the associated eigenvector ```W``` that satisfy ```QW = λW```
- We are interested in a larger eigenvalue ```λ```, because its associated eigenvector ```W``` will be a dimension that can account for more variance in Q.
- Q = WΛW'
- Λ = W'QW

<b>W</b> | the p-by-p matrix of weights whose columns are the <a href="https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors">eigenvectors</a> of <b>Q</b>
<b>Λ</b> | the <a href="https://en.wikipedia.org/wiki/Diagonal_matrix">diagonal matrix</a> of <a href="https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors">eigenvalues</a> λ<sub>(k)</sub> of <b>Q</b>

Note.
- Detailed explanation of <a href="./eigenvector_and_eigenvalue.md">eigenvector and eigenvalue</a>

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
