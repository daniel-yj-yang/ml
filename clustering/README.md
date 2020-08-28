# Clustering
To see how customers' behaviors tend to group together

<hr>

## Example of transactions data
The following bubble plot was generated using the <a href="https://archive.ics.uci.edu/ml/datasets/online+retail">UCI online retail dataset</a> and my <a href="./clustering.R">variation</a> of a <a href="https://rpubs.com/tahamokfi/Part1_AnalyzeTransactionData">clustering code</a> in R. The size of the bubble reflects the number of customers, and the cluster reflects the consuming habits of the customers.

<p align="center"><img src="./images/Invoices_cluster_k=3.png" width="800px"></p>

<hr>

## Concepts (k-means clustering)

Step#1 | Step#2 | Step#3 | Step#4
--- | --- | --- | ---
<img src="./images/k-means-step1.png" width="250px"> | <img src="./images/k-means-step2.png" width="250px"> | <img src="./images/k-means-step3.png" width="250px"> | <img src="./images/k-means-step4.png" width="250px">
Goal: to assign to k clusters | Randomly pick k points as centers;<br/>Assign to nearest center  | Revise centers<br/> as the means of points  | Repeat step#2 and #3<br/>until convergence

<hr>

## To determine the number of clusters
Method | Use | Examples
--- | --- | ---
<a href="https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index">Davies–Bouldin index</a> | The ratio of within-cluster distances to between-cluster distances.<br/><br/>With smaller within-cluster distances and bigger between-cluster distances, the DB value is lower, indicating better cluster solution | <p><img src="./images/invoce_data_DB_value.png" width="600px"></p>
<a href="https://en.wikipedia.org/wiki/Elbow_method_(clustering)">Elbow method</a> | Plotting the explained variation as a function of the number of clusters | <p><img src="./images/Elbow_method.png" width="600px"></p>
<a href="https://en.wikipedia.org/wiki/Silhouette_(clustering)">Silhouette method</a><br/>[sil-oo-et] | A measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation).<br/><br/>That is, the similarity between a data point and points in the same cluster, compared to points outside of the cluster. | <p><img src="./images/Silhouette_k=4.png" width="800px"></p>

<hr>

## References:
* k-means clustering: <a href="./k-means-clustering.py">Python code</a> and <a href="./k-means clustering.ipynb">Jupyter Notebook</a>
