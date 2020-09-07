rm(list=ls())
cat("\014")

# Daniel Yang, Ph.D. (daniel.yj.yang@gmail.com)

# https://www.datacamp.com/community/tutorials/pca-analysis-r

require(datasets)

#gender <- read.csv('/Users/daniel/Data-Science/Data/Gender/01_heights_weights_genders.csv')
#data <- gender[,c(2:3)]
#groups <- gender$Gender

#data(swiss)
#data <- swiss[,c(2:6)]
#groups <- rep(0, nrow(swiss))

data(iris)
data <- iris[,c(1:4)]
groups <- iris$Species

########################################################################################
## Use SVD to perform PCA without calculating Q, the covariance matrix of X
X <- scale(data, center = TRUE, scale = FALSE) # X = UDV'
n <- nrow(X)
svd_results <- svd(X)
U <- svd_results$u
t(U) %*% U # U'U = I
D <- diag(svd_results$d) 
eigenvalues <- (svd_results$d/sqrt(n-1))**2 # equivalent to eigenvalues of Q, the covariance matrix of X
V <- svd_results$v # equivalent to W, the eigenvectors of Q, the covariance matrix of X
t(V) %*% V # V'V = I
PCs <- U %*% D # projected score matrix T, equivalent to X %*% V
########################################################################################

pca_results <- prcomp(data, center = TRUE, scale. = TRUE)

summary(pca_results)

eigenvalues <- pca_results$sdev^2
eigenvectors <- pca_results$rotation

sum(eigenvalues)
require(magrittr)
eigenvectors ^2 %>% colSums() # the eigenvector is normalized to have magnitude/length of 1

PCs <- scale(data) %*% eigenvectors # projected score matrix T
apply(PCs, 2, var)
apply(PCs, 2, var) - eigenvalues  ## Var(PCs) = eigenvalues


###########################################################################
################# Reconstruction using PCA
# https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com
X = iris[,1:4]
mu = colMeans(X)

Xpca = prcomp(X)

nComp = 2
Xhat = Xpca$x[,1:nComp] %*% t(Xpca$rotation[,1:nComp])
Xhat = scale(Xhat, center = -mu, scale = FALSE)

# compare the first iris
Xhat[1,]
iris[1,]
###########################################################################

# scree plot / cumulative plot
# https://rpubs.com/njvijay/27823
pcaCharts <- function(x) {
  x.var <- x$sdev ^ 2
  x.pvar <- x.var/sum(x.var)
  #print("proportions of variance:")
  #print(x.pvar)
  
  par(mfrow=c(2,2))
  plot(x.pvar,xlab="Principal component", ylab="Proportion of variance explained", ylim=c(0,1), type='b')
  plot(cumsum(x.pvar),xlab="Principal component", ylab="Cumulative Proportion of variance explained", ylim=c(0,1), type='b')
  screeplot(x)
  screeplot(x,type="l")
  par(mfrow=c(1,1))
}

pcaCharts(pca_results)

###########################################################################

# http://www.sthda.com/english/articles/31-principal-component-methods-in-r-practical-guide/118-principal-component-analysis-in-r-prcomp-vs-princomp/
require(factoextra)

# scree plot
fviz_eig(pca_results, choice = "eigenvalue", geom = "line", addlabels = TRUE, xlab = "Principal Component" )
fviz_eig(pca_results, choice = "variance", geom = "bar", addlabels = TRUE, xlab = "Principal Component" )

# loading plot
fviz_pca_var(pca_results,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)
fviz_pca_var(pca_results,
             label = "none",
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)
fviz_pca_biplot(pca_results, 
                repel = TRUE,
                col.var = "#2E9FDF", # Variables color
                col.ind = "#696969"  # Individuals color
)
fviz_pca_biplot(pca_results,
                label = "none",
                repel = TRUE,
                col.var = "#2E9FDF", # Variables color
                col.ind = "#696969"  # Individuals color
)

# biplot
require(ggbiplot)
ggbiplot(pca_results, ellipse=TRUE, groups=groups, obs.scale = 1, var.scale = 1)
