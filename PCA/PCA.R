# Daniel Yang, Ph.D. (daniel.yj.yang@gmail.com)

# https://www.datacamp.com/community/tutorials/pca-analysis-r

require(datasets)
data(iris)

iris.pca <- prcomp(iris[,c(1:4)], center = TRUE, scale. = TRUE)

summary(iris.pca)

require(magrittr)
iris.pca$rotation ^2 %>% colSums() # the eigenvector is normalized to have magnitude/length of 1

require(ggbiplot)
ggbiplot(iris.pca, ellipse=TRUE, groups=iris$Species, obs.scale = 1, var.scale = 1)

# http://www.sthda.com/english/articles/31-principal-component-methods-in-r-practical-guide/118-principal-component-analysis-in-r-prcomp-vs-princomp/
require(factoextra)
fviz_eig(iris.pca)
fviz_eig(iris.pca, choice = "eigenvalue", geom = "line", addlabels = TRUE, xlab = "Principal Component" )
fviz_eig(iris.pca, choice = "variance", geom = "bar", addlabels = TRUE, xlab = "Principal Component" )
fviz_pca_var(iris.pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)
fviz_pca_biplot(iris.pca, 
                repel = TRUE,
                col.var = "#2E9FDF", # Variables color
                col.ind = "#696969"  # Individuals color
)