rm(list=ls())
cat("\014")

# Daniel Yang, Ph.D. (daniel.yj.yang@gmail.com)

require(png)
require(colorspace)

# to read the test image as a matrix
# https://stackoverflow.com/questions/31800687/how-to-get-a-pixel-matrix-from-grayscale-image-in-r
x <- readPNG("/Users/daniel/Data-Science/Data/Faces/Lena/Lena-gray-512x512.png")
dim(x)

y <- rgb(x[,,1], x[,,2], x[,,3], alpha = x[,,4])
yg <- desaturate(y)
yn_255 <- col2rgb(yg)[1, ]
yn <- yn_255/255
dim(y) <- dim(yg) <- dim(yn) <- dim(yn_255) <- dim(x)[1:2]

pixmatplot <- function (x, ...) {
  d <- dim(x)
  xcoord <- t(expand.grid(1:d[1], 1:d[2]))
  xcoord <- t(xcoord/d)
  par(mar = rep(1, 4))
  plot(0, 0, type = "n", xlab = "", ylab = "", axes = FALSE, 
       xlim = c(0, 1), ylim = c(0, 1), ...)
  rect(xcoord[, 2L] - 1/d[2L], 1 - (xcoord[, 1L] - 1/d[1L]), 
       xcoord[, 2L], 1 - xcoord[, 1L], col = x, border = "transparent")
}

#pixmatplot(y)
#pixmatplot(yg)

plot_matrix_gray_image <- function(this_array, filename = 'test.png', text = "", width=dim(this_array)[1], height=dim(this_array)[2]) {
  # https://stackoverflow.com/questions/5638462/r-image-of-a-pixel-matrix
  png(file.path('/Users/daniel/tmp', filename), width=width, height=height)
  par(mar = rep(0, 4)) # Set up a plot with no margin
  image_reoriented <- t(apply(this_array, 2, rev)) # https://stackoverflow.com/questions/31882079/r-image-plots-matrix-rotated
  image(image_reoriented, axes = FALSE, col = grey(seq(0, 1, length = 256)))
  text(0.15,0.95,text,col = "yellow",cex = 1.5)
  dev.off()
}

plot_matrix_gray_image(yn_255, text = "Original", width = 256, height = 256)

# to use PCA to reconstruct Lena
# https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com
X = yn_255
mu = colMeans(X)

Xpca = prcomp(X, center = TRUE, scale. = FALSE)

for(n_PCs in 1:50) {
  print(n_PCs)
  nComp = n_PCs
  Xhat = Xpca$x[,1:nComp] %*% t(Xpca$rotation[,1:nComp])
  Xhat = scale(Xhat, center = -mu, scale = FALSE) # reconstructed
  
  plot_matrix_gray_image(Xhat, filename = paste0(sprintf("%03d", nComp),'.png'), text = paste0('PCs: ',nComp), width = 256, height = 256)
}

# finally, use imagemagick to create gif
# https://www.r-graph-gallery.com/animation.html
# e.g., convert -delay 20 -loop 0 *.png all.gif  (delaying 20/100 = 0.20 seconds per frame and infinite loop)

### To figure how many PCs to use
# scree plot
require(factoextra)
data = yn_255
pca_results <- prcomp(data, center = TRUE, scale. = TRUE)
fviz_eig(pca_results, choice = "eigenvalue", geom = "line", addlabels = TRUE, xlab = "Principal Component" )
fviz_eig(pca_results, choice = "variance", geom = "bar", addlabels = TRUE, xlab = "Principal Component" )
