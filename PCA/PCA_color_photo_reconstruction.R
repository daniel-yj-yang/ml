rm(list=ls())
cat("\014")

# Daniel Yang, Ph.D. (daniel.yj.yang@gmail.com)

require(imager) # https://dahtah.github.io/imager/imager.html

x <- load.image("/Users/daniel/Data-Science/Data/Photos/Lego_art.png")
img_width <- 256
img_height <- 414
x

plot_array_RGB <- function(this_array_RGB, filename = 'test.png', text = "", width=dim(this_array)[1], height=dim(this_array)[2]) {
  png(file.path('/Users/daniel/tmp', filename), width=width, height=height)
  par(mar = rep(0, 4)) # Set up a plot with no margin
  this_img <- as.cimg(this_array_RGB)
  this_img <- draw_text( im = this_img, x = 50, y = 40, text = text, color = "yellow", fsize = 50 )
  plot(this_img)
  dev.off()
}

plot_array_RGB( x[,,1,1:3], text = "Original", width = img_width, height = img_height )

# to use PCA to reconstruct 

width = dim(x)[1]
height = dim(x)[2]

X_R = x[,,1,1]
X_G = x[,,1,2]
X_B = x[,,1,3]

mu_R = colMeans(X_R)
mu_G = colMeans(X_G)
mu_B = colMeans(X_B)

X_PCA_R = prcomp(X_R, center = TRUE, scale. = FALSE)
X_PCA_G = prcomp(X_G, center = TRUE, scale. = FALSE)
X_PCA_B = prcomp(X_B, center = TRUE, scale. = FALSE)

for(n_PCs in 1:50) {
  print(n_PCs)
  nComp = n_PCs
  Xhat_R = X_PCA_R$x[,1:nComp] %*% t(X_PCA_R$rotation[,1:nComp]) # X_L = T_L * W_L
  Xhat_G = X_PCA_G$x[,1:nComp] %*% t(X_PCA_G$rotation[,1:nComp]) # X_L = T_L * W_L
  Xhat_B = X_PCA_B$x[,1:nComp] %*% t(X_PCA_B$rotation[,1:nComp]) # X_L = T_L * W_L
  Xhat_R = scale(Xhat_R, center = -mu_R, scale = FALSE) # reconstructed
  Xhat_G = scale(Xhat_G, center = -mu_G, scale = FALSE) # reconstructed
  Xhat_B = scale(Xhat_B, center = -mu_B, scale = FALSE) # reconstructed
  Xhat_RGB = array(NA, dim=c(width, height, 3))
  Xhat_RGB[,,1] <- Xhat_R
  Xhat_RGB[,,2] <- Xhat_G
  Xhat_RGB[,,3] <- Xhat_B
  plot_array_RGB(Xhat_RGB, filename = paste0(sprintf("%03d", nComp),'.png'), text = paste0('PCs: ',nComp), width = img_width, height = img_height)
}

# finally, use imagemagick to create gif
# e.g., convert -delay 20 -loop 0 *.png all.gif  (delaying 20/100 = 0.20 seconds per frame and infinite loop)
