# Disclaimer: the original code was from https://github.com/ihh300/Fraud_Detection
#             I have made several modifications to the code during exploration
#             By Daniel Yang, Ph.D. (daniel.yj.yang@gmail.com) @ 2020

####################################################################################################################################
####################################################################################################################################

library(tidyverse)
library(ggplot2)
library(ggthemes)
library(gridExtra)
library(grid)
library(caret)
library(ROSE)
library(umap)

# importing the dataset
setwd('/Users/daniel/Data-Science/Data/Finance/CreditCard-Transactions')
if (!file.exists('creditcard.rds')) {
  df <- read.csv("creditcard.csv")
  df <- rename(df, Class_old = Class) %>%
    mutate(df, Class = ifelse(Class_old == 1, 'Fraud', 'Legitimate')) %>%
    select(-Class_old)
  saveRDS(df, 'creditcard.rds')
}
df <- readRDS('creditcard.rds')


##################################################################################################
# https://cran.r-project.org/web/packages/umap/vignettes/umap.html
##################################################################################################
plot_umap <- function(x, labels,
                      main="A UMAP visualization",
                      colors=c("#ff7f00", "#e377c2", "#17becf"),
                      pad=0.1, cex=0.65, pch=19, legend.suffix="",
                      cex.main=1, cex.legend=1,
                      x_coord = 10, y_coord = 15) {
  
  layout = x
  if (is(x, "umap")) {
    layout = x$layout
  } 
  
  xylim = range(layout)
  xylim = xylim + ((xylim[2]-xylim[1])*pad)*c(-0.5, 0.5)
  
  par(mar=c(0.2,0.7,1.2,0.7), ps=10)
  plot(xylim, xylim, type="n", axes=F, frame=F)
  rect(xylim[1], xylim[1], xylim[2], xylim[2], border="#000000", lwd=0.25)  
  
  points(layout[,1], layout[,2], col=colors[as.integer(labels)],
         cex=cex, pch=pch)
  mtext(side=3, main, cex=cex.main)
  
  labels.u = unique(labels)
  legend.text = as.character(labels.u)
  legend(x_coord, y_coord, legend=legend.text,
         col=colors[as.integer(labels.u)],
         bty="n", pch=pch, cex=cex.legend)
}

##################################################################################################
#Data preparation
##################################################################################################
# Factorise
df  <- mutate(df, Class = factor(Class, levels = c('Fraud','Legitimate')))
str(df$Class)
print(table(df$Class))

# Drop variables Time 
df <- select(df, -Time)
summary(df)

##################################################################################################
# Train / Test / Split
##################################################################################################

train <- df # use the whole dataset

##################################################################################################
# PRE-MODELING DATA PREPARATION
##################################################################################################


# whole scaler to train
scaler_train <- preProcess(train, method = "scale")
train_scaled <- predict(scaler_train,train)
print(table(train_scaled$Class))

############################################
# UMAP on unbalanced data
############################################

#### Sampling the Unbalanced Data
set.seed(12345)
df.unbalance.sample <- sample_n(train_scaled, 56961) # about 20% of the data
print(table(df.unbalance.sample$Class))

#### Executing the algorithm on balanced data 
umap_subset <- select(df.unbalance.sample, -Class)
umap_unbalanced <- umap(umap_subset)

#### Plotting the UMAP
Classes.unb <- as.factor(df.unbalance.sample$Class)
plot_umap(x = umap_unbalanced, 
          labels = Classes,
          main = "UMAP visualization of credit card transactions - Unbalanced data",
          colors = c("#0A0AFF","#AF0000"),
          x_coord = 11, y_coord = 16)

############################################
# UMAP on undersampled/balanced data
############################################

#### Undersampling
set.seed(12345)
train_scaled_under <- ovun.sample(Class ~ ., data = train_scaled, method = "under", N = 492*2, seed = 1)$data
table(train_scaled_under$Class)

#### Executing the algorithm on unbalanced data 
umap_subset_under <- select(train_scaled_under, -Class)
umap_balanced <- umap(umap_subset_under)

#### Plotting the UMAP
Classes <- as.factor(train_scaled_under$Class)
plot_umap(x = umap_balanced, 
          labels = Classes,
          main = "UMAP visualization of credit card transactions - Balanced data",
          colors = c("#0A0AFF","#AF0000"),
          x_coord = 11, y_coord = 16)
