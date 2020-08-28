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
library(Rtsne)
library(caret)
library(ROSE)

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

# # and SEPARATELY whole scaler to test
# 
# scaler_test <- preProcess(test, method = "scale")
# test_scaled <- predict(scaler_test, test)
# print(table(test_scaled$Class))

############################################
# t-SNE on unbalanced data
############################################

#### Sampling the Unbalanced Data
set.seed(12345)
df.unbalance.sample <- sample_n(train_scaled, 56961) # about 20% of the data
print(table(df.unbalance.sample$Class))

#### Executing the algorithm on balanced data 
tsne_subset <- select(df.unbalance.sample, -Class)
tsne <- Rtsne(tsne_subset, dims = 2, perplexity=29, verbose=TRUE, max_iter = 1000,check_duplicates = F)

#### Plotting the t-SNE
Classes.unb <- as.factor(df.unbalance.sample$Class)
df_tsne <- as.data.frame(tsne$Y)
head(df_tsne)

plot_tsne <- ggplot(df_tsne, aes(x = V1, y = V2)) + 
  geom_point(aes(color = Classes.unb)) + 
  ggtitle("t-SNE visualization of credit card transactions - Unbalanced data") + 
  scale_color_manual(values = c("#AF0000", "#0A0AFF"))+ 
  theme_bw()

plot_tsne


############################################
# t-SNE on undersampled data
############################################

#### Undersampling
set.seed(12345)
train_scaled_under <- ovun.sample(Class ~ ., data = train_scaled, method = "under", N = 492*2, seed = 1)$data
table(train_scaled_under$Class)

#### Executing the algorithm on unbalanced data 
tsne_subset_under <- select(train_scaled_under, -Class)
tsne_under <- Rtsne(tsne_subset_under, dims = 2, perplexity=29, verbose=TRUE, max_iter = 1000,check_duplicates = F)

#### Plotting the t-SNE
Classes <- as.factor(train_scaled_under$Class)
df_tsne_under <- as.data.frame(tsne_under$Y)
head(df_tsne_under)

plot_tsne_under <- ggplot(df_tsne_under, aes(x = V1, y = V2)) + 
  geom_point(aes(color = Classes)) + 
  ggtitle("t-SNE visualization of credit card transactions - Balanced data") + 
  scale_color_manual(values = c("#0A0AFF","#AF0000"))+ 
  theme_bw()

plot_tsne_under

# ## Alternatively with ggplot2 
# tsne_under_plot <- data.frame(x = tsne_under$Y[,1], y = tsne_under$Y[,2], col = train_scaled_under$Class) 
# ggplot(tsne_under_plot) + geom_point(aes(x=x, y=y, color=col))

