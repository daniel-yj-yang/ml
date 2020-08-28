# Disclaimer: the code was improved from a previous version: https://towardsdatascience.com/k-nearest-neighbors-algorithm-with-examples-in-r-simply-explained-knn-1f2c88da405c
# By Daniel Yang, Ph.D. (daniel.yj.yang@gmail.com)

##load data
df <- data(iris) 

## see the studcture
head(iris) 

## Generate a random number that is 90% of the total number of rows in dataset.
set.seed(12345)
ran <- sample(1:nrow(iris), 0.9 * nrow(iris)) 

## Create a normalization function
nor <-function(x) {
  (x-min(x))/(max(x)-min(x)) # Rescaling (min-max normalization), also known as min-max scaling or min-max normalization, is the simplest method and consists in rescaling the range of features to scale the range in [0, 1]
}

## Run nomalization on first 4 coulumns of dataset because they are the predictors
iris_norm <- as.data.frame(lapply(iris[,c(1,2,3,4)], nor))

summary(iris_norm)

## Extract training set
iris_train <- iris_norm[ran,]

## Extract testing set
iris_test <- iris_norm[-ran,]

## Extract 5th column of train dataset because it will be used as 'cl' argument in knn function.
iris_target_category <- iris[ran,5]

## Extract 5th column if test dataset to measure the accuracy
iris_test_category <- iris[-ran,5]

## Load the package class
library(class)

## Run knn function
pr <- knn(iris_train,iris_test,cl=iris_target_category,k=13)

## Create confusion matrix
tab <- table(pr,iris_test_category)

## Divide the correct predictions by total number of predictions to see how accurate the model is.

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy(tab)

### Plot decision bounary of kNN
# https://stackoverflow.com/questions/32449280/how-to-create-a-decision-boundary-graph-for-knn-models-in-the-caret-package

library(caret)

train <- iris[ran,]
test <- iris[-ran,]

knnModel <- train(Species ~.,
                 data = iris[ran,],
                 method = 'knn')

pl = seq(min(test$Petal.Length), max(test$Petal.Length), by=0.1)
pw = seq(min(test$Petal.Width), max(test$Petal.Width), by=0.1)

# generates the boundaries for your graph
lgrid <- expand.grid(Petal.Length=pl, 
                     Petal.Width=pw,
                     Sepal.Length = 5.4,
                     Sepal.Width=3.1)

knnPredGrid <- predict(knnModel, newdata=lgrid)
knnPredGrid = as.numeric(knnPredGrid)

# get the points from the test data...
testPred <- predict(knnModel, newdata=test)
testPred <- as.numeric(testPred)
# this gets the points for the testPred...
test$Pred <- testPred

probs <- matrix(knnPredGrid, length(pl), length(pw))

contour(pl, pw, probs, labels="", xlab="", ylab="", main="Decision Boundary of kNN", axes=F)
gd <- expand.grid(x=pl, y=pw)

points(gd, pch=".", cex=5, col=probs)

# add the test points to the graph
points(test$Petal.Length, test$Petal.Width, col=test$Pred, cex=2)
box()

library(ggplot)

ggplot(data=lgrid) + stat_contour(aes(x=Petal.Length, y=Petal.Width, z=knnPredGrid),
                                  bins=2.1) +
  geom_point(aes(x=Petal.Length, y=Petal.Width, colour=as.factor(knnPredGrid))) +
  geom_point(data=test, aes(x=test$Petal.Length, y=test$Petal.Width, colour=as.factor(test$Pred)),
             size=5, alpha=0.5, shape=1)+
  theme_bw() + ggtitle( "Decision Boundary of kNN" ) + theme(plot.title = element_text(hjust = 0.5))

#####################################################################

## Load diamonds dataset is in ggplot2 package
library(ggplot2)
data(diamonds)

## Store it as data frame
dia <- data.frame(diamonds)

## Create a random number equal 90% of total number of rows
set.seed(12345)
ran <- sample(1:nrow(dia),0.9 * nrow(dia))

## Create a normalization function
nor <-function(x) {
  (x-min(x))/(max(x)-min(x)) # Rescaling (min-max normalization), also known as min-max scaling or min-max normalization, is the simplest method and consists in rescaling the range of features to scale the range in [0, 1]
}

## Run nomalization on selected coulumns of dataset because they are the predictors
dia_nor <- as.data.frame(lapply(dia[,c(1,5,6,7,8,9,10)], nor))

## Extract training dataset
dia_train <- dia_nor[ran,]

## Extract test dataset
dia_test <- dia_nor[-ran,]

## the 2nd column of training dataset because that is what we need to predict about testing dataset
## also convert ordered factor to normal factor
dia_target <- as.factor(dia[ran,2])

## the actual values of 2nd couln of testing dataset to compaire it with values that will be predicted
## also convert ordered factor to normal factor
test_target <- as.factor(dia[-ran,2])

## Run knn function
library(class)
pr <- knn(dia_train,dia_test,cl=dia_target,k=20)

## Create the confucion matrix
tb <- table(pr,test_target)

## Check the accuracy
accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy(tb)
