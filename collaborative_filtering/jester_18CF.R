
# Disclaimer: the code was originally based on another code (https://rpubs.com/jt_rpubs/285729),
# while I have made several improvements to the code 
# By Daniel Yang, Ph.D. (daniel.yj.yang@gmail.com) @ 2020

require(data.table)
require(magrittr)
require(knitr)

# load CSV version of jester subset
setwd("/Users/daniel/Data-Science/Data/Jester_Joke/dataset1/")
if (!file.exists("jester-data-3.rds")) {
  jester <- read.csv("/Users/daniel/Data-Science/Data/Jester_Joke/dataset1/jester-data-3.csv", header = FALSE)
  saveRDS(as.data.table(jester), "jester-data-3.rds")
  rm("jester")
}
jester_dt <- readRDS("jester-data-3.rds")

dim(jester_dt)

summary(jester_dt[,V1])

# The first column gives the number of jokes rated by that user
# remove first column since it does not contain user ratings
jester_dt[,V1:=NULL] 

# set all 99's to NA
jester_dt[ jester_dt == 99 ] <- NA

# .SD means subset of data
# means by columns
# see the highest rated jokes (e.g., V7 -> joke#6, V12 -> joke#11, V26 -> joke#25)
jester_dt[, lapply(.SD, mean, na.rm=TRUE) ] %>% sort()
colMeans(jester_dt, na.rm = TRUE) %>% sort()

jester_dt[,min(.SD,na.rm=T)]

jester_dt[,max(.SD,na.rm=T)]

hist(as.vector(as.matrix(jester_dt)), main = "Distribution of Jester Ratings",
     col = "yellow", xlab = "Ratings")

boxplot(as.vector(as.matrix(jester_dt)), col = "yellow", main = "Distribution of Jester Ratings", ylab = "Ratings")

summary(as.vector(as.matrix(jester_dt)))

average_ratings_per_user <- rowMeans(jester_dt, na.rm = TRUE)
hist(average_ratings_per_user, main = "Distribution of the average rating per user",
     col = "yellow")

# Creating Training and Testing Subsets

rmat <- as.matrix(jester_dt)

require(recommenderlab)
# convert matrix to a recommenderlab realRatingMatrix
rmat <- as(rmat,"realRatingMatrix")

# split the data into the training and the test set:
set.seed(12345)
e <- evaluationScheme(rmat, method="split", train=0.8, given=15, goodRating=0)

#####################################################################################

# The Recommendation Algorithms

# Six combinations:
# (User-Based, Item-Based) Collaborative Filtering: (Cosine Similarity, Euclidean Distance, Pearson Correlation)

# 1. User-Based Collaborative Filtering: Cosine Similarity

#train UBCF cosine similarity models

# non-normalized
UBCF_N_C <- Recommender(getData(e, "train"), "UBCF", 
                        param=list(normalize = NULL, method="Cosine"))

# centered
UBCF_C_C <- Recommender(getData(e, "train"), "UBCF", 
                        param=list(normalize = "center",method="Cosine"))

# Z-score normalization
UBCF_Z_C <- Recommender(getData(e, "train"), "UBCF", 
                        param=list(normalize = "Z-score",method="Cosine"))

# compute predicted ratings
p1 <- predict(UBCF_N_C, getData(e, "known"), type="ratings")
p2 <- predict(UBCF_C_C, getData(e, "known"), type="ratings")
p3 <- predict(UBCF_Z_C, getData(e, "known"), type="ratings")

# set all predictions that fall outside the valid range to the boundary values
p1@data@x[p1@data@x[] < -10] <- -10
p1@data@x[p1@data@x[] > 10] <- 10

p2@data@x[p2@data@x[] < -10] <- -10
p2@data@x[p2@data@x[] > 10] <- 10

p3@data@x[p3@data@x[] < -10] <- -10
p3@data@x[p3@data@x[] > 10] <- 10

# aggregate the performance statistics
error_UCOS <- rbind(
  UBCF_N_C = calcPredictionAccuracy(p1, getData(e, "unknown")),
  UBCF_C_C = calcPredictionAccuracy(p2, getData(e, "unknown")),
  UBCF_Z_C = calcPredictionAccuracy(p3, getData(e, "unknown"))
)
kable(error_UCOS)

boxplot(as.vector(as(p3, "matrix")), col = "yellow", main = "Distribution of Predicted Values for UBCF Z-Score/Cosine Model", ylab = "Ratings")
hist(as.vector(as(p3, "matrix")), main = "Distrib. of Predicted Values for UBCF Z-Score/Cosine Model", col = "yellow", xlab = "Predicted Ratings")
summary(as.vector(as.matrix(jester_dt)))
summary(as.vector(p3@data@x))

#####################################################################################

# 2. User-Based Collaborative Filtering: Euclidean Distance

#train UBCF Euclidean Distance models

# non-normalized
UBCF_N_E <- Recommender(getData(e, "train"), "UBCF", 
                        param=list(normalize = NULL, method="Euclidean"))

# centered
UBCF_C_E <- Recommender(getData(e, "train"), "UBCF", 
                        param=list(normalize = "center",method="Euclidean"))

# Z-score normalization
UBCF_Z_E <- Recommender(getData(e, "train"), "UBCF", 
                        param=list(normalize = "Z-score",method="Euclidean"))

# compute predicted ratings
rm(p1, p2, p3)
p1 <- predict(UBCF_N_E, getData(e, "known"), type="ratings")
p2 <- predict(UBCF_C_E, getData(e, "known"), type="ratings")
p3 <- predict(UBCF_Z_E, getData(e, "known"), type="ratings")

# set all predictions that fall outside the valid range to the boundary values
p1@data@x[p1@data@x[] < -10] <- -10
p1@data@x[p1@data@x[] > 10] <- 10

p2@data@x[p2@data@x[] < -10] <- -10
p2@data@x[p2@data@x[] > 10] <- 10

p3@data@x[p3@data@x[] < -10] <- -10
p3@data@x[p3@data@x[] > 10] <- 10

# aggregate the performance statistics
error_UEUC <- rbind(
  UBCF_N_E = calcPredictionAccuracy(p1, getData(e, "unknown")),
  UBCF_C_E = calcPredictionAccuracy(p2, getData(e, "unknown")),
  UBCF_Z_E = calcPredictionAccuracy(p3, getData(e, "unknown"))
)
kable(error_UEUC)

#####################################################################################

# 3. User-Based Collaborative Filtering: Pearson Correlation

#train UBCF pearson correlation models

# non-normalized
UBCF_N_P <- Recommender(getData(e, "train"), "UBCF", 
                        param=list(normalize = NULL, method="pearson"))

# centered
UBCF_C_P <- Recommender(getData(e, "train"), "UBCF", 
                        param=list(normalize = "center",method="pearson"))

# Z-score normalization
UBCF_Z_P <- Recommender(getData(e, "train"), "UBCF", 
                        param=list(normalize = "Z-score",method="pearson"))

# compute predicted ratings
rm(p1, p2, p3)
p1 <- predict(UBCF_N_P, getData(e, "known"), type="ratings") # If 'Error in neighbors[, x] : incorrect number of dimensions', try set.seed() and rerun evaluationScheme. Could be random seed issue.
p2 <- predict(UBCF_C_P, getData(e, "known"), type="ratings") # If 'Error in neighbors[, x] : incorrect number of dimensions', try set.seed() and rerun evaluationScheme. Could be random seed issue.
p3 <- predict(UBCF_Z_P, getData(e, "known"), type="ratings") # If 'Error in neighbors[, x] : incorrect number of dimensions', try set.seed() and rerun evaluationScheme. Could be random seed issue.

# set all predictions that fall outside the valid range to the boundary values
p1@data@x[p1@data@x[] < -10] <- -10
p1@data@x[p1@data@x[] > 10] <- 10

p2@data@x[p2@data@x[] < -10] <- -10
p2@data@x[p2@data@x[] > 10] <- 10

p3@data@x[p3@data@x[] < -10] <- -10
p3@data@x[p3@data@x[] > 10] <- 10

# aggregate the performance statistics
error_UPC <- rbind(
  UBCF_N_P = calcPredictionAccuracy(p1, getData(e, "unknown")),
  UBCF_C_P = calcPredictionAccuracy(p2, getData(e, "unknown")),
  UBCF_Z_P = calcPredictionAccuracy(p3, getData(e, "unknown"))
)
kable(error_UPC)

#####################################################################################

# 4. Item-Based Collaborative Filtering: Cosine Similarity

#train IBCF cosine similarity models

# non-normalized
IBCF_N_C <- Recommender(getData(e, "train"), "IBCF", 
                        param=list(normalize = NULL, method="Cosine"))

# centered
IBCF_C_C <- Recommender(getData(e, "train"), "IBCF", 
                        param=list(normalize = "center",method="Cosine"))

# Z-score normalization
IBCF_Z_C <- Recommender(getData(e, "train"), "IBCF", 
                        param=list(normalize = "Z-score",method="Cosine"))

# compute predicted ratings
rm(p1, p2, p3)
p1 <- predict(IBCF_N_C, getData(e, "known"), type="ratings")
p2 <- predict(IBCF_C_C, getData(e, "known"), type="ratings")
p3 <- predict(IBCF_Z_C, getData(e, "known"), type="ratings")

# set all predictions that fall outside the valid range to the boundary values
p1@data@x[p1@data@x[] < -10] <- -10
p1@data@x[p1@data@x[] > 10] <- 10

p2@data@x[p2@data@x[] < -10] <- -10
p2@data@x[p2@data@x[] > 10] <- 10

p3@data@x[p3@data@x[] < -10] <- -10
p3@data@x[p3@data@x[] > 10] <- 10

# aggregate the performance statistics
error_ICOS <- rbind(
  IBCF_N_C = calcPredictionAccuracy(p1, getData(e, "unknown")),
  IBCF_C_C = calcPredictionAccuracy(p2, getData(e, "unknown")),
  IBCF_Z_C = calcPredictionAccuracy(p3, getData(e, "unknown"))
)

kable(error_ICOS)

boxplot(as.vector(as(p1, "matrix")), col = "yellow", main = "Distribution of Predicted Values for IBCF Raw/Cosine Model", ylab = "Ratings")

hist(as.vector(as(p1, "matrix")), main = "Distrib. of Predicted Values for IBCF Raw/Cosine Model", col = "yellow", xlab = "Predicted Ratings")

summary(as.vector(as.matrix(jester_dt)))

summary(as.vector(p1@data@x))

#####################################################################################

# 5. Item-Based Collaborative Filtering: Euclidean Distance

#train IBCF Euclidean Distance models

# non-normalized
IBCF_N_E <- Recommender(getData(e, "train"), "IBCF", 
                        param=list(normalize = NULL, method="Euclidean"))

# centered
IBCF_C_E <- Recommender(getData(e, "train"), "IBCF", 
                        param=list(normalize = "center",method="Euclidean"))

# Z-score normalization
IBCF_Z_E <- Recommender(getData(e, "train"), "IBCF", 
                        param=list(normalize = "Z-score",method="Euclidean"))

# compute predicted ratings
rm(p1, p2, p3)
p1 <- predict(IBCF_N_E, getData(e, "known"), type="ratings")
p2 <- predict(IBCF_C_E, getData(e, "known"), type="ratings")
p3 <- predict(IBCF_Z_E, getData(e, "known"), type="ratings")

# set all predictions that fall outside the valid range to the boundary values
p1@data@x[p1@data@x[] < -10] <- -10
p1@data@x[p1@data@x[] > 10] <- 10

p2@data@x[p2@data@x[] < -10] <- -10
p2@data@x[p2@data@x[] > 10] <- 10

p3@data@x[p3@data@x[] < -10] <- -10
p3@data@x[p3@data@x[] > 10] <- 10

# aggregate the performance statistics
error_IEUC <- rbind(
  IBCF_N_E = calcPredictionAccuracy(p1, getData(e, "unknown")),
  IBCF_C_E = calcPredictionAccuracy(p2, getData(e, "unknown")),
  IBCF_Z_E = calcPredictionAccuracy(p3, getData(e, "unknown"))
)
kable(error_IEUC)

#####################################################################################

# 6. Item-Based Collaborative Filtering: Pearson Correlation

#train IBCF pearson correlation models

# non-normalized
IBCF_N_P <- Recommender(getData(e, "train"), "IBCF", 
                        param=list(normalize = NULL, method="pearson"))

# centered
IBCF_C_P <- Recommender(getData(e, "train"), "IBCF", 
                        param=list(normalize = "center",method="pearson"))

# Z-score normalization
IBCF_Z_P <- Recommender(getData(e, "train"), "IBCF", 
                        param=list(normalize = "Z-score",method="pearson"))

# compute predicted ratings
rm(p1, p2, p3)
p1 <- predict(IBCF_N_P, getData(e, "known"), type="ratings")
p2 <- predict(IBCF_C_P, getData(e, "known"), type="ratings")
p3 <- predict(IBCF_Z_P, getData(e, "known"), type="ratings")

# set all predictions that fall outside the valid range to the boundary values
p1@data@x[p1@data@x[] < -10] <- -10
p1@data@x[p1@data@x[] > 10] <- 10

p2@data@x[p2@data@x[] < -10] <- -10
p2@data@x[p2@data@x[] > 10] <- 10

p3@data@x[p3@data@x[] < -10] <- -10
p3@data@x[p3@data@x[] > 10] <- 10

# aggregate the performance statistics
error_IPC <- rbind(
  IBCF_N_P = calcPredictionAccuracy(p1, getData(e, "unknown")),
  IBCF_C_P = calcPredictionAccuracy(p2, getData(e, "unknown")),
  IBCF_Z_P = calcPredictionAccuracy(p3, getData(e, "unknown"))
)
kable(error_IPC)

#####################################################################################

# Conclusions

c_res <- data.frame(rbind(error_UCOS, error_UEUC, error_UPC, error_ICOS, error_IEUC, error_IPC))

c_res <- c_res[order(c_res$RMSE ),]

kable(c_res)

# las = 3: rotate x axis labels to perendicular; las = 1: rotate y axis labels
barplot(c_res$RMSE, col = "lightblue", main = "Barplot of Model RMSE's", las = 2, ylab = "RMSE", horiz = FALSE, names.arg = rownames(c_res), cex.names=.8)
