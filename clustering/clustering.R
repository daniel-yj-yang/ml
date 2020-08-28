# Disclaimer: The code was improved from a previous version: https://rpubs.com/tahamokfi/Part1_AnalyzeTransactionData
# By Daniel Yang, Ph.D. (daniel.yj.yang@gmail.com)

# Column description
# InvoiceNo: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation.
# StockCode: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.
# Description: Product (item) name. Nominal.
# Quantity: The quantities of each product (item) per transaction. Numeric.
# InvoiceDate: Invice Date and time. Numeric, the day and time when each transaction was generated.
# UnitPrice: Unit price. Numeric, Product price per unit in sterling.
# CustomerID: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.
# Country: Country name. Nominal, the name of the country where each customer resides.

rawdata <- read.csv("/Users/daniel/Data-Science/Data/Retail/Online_Retail/Online Retail.csv")
rawdata$InvoiceDate <- as.POSIXct(rawdata$InvoiceDate, format = "%m/%d/%y %H:%M")
library(data.table)
library(magrittr)
rawdata_dt <- as.data.table(rawdata)
dt1 <- rawdata_dt[! StockCode %in% c('M','POST','PADS','S','D'), .(InvoiceNo, StockCode, InvoiceDate, CustomerID, Country)] %>% as.data.frame()
dt1 <- dt1[1:100000,] # to make it more manageable, about 20% of the data

#Reshape the dataset
library(reshape2)
mlt1 = melt(dt1,id.vars =  names(dt1)[c(1,3:5)])
dt2 <- dcast(mlt1, CustomerID  ~ value, value.var = c("CustomerID"),fun.aggregate =length)
#Remove first variable containing the CustomerID
dt3 <- dt2[, -1]

#Transform dataset
for (i in 1:ncol(dt3)){
  dt3[,i] <- ifelse(dt3[,i]==0,0,1)
}
#Apply PCA
for (i in 1:ncol(dt3)){
  dt3[, i] <- as.numeric(dt3[, i])
}
pcaout <- prcomp(dt3)
pcdt <- pcaout$x[,1:6]

#Clustering Function
kmcl <- function(data, kmin, kmax, rnd = 12341){
  library(clusterCrit)
  DB <- c()
  for (i in 1:ncol(data))
  {
    data[, i] <- as.numeric(data[, i])
  }
  for (j in kmin:kmax)
  {
    set.seed(rnd)
    km <- kmeans(data, centers = j)
    data1 <- as.matrix(data)
    # Computing the Davies-Bouldin
    DB[j] <- intCriteria(data1, km$cluster, crit = c("Davies_Bouldin"))
    #print(paste0("K is equal to= ",j))
  }
  return(DB)
}

#Davies-Bouldin Plot
dbplot <- function(obj, tit1 = "Null"){
  plot(2:15, unlist(obj), type = "o", col = "black", ylim = c(0, 3), 
       lwd = 2, ylab = "DB Vlaue", xlab = "Number of Clusters", cex.lab = 1.5, 
       cex.axis = 0.75)
  grid(nx = 25, ny = 25, col = "lightgray", lty = "dotted", lwd = par("lwd"), 
       equilogs = TRUE)
  axis(side = 1, at = seq(2, 15, by = 1), cex.axis = 0.75)
  box(col = "black", lwd = 2)
  title(tit1)
}

#Apply clustering for different K
clusdb <- kmcl(pcdt,2,15,rnd = 12340)
#Apply DB plot
dbplot(clusdb,"Davies Bouldin for Clustering")

#Apply K-means with the best # of clusters
set.seed(12340)
n_clusters <- 3
km1 <- kmeans(pcdt,centers = n_clusters)
table(km1$cluster)

#Adding cluster
rec.inv <- cbind(dt2, km1$cluster)[, c(1, length(dt2)+1)]
names(rec.inv) <- c("CustomerID", "invCluster")
#Merging with rest of variables
dt3 <- unique(dt1[,c(4,5)])
dt4 <- merge(dt3,rec.inv,by="CustomerID")
names(dt4)[3] <- c("Cluster")
#Ploting the cluster VS country
tb2 <- as.data.frame.matrix(table(dt4$Country,dt4$Cluster))
tb1 <- as.data.frame(table(dt4$Country,dt4$Cluster))

#Normalization
norm1 <- function(x){
  (x - min(x, na.rm = TRUE))/(max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
}
tb1$col1 <- as.vector(as.matrix(as.data.frame(lapply(tb2, norm1))))
tb1$row1 <- as.vector(as.matrix(t(as.data.frame(lapply(as.data.frame(t(tb2)), norm1)))))
tb1$Var1 <- as.factor(tb1$Var1)
tb1$Var2 <- as.factor(tb1$Var2)
library(ggplot2)
ggplot(tb1, aes(Var2,Var1))+geom_point(aes(alpha=row1,size=row1),colour = "blue")+scale_size_continuous(range = c(0.5,5))+
  labs(x="Invoice Clusters",y="Countries",size="Frequency",alpha="Frequency") +
  ggtitle("Bubble plot of customer clusters and countries - Normalized by country")+
  theme(axis.text.x = element_text(hjust=1,vjust=0.5,size = 12,face = "bold",color = "Black"),
        plot.title = element_text(hjust=0.5,size = 14,face = "bold",color = "Black"),
        axis.title.x =element_text(hjust=0.5,size = 16,face = "bold",color = "Black"),
        axis.title.y =element_text(hjust=0.5,size = 16,face = "bold",color = "Black"),
        legend.title = element_text(hjust=0.5,size = 14,face = "bold",color = "Black"),
        legend.text = element_text(hjust=0.5,size = 10,face = "bold",color = "Black"))

View(tb2)

#cluster based on countries
clcnt <- kmcl(tb2,2,15,rnd = 12345)
#Plot Davies Bouldin
dbplot(clcnt,"Davies Bouldin for Clustering")

#Apply best K 
set.seed(12345)
km2 <- kmeans(tb2,centers = 2)

contcl.data <- cbind(tb2,km2$cluster)
names(contcl.data)[ncol(contcl.data)] <- "cluster"
contcl.data <- contcl.data[order(contcl.data$cluster),]
or1 <-rownames(contcl.data)
#Sort con
tb1$Var1 <- factor(tb1$Var1 , levels = or1)
tb1$clus <- rep(as.integer(km2$cluster),n_clusters)
ggplot(tb1, aes(Var2,Var1))+geom_point(aes(alpha=row1,size=row1,colour = as.factor(tb1$clus)))+scale_size_continuous(range = c(0.5,5))+
  labs(x="Invoice Clusters",y="Countries",size="Frequency",alpha="Frequency") +
  ggtitle("Bubble plot of customer clusters and countries - Normalized by country")+
  theme(axis.text.x = element_text(hjust=1,vjust=0.5,size = 12,face = "bold",color = "Black"),
        plot.title = element_text(hjust=0.5,size = 14,face = "bold",color = "Black"),
        axis.title.x =element_text(hjust=0.5,size = 16,face = "bold",color = "Black"),
        axis.title.y =element_text(hjust=0.5,size = 16,face = "bold",color = "Black"),
        legend.title = element_text(hjust=0.5,size = 14,face = "bold",color = "Black"),
        legend.text = element_text(hjust=0.5,size = 10,face = "bold",color = "Black"))
