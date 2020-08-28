# Disclaimer: The code was improved from a previous version: https://rpubs.com/tahamokfi/Part2_AnalyzeTransactionData
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
library(stringr)

rawdata_dt <- as.data.table(rawdata)

# keep only regular items, namely, having numbers and no letters in StockCode
# which removes non-typical items, such as c('M','POST','PADS','S','D','BANK CHARGES')
# M = Manual, POST = Postage, AMAZONFEE = Amazon Fee, Pads = Pads to match all cushions, S = Samples, D = Discount
dt1 <- rawdata_dt[str_detect(StockCode, '[:digit:]{3,}') & (!str_detect(StockCode, '[:alpha:]')), .(InvoiceNo, StockCode, InvoiceDate, CustomerID, Country)] %>% as.data.frame()

N_transactions <- nrow(dt1)
sampling_percentage <- 0.10 # to make it more manageable, randomly sampling 10% of the data

set.seed(12345)
dt1 <- dt1[sample(N_transactions, trunc(N_transactions * sampling_percentage), replace = F),] 

#Create a list of customer ID (entity ID)
un2 <- unique(dt1$CustomerID)
#Order the data based on entity and date variable (customer and invoice date)
dt2 <- dt1[order(dt1$CustomerID,dt1$InvoiceDate),]

#For loop for creating transition matrix
edg3=c()
for (i in 1:length(un2)){
  un3 <- base::unique(dt2[which(dt2$CustomerID==un2[i]),]$InvoiceDate)
  #print(paste0("Running ",i," of the ",length(un2)))
  edg2=c()
  for (j in 1:length(un3)-1){
    lis1 <- dt2[which(dt2$CustomerID==un2[i] & dt2$InvoiceDate==un3[j]),2]
    lis2 <- dt2[which(dt2$CustomerID==un2[i] & dt2$InvoiceDate==un3[j+1]),2]
    edg1 <- expand.grid(lis1,lis2)
    edg2 <- rbind(edg1,edg2)
  }
  edg3 <- rbind(edg2,edg3)
}

#Creating graph from the matrix
library(igraph)
g1 <- graph.data.frame(edg3,directed=FALSE)
adj1 <- as_adjacency_matrix(g1,type="both",names=TRUE,sparse=FALSE)
#Afjacency matrix
g1 <- graph.adjacency(as.matrix(adj1),mode="directed",weighted=TRUE)
#Compute the betweeenness using igraph
cen1 <- igraph::betweenness(g1, v = V(g1), directed = TRUE)
#Top 10 betweenness
names(sort(cen1,decreasing = T)[1:10])

#Clustering the graph using spinglass techniques
set.seed(12345)
cm1 <- cluster_spinglass(g1)
#Members of each community
table(cm1$membership)

grclus <- as.data.frame(cbind(cm1$names,cm1$membership))
#Filter the nodes which are in the community 1
ver1 <- as.character(grclus[which(grclus$V2 %in% c(1)),1])

#Compute the centrality for the first community (cluster)
g2 <- induced_subgraph(g1,v = ver1)
cen2 <- igraph::betweenness(g2, v = V(g2), directed = TRUE)
top1 <- names(sort(cen2,decreasing = T)[1:7])
g3 <- adj1[which(rownames(adj1) %in% top1),which(colnames(adj1) %in% top1)]

library(sna)
net1 <- network(x = g3,directed = T)
set.seed(12345)
gplot(net1,gmode="digraph",displaylabels = T,object.scale=0.009,label.cex = 1.1,edge.col = "Blue")
