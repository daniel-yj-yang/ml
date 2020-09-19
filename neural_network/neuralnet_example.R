require("neuralnet")

# Based on https://www.datacamp.com/community/tutorials/neural-network-models-r

# creating training data set
TKS=c(20,10,30,20,80,30) # Technical Knowledge Score
CSS=c(90,20,40,50,50,80) # Communication Skills Score
Placed=c(1,0,0,0,1,1) # Student Placed

# combine multiple columns or features into a single set of data
df=data.frame(TKS,CSS,Placed)

# fit neural network
nn=neuralnet(Placed~TKS+CSS,data=df, hidden=3,act.fct = "logistic",
             linear.output = FALSE)

# plot neural network
plot(nn)

# creating test set
TKS=c(30,40,85)
CSS=c(85,50,40)
test=data.frame(TKS,CSS)

## Prediction using neural network
Predict=compute(nn,test)
Predict$net.result

# Converting probabilities into binary classes setting threshold level 0.5
prob <- Predict$net.result
pred <- ifelse(prob>0.5, 1, 0)
pred
