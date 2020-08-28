mydata <- read.csv("/Users/Daniel/Documents/GitHub/logistic-regression/Examples/Passing-exam/Passing-exam.csv")
mymodel <- glm(Pass ~ Hours, data = mydata, family=binomial(link="logit"))
summary(mymodel)
#predict(mymodel) # this gives # type: the type of prediction required. The default is on the scale of the linear predictors; the alternative "response" is on the scale of the response variable
predict(mymodel,type="resp") # this also gives prob(passing the exam) # type: the type of prediction required. The default is on the scale of the linear predictors; the alternative "response" is on the scale of the response variable
fitted(mymodel) # this gives prob(passing the exam)

odds = fitted(mymodel) / (1-fitted(mymodel))
odds
exp(predict(mymodel)) # the same


beta0 = mymodel$coefficients[1] # intercept
beta1 = mymodel$coefficients[2] # slope
aic = mymodel$aic # AIC
# https://www.analyticsvidhya.com/blog/2015/11/beginners-guide-on-logistic-regression-in-r/
predicted <- predict(mymodel, type = 'response')
confusion_matrix <- table(mydata$Pass, predicted > 0.5)
rownames(confusion_matrix) <- c('Actual Fail(0)','Actual Pass(1)')
colnames(confusion_matrix) <- c('Predict Fail(0)','Predict Pass(1)')
confusion_matrix <- confusion_matrix[2:1,2:1] # https://stackoverflow.com/questions/32593434/r-change-row-order
confusion_matrix # https://www.edrm.net/glossary/confusion-matrix/
tp <- confusion_matrix[1,1]
fn <- confusion_matrix[1,2]
fp <- confusion_matrix[2,1]
tn <- confusion_matrix[2,2]
p = tp+fn
n = fp+tn
model_recall = tp/p # sensitivity
model_specificity = tn/n
model_precision = tp/(tp+fp)
model_false_alarm = fp/n
print(sprintf("Recall = %.02f, Specificity = %.02f, Precision = %.02f, False alarm = %.02f", model_recall, model_specificity, model_precision, model_false_alarm))

library(caret)
confusionMatrix(as.factor(predicted > 0.5), as.factor(mydata$Pass > 0.5))

# https://www.analyticsvidhya.com/blog/2015/11/beginners-guide-on-logistic-regression-in-r/
library(ROCR)
ROCRpred <- prediction(predicted, mydata$Pass)
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf, colorize = FALSE, text.adj = c(-0.2,1.7))
abline(a=0, b=1)

library(PRROC)
fg <- predicted[mydata$Pass == 1]
bg <- predicted[mydata$Pass == 0]
# ROC Curve    
roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
plot(roc)

# PR Curve
pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
plot(pr)


# Reference: http://www.shizukalab.com/toolkits/plotting-logistic-regression-in-r
quartz(title="Hours of studying vs. Probability of passing the exam")
par(col="black")
plot(mydata$Hours,mydata$Pass,xlab="Hours of studying",ylab="Probability of passing the exam")
par(col="blue")
curve(predict(mylogit,data.frame(Hours=x),type="response"),add=TRUE) 
par(col="black")
points(mydata$Hours,fitted(mylogit),pch=20) 

# http://www.cookbook-r.com/Statistical_analysis/Logistic_regression/
#library(ggplot2)
#ggplot(mydata, aes(x=Hours, y=Pass)) + geom_point() + stat_smooth(method="glm", method.args=list(family="binomial"), se=FALSE)
#par(mar = c(4, 4, 1, 1)) # Reduce some of the margins so that the plot fits better
