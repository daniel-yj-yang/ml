# By Daniel Yang Ph.D (daniel.yj.yang@gmail.com)

library(shiny)
library(DT) # https://shiny.rstudio.com/articles/datatables.html
#library(caret)
#library(ROCR) #https://www.analyticsvidhya.com/blog/2015/11/beginners-guide-on-logistic-regression-in-r/
library(PRROC) # https://stats.stackexchange.com/questions/10501/calculating-aupr-in-r

# The data is based on https://www.kaggle.com/rakeshrau/social-network-ads
mydata <- matrix(c(
  19, 0, 
  35, 0, 
  26, 0, 
  27, 0, 
  19, 0, 
  27, 0, 
  27, 0, 
  32, 1, 
  25, 0, 
  35, 0, 
  26, 0, 
  26, 0, 
  20, 0, 
  32, 0, 
  18, 0, 
  29, 0, 
  47, 1, 
  45, 1, 
  46, 1, 
  48, 1, 
  45, 1, 
  47, 1, 
  48, 1, 
  45, 1, 
  46, 1, 
  47, 1, 
  49, 1, 
  47, 1, 
  29, 0, 
  31, 0, 
  31, 0, 
  27, 1, 
  21, 0, 
  28, 0, 
  27, 0, 
  35, 0, 
  33, 0, 
  30, 0, 
  26, 0, 
  27, 0, 
  27, 0, 
  33, 0, 
  35, 0, 
  30, 0, 
  28, 0, 
  23, 0, 
  25, 0, 
  27, 0, 
  30, 1, 
  31, 0, 
  24, 0, 
  18, 0, 
  29, 0, 
  35, 0, 
  27, 0, 
  24, 0, 
  23, 0, 
  28, 0, 
  22, 0, 
  32, 0, 
  27, 0, 
  25, 0, 
  23, 0, 
  32, 1, 
  59, 0, 
  24, 0, 
  24, 0, 
  23, 0, 
  22, 0, 
  31, 0, 
  25, 0, 
  24, 0, 
  20, 0, 
  33, 0, 
  32, 0, 
  34, 1, 
  18, 0, 
  22, 0, 
  28, 0, 
  26, 0, 
  30, 0, 
  39, 0, 
  20, 0, 
  35, 0, 
  30, 0, 
  31, 1, 
  24, 0, 
  28, 0, 
  26, 0, 
  35, 0, 
  22, 0, 
  30, 0, 
  26, 0, 
  29, 0, 
  29, 0, 
  35, 0, 
  35, 0, 
  28, 1, 
  35, 0, 
  28, 0, 
  27, 0, 
  28, 0, 
  32, 0, 
  33, 1, 
  19, 0, 
  21, 0, 
  26, 0, 
  27, 0, 
  26, 0, 
  38, 0, 
  39, 0, 
  37, 0, 
  38, 0, 
  37, 0, 
  42, 0, 
  40, 0, 
  35, 0, 
  36, 0, 
  40, 0, 
  41, 0, 
  36, 0, 
  37, 0, 
  40, 0, 
  35, 0, 
  41, 0, 
  39, 0, 
  42, 0, 
  26, 0, 
  30, 0, 
  26, 0, 
  31, 0, 
  33, 0, 
  30, 0, 
  21, 0, 
  28, 0, 
  23, 0, 
  20, 0, 
  30, 1, 
  28, 0, 
  19, 0, 
  19, 0, 
  18, 0, 
  35, 0, 
  30, 0, 
  34, 0, 
  24, 0, 
  27, 1, 
  41, 0, 
  29, 0, 
  20, 0, 
  26, 0, 
  41, 0, 
  31, 0, 
  36, 0, 
  40, 0, 
  31, 0, 
  46, 0, 
  29, 0, 
  26, 0, 
  32, 1, 
  32, 1, 
  25, 0, 
  37, 0, 
  35, 0, 
  33, 0, 
  18, 0, 
  22, 0, 
  35, 0, 
  29, 1, 
  29, 0, 
  21, 0, 
  34, 0, 
  26, 0, 
  34, 0, 
  34, 0, 
  23, 0, 
  35, 0, 
  25, 0, 
  24, 0, 
  31, 0, 
  26, 0, 
  31, 0, 
  32, 1, 
  33, 0, 
  33, 0, 
  31, 0, 
  20, 0, 
  33, 0, 
  35, 0, 
  28, 0, 
  24, 0, 
  19, 0, 
  29, 0, 
  19, 0, 
  28, 0, 
  34, 0, 
  30, 0, 
  20, 0, 
  26, 0, 
  35, 0, 
  35, 0, 
  49, 0, 
  39, 1, 
  41, 0, 
  58, 1, 
  47, 0, 
  55, 1, 
  52, 0, 
  40, 1, 
  46, 0, 
  48, 1, 
  52, 1, 
  59, 0, 
  35, 0, 
  47, 0, 
  60, 1, 
  49, 0, 
  40, 0, 
  46, 0, 
  59, 1, 
  41, 0, 
  35, 1, 
  37, 1, 
  60, 1, 
  35, 0, 
  37, 0, 
  36, 1, 
  56, 1, 
  40, 0, 
  42, 1, 
  35, 1, 
  39, 0, 
  40, 1, 
  49, 1, 
  38, 0, 
  46, 1, 
  40, 0, 
  37, 0, 
  46, 0, 
  53, 1, 
  42, 1, 
  38, 0, 
  50, 1, 
  56, 1, 
  41, 0, 
  51, 1, 
  35, 0, 
  57, 1, 
  41, 0, 
  35, 1, 
  44, 0, 
  37, 0, 
  48, 1, 
  37, 1, 
  50, 0, 
  52, 1, 
  41, 0, 
  40, 0, 
  58, 1, 
  45, 1, 
  35, 0, 
  36, 1, 
  55, 1, 
  35, 0, 
  48, 1, 
  42, 1, 
  40, 0, 
  37, 0, 
  47, 1, 
  40, 0, 
  43, 0, 
  59, 1, 
  60, 1, 
  39, 1, 
  57, 1, 
  57, 1, 
  38, 0, 
  49, 1, 
  52, 1, 
  50, 1, 
  59, 1, 
  35, 0, 
  37, 1, 
  52, 1, 
  48, 0, 
  37, 1, 
  37, 0, 
  48, 1, 
  41, 0, 
  37, 1, 
  39, 1, 
  49, 1, 
  55, 1, 
  37, 0, 
  35, 0, 
  36, 0, 
  42, 1, 
  43, 1, 
  45, 0, 
  46, 1, 
  58, 1, 
  48, 1, 
  37, 1, 
  37, 1, 
  40, 0, 
  42, 0, 
  51, 0, 
  47, 1, 
  36, 1, 
  38, 0, 
  42, 0, 
  39, 1, 
  38, 0, 
  49, 1, 
  39, 0, 
  39, 1, 
  54, 1, 
  35, 0, 
  45, 1, 
  36, 0, 
  52, 1, 
  53, 1, 
  41, 0, 
  48, 1, 
  48, 1, 
  41, 0, 
  41, 0, 
  42, 0, 
  36, 1, 
  47, 1, 
  38, 0, 
  48, 1, 
  42, 0, 
  40, 0, 
  57, 1, 
  36, 0, 
  58, 1, 
  35, 0, 
  38, 0, 
  39, 1, 
  53, 1, 
  35, 0, 
  38, 0, 
  47, 1, 
  47, 1, 
  41, 0, 
  53, 1, 
  54, 1, 
  39, 0, 
  38, 0, 
  38, 1, 
  37, 0, 
  42, 1, 
  37, 0, 
  36, 1, 
  60, 1, 
  54, 1, 
  41, 0, 
  40, 1, 
  42, 0, 
  43, 1, 
  53, 1, 
  47, 1, 
  42, 0, 
  42, 1, 
  59, 1, 
  58, 1, 
  46, 1, 
  38, 0, 
  54, 1, 
  60, 1, 
  60, 1, 
  39, 0, 
  59, 1, 
  37, 0, 
  46, 1, 
  46, 0, 
  42, 0, 
  41, 1, 
  58, 1, 
  42, 0, 
  48, 1, 
  44, 1, 
  49, 1, 
  57, 1, 
  56, 1, 
  49, 1, 
  39, 0, 
  47, 1, 
  48, 1, 
  48, 1, 
  47, 1, 
  45, 1, 
  60, 1, 
  39, 0, 
  46, 1, 
  51, 1, 
  50, 1, 
  36, 0, 
  49, 1)
  ,ncol=2,byrow=TRUE)
colnames(mydata) <- c("Age","Purchased")
rownames(mydata) <- c(1:400)

raw_data <- as.data.frame(mydata)
raw_data0 <- raw_data

ui <- fluidPage(
  withMathJax(),
  titlePanel("Logitic Regression Demo: Using Age to Explain (the Odds of the Probability of) Purchased"),
  fluidRow(column(12, h3('Double click to edit the table and see instant results'), hr())),
  actionButton("reset_the_table", "Reset the table"),
  sidebarLayout(
    sidebarPanel(
      DT::dataTableOutput("data_table")
    ),
    mainPanel(
      verbatimTextOutput('text_output'),
      plotOutput('main_plot')
    )
  ),
  uiOutput('LR_fundamentals'),
  plotOutput('LR_fundamentals_plot',height = "700px"),
  uiOutput('beta0'),
  uiOutput('beta1'),
  uiOutput('linear_func'),
  uiOutput('logistic_func'),
  uiOutput('odds_of_y'),
  uiOutput('log_odds'),
  uiOutput('odds_ratio'),
  uiOutput('aic'),
  verbatimTextOutput('confusion_matrix'),  # see also: https://yihui.shinyapps.io/DT-rows/
  plotOutput('ROC_curve'),
  plotOutput('PR_curve'),
  fluidRow(column(12, h4('By Daniel Yang, Ph.D. (daniel.yj.yang@gmail.com) @ 2020')))
)

# https://shiny.rstudio.com/gallery/option-groups-for-selectize-input.html
server <- function(input, output, session) {
  
  # https://github.com/daattali/advanced-shiny/tree/master/reactive-trigger
  rv <- reactiveValues(a = 0) # https://stackoverflow.com/questions/33382525/how-to-invalidate-reactive-observer-using-code
  
  my_model <- reactive({
    rv$a # https://stackoverflow.com/questions/33382525/how-to-invalidate-reactive-observer-using-code
    glm(Purchased ~ Age, data = raw_data, family=binomial(link="logit"))
  })
  beta0 <- reactive({
    rv$a
    my_model()$coefficients[1]
  })
  beta1 <- reactive({
    rv$a
    my_model()$coefficients[2]
  })
  predicted <- reactive({
    rv$a
    predict(my_model(), type = 'response') # the same as fitted(my_model())
  })
  Odds <- reactive({
    rv$a
    exp(predict(my_model()))
    # equivalent statement below
    #Prob_purchased=fitted(my_model())
    #DV=Prob_purchased/(1-Prob_purchased)  
    #DV
  })
  
  output$data_table <- renderDT(  # https://rstudio.github.io/DT/shiny.html
    raw_data, selection = 'none', editable = TRUE
  )
  
  raw_data_proxy = dataTableProxy('data_table')
  
  observeEvent(input$data_table_cell_edit, { # https://yihui.shinyapps.io/DT-edit/
    info = input$data_table_cell_edit # https://yihui.shinyapps.io/DT-edit/
    #str(info)
    i = info$row
    j = info$col
    v = info$value
    raw_data[i, j] <<- DT::coerceValue(v, raw_data[i, j])
    replaceData(raw_data_proxy, raw_data, resetPaging = FALSE)  # important
    rv$a <- rv$a + 1 # https://stackoverflow.com/questions/33382525/how-to-invalidate-reactive-observer-using-code
  })
  
  observeEvent(input$reset_the_table, {
    raw_data <<- raw_data0
    replaceData(raw_data_proxy, raw_data, resetPaging = FALSE)  # important
    rv$a <- rv$a + 1 # https://stackoverflow.com/questions/33382525/how-to-invalidate-reactive-observer-using-code
  })
  
  output$main_plot <- renderPlot({
    par(mfrow=c(1,1))
    par(col="black")
    par(pin=c(5,3.5))
    plot(raw_data$Age,raw_data$Purchased,xlab="IV: Age",ylab="Purchased (Y or N) = Odds/(1+Odds)") 
    par(col="blue",lwd=1)
    curve(predict(my_model(),data.frame(Age=x),type="resp"),add=TRUE)  # type = "response" gives the predicted probabilities
    par(col="black")
    points(raw_data$Age,fitted(my_model()),pch=20)
    title(main="y = Prob(Purchased), x = Age")
    # add a horizontal line at p=.5
    abline(h=.5, lty=2)
    abline(v=-beta0()/beta1(), lty=2)
  })
  output$text_output <- renderPrint({
    summary(my_model())
  })
  output$LR_fundamentals <- renderUI({
    withMathJax(sprintf("The fundamental interpretation of a logistic regression is based on the natural exponential function, where DV = the Odds of the probability of Purchased(=1) :$$odds(prob_{Purchased}) = \\frac{prob_{Purchased_Y}}{prob_{Purchased_N}} = e^{\\beta_0+\\beta_1x}$$And the minimal Age (that is, the x) required to make the DV = Odds of P(Purchased) ≥ 1 or P(Purchased) ≥ 0.50, is given by the following:$$x\\geq-\\frac{\\beta_0}{\\beta_1}=%.02f$$", -beta0()/beta1() ))
  })
  output$LR_fundamentals_plot <- renderPlot({
    par(mfrow=c(1,1))
    par(col="black")
    par(pin=c(5,7.5))
    plot(raw_data$Age,Odds(),xlab="IV: Age",ylab="DV: Odds of P(Purchased)")
    par(col="blue",lwd=1)
    curve(exp(predict(my_model(),data.frame(Age=x))),add=TRUE)
    par(col="black")
    points(raw_data$Age,Odds(),pch=20)
    # add a horizontal line at odds=1
    abline(h=1, lty=2)
    abline(v=-beta0()/beta1(), lty=2)
    title(main="Logistic Regression: Fundamental Relationship")
  })
  output$beta0 <- renderUI({
    withMathJax(sprintf("1. Intercept (interpretation: the log odds when x=0):$$\\beta_0 = %.02f$$\nThus, the baseline odds_of_y when x=0:$$odds(y|_{x=0}) = \\frac{p|_{x=0}}{q|_{x=0}} = e^{\\beta_0} = e^{%.02f} = %.05f, where\\,p|_{x=0} = %.05f\\,and\\,q|_{x=0} = %.05f$$",beta0(),beta0(),exp(beta0()),exp(beta0())/(1+exp(beta0())), 1-(exp(beta0())/(1+exp(beta0()))) ))
  })
  output$beta1 <- renderUI({
    withMathJax(sprintf("2. Slope (interpretation: the natural log of the odds ratio):$$\\beta_1 = %.02f$$\nThus, when x increases 1 unit from 0 to 1, the odds_of_y will be multipled by the Odds Ratio (OR):$$odds(y|_{x=1}) = odds(y|_{x=0})*OR = e^{\\beta_0+\\beta_1x=\\beta_0}e^{\\beta_1} = e^{%.02f}e^{%.02f} = %.05f*%.02f = %.05f$$\nSimilarly, when x increases 1 unit from 1 to 2, the odds_of_y will also be multipled by the Odds Ratio (OR):$$odds(y|_{x=2}) = odds(y|_{x=1})*OR = e^{\\beta_0+\\beta_1x=\\beta_0+\\beta_1}e^{\\beta_1} = e^{%.02f}e^{%.02f} = %.05f*%.02f = %.05f$$\nIn general, when x increases 1 unit from x to x+1, the odds_of_y will be multipled by the Odds Ratio (OR):$$odds(y|_{x+1}) = odds(y|_{x})*OR = e^{\\beta_0+\\beta_1x}e^{\\beta_1} = e^{\\beta_0+\\beta_1x}e^{%.02f} = e^{\\beta_0+\\beta_1x}*%.02f$$",beta1(),beta0(),beta1(),exp(beta0()),exp(beta1()),exp(beta0())*exp(beta1()), beta0()+beta1(), beta1(), exp(beta0()+beta1()), exp(beta1()), exp(beta0()+beta1())*exp(beta1()), beta1(), exp(beta1()) ))
  })
  output$linear_func <- renderUI({
    withMathJax(sprintf("3. Linear regression function (where x = Age), x is linearly related to the log odds (note: log odds is simply taking the natural log of the odds):$$z = log\\,odds = ln(\\frac{p}{q}\\!) = \\beta_0 + \\beta_1x$$"))
  })
  output$logistic_func <- renderUI({
    withMathJax(sprintf("4. Logistic/sigmoid function (where y = probability of Purchased, range: 0-1; if odds(y) is very big, y is approaching 1; if odds(y) is very small, y is approaching 0):$$y = prob_{Purchased_Y} = p = (1-prob_{Purchased_N}) = (1-q) = \\frac{1}{1+e^{-z}}\\! = \\frac{odds(y)}{1+odds(y)}\\! = \\frac{e^z}{1+e^z}\\!$$"))
  })
  output$odds_of_y <- renderUI({
    withMathJax(sprintf("5. Odds_of_y (input (y) range: 0-1; output range: 0-Inf, since exponential func is always positive and up to infinity) (this relationship is important as it provides the link between log odds and odds_of_y):$$odds = \\frac{p}{q}\\! = odds(y) = \\frac{y}{1-y}\\! = \\frac{prob_{Purchased_Y}}{prob_{Purchased_N}}\\! = e^z = e^{\\beta_0 + \\beta_1x}$$"))
  })
  output$log_odds <- renderUI({
    withMathJax(sprintf("6. Log odds (logit func; input (y) range: 0-1; output range: -Inf ~ +Inf):$$log\\,odds = ln(odds(y)) = ln(\\frac{y}{1-y}\\!) = ln(\\frac{prob_{Purchased_Y}}{prob_{Purchased_N}}\\!) = ln(\\frac{p}{q}\\!) = ln(e^z) = z = \\beta_0 + \\beta_1x$$"))
  })
  output$odds_ratio <- renderUI({
    withMathJax(sprintf("7. Odds Ratio (OR; the effect size measure; for every 1-unit increase in x, the odds_of_y, (p/q), not just p, is multiplied by this factor) (this is true, even with multiple x's, assuming no interaction among x's). \nIn this demo, for each 1 additional unit of age, the odds of the prob. of Purchased will be OR = %.02f times as big:$$OR = \\frac{odds(y|_{x+1})}{odds(y|_x)}\\! = \\frac{e^{\\beta_0 + \\beta_1(x+1)}}{e^{\\beta_0 + \\beta_1x}}\\! = e^{\\beta_1} = e^{%.02f} = %.02f$$",exp(beta1()),beta1(),exp(beta1())))
  })
  output$aic <- renderUI({
    withMathJax(sprintf("8. Evaluate Performance: AIC (the lower the better):$$AIC = %.02f$$",my_model()$aic))
  })
  output$confusion_matrix <- renderPrint({
    cat(sprintf("9. Evaluate Performance: Confusion matrix:\n\n"))
    confusion_matrix <- table(raw_data$Purchased, predicted() > 0.5)
    rownames(confusion_matrix) <- c('Actual Purchased_N(0)','Actual Purchased_Y(1)')
    colnames(confusion_matrix) <- c('Predict Purchased_N(0)','Predict Purchased_Y(1)')
    confusion_matrix <- confusion_matrix[2:1,2:1] # https://stackoverflow.com/questions/32593434/r-change-row-order
    
    tp <- confusion_matrix[1,1]
    fn <- confusion_matrix[1,2]
    fp <- confusion_matrix[2,1]
    tn <- confusion_matrix[2,2]
    p = tp+fn
    n = fp+tn
    
    cat(sprintf("Actual\\Predict | Predict_YES\t | Predict_NO    | Sum\n"))
    cat(sprintf("---------------|-----------------|---------------|--------------------\n"))
    cat(sprintf("Actual_YES     | TP (n=%d)        | FN (n=%d)      | P (Positive) (n=%d)\n",tp,fn,p))
    cat(sprintf("Actual_NO      | FP (n=%d)        | TN (n=%d)      | N (Negative) (n=%d)\n",fp,tn,n))
    cat(sprintf("---------------|-----------------|---------------|--------------------\n"))
    cat(sprintf("Sum            | TP+FP (n=%d)    | FN+TN (n=%d)  | Total (n=%d)\n\n",tp+fp,fn+tn,p+n))
    
    
    model_recall = tp/p # sensitivity
    model_specificity = tn/n
    model_precision = tp/(tp+fp)
    model_false_positive_rate = fp/n
    
    cat(sprintf("> X-axis in ROC curve = False Positive Rate = (1-Specificity) = FP/N = %.02f\n", model_false_positive_rate))
    cat(sprintf("> Y-axis in ROC curve = True Positive Rate = Recall (to minimize Type II error) = Sensitivity = Power = TP/P = %.02f\n\n", model_recall))
    cat(sprintf("> Specificity = TN/N = %.02f\n", model_specificity))
    cat(sprintf("> Precision (to minimize Type I error) = Positive Predictive Value = TP/(TP+FP) = %.02f\n\n", model_precision))
    
    cat(sprintf("> Positive Likelihood Ratio (LR+) = (1-β)/α = %.02f\n", model_recall/model_false_positive_rate ))
    cat(sprintf("> Negative Likelihood Ratio (LR-) = β/(1-α) = %.02f\n", (1-model_recall)/(1-model_false_positive_rate) ))
    cat(sprintf("> Diagnostic Odds Ratio (DOR) = (LR+)/(LR-) = %.02f\n", (model_recall/model_false_positive_rate) / ((1-model_recall)/(1-model_false_positive_rate)) ))
    
    #confusionMatrix(as.factor(predicted() > 0.5), as.factor(raw_data$Purchased > 0.5))
  })
  output$ROC_curve <- renderPlot({
    par(mfrow=c(1,1))
    par(pin=c(6,4))
    
    # https://stats.stackexchange.com/questions/10501/calculating-aupr-in-r
    fg <- predicted()[raw_data$Purchased == 1]
    bg <- predicted()[raw_data$Purchased == 0]
    # ROC Curve    
    roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
    plot(roc)
    
    # another workable version
    # https://www.analyticsvidhya.com/blog/2015/11/beginners-guide-on-logistic-regression-in-r/
    #ROCRpred <- prediction(predicted(), raw_data$Purchased)
    #ROCRperf <- performance(ROCRpred, 'tpr', 'fpr')
    #plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2,1.7))
    #title(main=sprintf("10. Evaluate Performance: ROC curve (AUC = %.03f)", performance(ROCRpred, measure="auc")@y.values))
    #abline(a=0, b=1)
  })
  output$PR_curve <- renderPlot({
    par(mfrow=c(1,1))
    par(pin=c(6,4))
    # https://stats.stackexchange.com/questions/10501/calculating-aupr-in-r
    fg <- predicted()[raw_data$Purchased == 1]
    bg <- predicted()[raw_data$Purchased == 0]
    # PR Curve
    pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
    plot(pr)
  })
}

shinyApp(ui = ui, server = server)
