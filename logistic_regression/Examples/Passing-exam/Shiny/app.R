library(shiny)
library(DT) # https://shiny.rstudio.com/articles/datatables.html
#library(caret)
#library(ROCR) #https://www.analyticsvidhya.com/blog/2015/11/beginners-guide-on-logistic-regression-in-r/
library(PRROC) # https://stats.stackexchange.com/questions/10501/calculating-aupr-in-r

mydata <- matrix(c(
  0.5,0,
  0.75,0,
  1,0,
  1.25,0,
  1.5,0,
  1.75,0,
  1.75,1,
  2,0,
  2.25,1,
  2.5,0,
  2.75,1,
  3,0,
  3.25,1,
  3.5,0,
  4,1,
  4.25,1,
  4.5,1,
  4.75,1,
  5,1,
  5.5,1)
  ,ncol=2,byrow=TRUE)
colnames(mydata) <- c("Hours","Pass")
rownames(mydata) <- c(1:20)
raw_data <- as.data.frame(mydata)
raw_data0 <- raw_data

ui <- fluidPage(
  withMathJax(),
  titlePanel("Logitic Regression Example: Using Hours of Studying to Predict the Odds of the Probability of Passing the Exam"),
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
  plotOutput('PR_curve')
)

# https://shiny.rstudio.com/gallery/option-groups-for-selectize-input.html
server <- function(input, output, session) {

  # https://github.com/daattali/advanced-shiny/tree/master/reactive-trigger
  rv <- reactiveValues(a = 0) # https://stackoverflow.com/questions/33382525/how-to-invalidate-reactive-observer-using-code
  
  my_model <- reactive({
    rv$a # https://stackoverflow.com/questions/33382525/how-to-invalidate-reactive-observer-using-code
    glm(Pass ~ Hours, data = raw_data, family=binomial(link="logit"))
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
    #Prob_pass=fitted(my_model())
    #DV=Prob_pass/(1-Prob_pass)  
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
    plot(raw_data$Hours,raw_data$Pass,xlab="IV: Hours of studying",ylab="Prob(passing the exam)=Odds/(1+Odds)") 
    par(col="blue",lwd=1)
    curve(predict(my_model(),data.frame(Hours=x),type="resp"),add=TRUE)  # type = "response" gives the predicted probabilities
    par(col="black")
    points(raw_data$Hours,fitted(my_model()),pch=20)
    title(main="y = Prob(passing the exam), x = Hours of studying")
    # add a horizontal line at p=.5
    abline(h=.5, lty=2)
    abline(v=-beta0()/beta1(), lty=2)
  })
  output$text_output <- renderPrint({
    summary(my_model())
  })
  output$LR_fundamentals <- renderUI({
    withMathJax(sprintf("The fundamental interpretation of a logistic regression is based on the natural exponential function, where DV = the Odds of the probability of passing(=1) the exam:$$odds(prob_{pass}) = \\frac{prob_{pass}}{prob_{fail}} = e^{\\beta_0+\\beta_1x}$$And the minimal hours of studying (that is, the x) required to make the DV = Odds of P(passing) ≥ 1, is given by the following:$$x\\geq-\\frac{\\beta_0}{\\beta_1}=%.02f$$", -beta0()/beta1() ))
  })
  output$LR_fundamentals_plot <- renderPlot({
    par(mfrow=c(1,1))
    par(col="black")
    par(pin=c(5,7.5))
    plot(raw_data$Hours,Odds(),xlab="IV: Hours of studying",ylab="DV: Odds of P(passing)")
    par(col="blue",lwd=1)
    curve(exp(predict(my_model(),data.frame(Hours=x))),add=TRUE)
    par(col="black")
    points(raw_data$Hours,Odds(),pch=20)
    # add a horizontal line at odds=1
    abline(h=1, lty=2)
    abline(v=-beta0()/beta1(), lty=2)
    title(main="Logistic Regression: Fundamental Relationship")
  })
  output$beta0 <- renderUI({
    withMathJax(sprintf("1. Intercept (interpretation: the log odds when x=0):$$\\beta_0 = %.02f$$\nThus, the baseline odds_of_y when x=0:$$odds(y|_{x=0}) = \\frac{p|_{x=0}}{q|_{x=0}} = e^{\\beta_0} = e^{%.02f} = %.02f, where\\,p|_{x=0} = %.02f\\,and\\,q|_{x=0} = %.02f$$",beta0(),beta0(),exp(beta0()),exp(beta0())/(1+exp(beta0())), 1-(exp(beta0())/(1+exp(beta0()))) ))
  })
  output$beta1 <- renderUI({
    withMathJax(sprintf("2. Slope (interpretation: the natural log of the odds ratio):$$\\beta_1 = %.02f$$\nThus, when x increases 1 unit from 0 to 1, the odds_of_y will be multipled by the Odds Ratio (OR):$$odds(y|_{x=1}) = odds(y|_{x=0})*OR = e^{\\beta_0+\\beta_1x=\\beta_0}e^{\\beta_1} = e^{%.02f}e^{%.02f} = %.02f*%.02f = %.02f$$\nSimilarly, when x increases 1 unit from 1 to 2, the odds_of_y will also be multipled by the Odds Ratio (OR):$$odds(y|_{x=2}) = odds(y|_{x=1})*OR = e^{\\beta_0+\\beta_1x=\\beta_0+\\beta_1}e^{\\beta_1} = e^{%.02f}e^{%.02f} = %.02f*%.02f = %.02f$$\nIn general, when x increases 1 unit from x to x+1, the odds_of_y will be multipled by the Odds Ratio (OR):$$odds(y|_{x+1}) = odds(y|_{x})*OR = e^{\\beta_0+\\beta_1x}e^{\\beta_1} = e^{\\beta_0+\\beta_1x}e^{%.02f} = e^{\\beta_0+\\beta_1x}*%.02f$$",beta1(),beta0(),beta1(),exp(beta0()),exp(beta1()),exp(beta0())*exp(beta1()), beta0()+beta1(), beta1(), exp(beta0()+beta1()), exp(beta1()), exp(beta0()+beta1())*exp(beta1()), beta1(), exp(beta1()) ))
  })
  output$linear_func <- renderUI({
    withMathJax(sprintf("3. Linear regression function (where x = hours), x is linearly related to the log odds (note: log odds is simply taking the natural log of the odds):$$z = log\\,odds = ln(\\frac{p}{q}\\!) = \\beta_0 + \\beta_1x$$"))
  })
  output$logistic_func <- renderUI({
    withMathJax(sprintf("4. Logistic/sigmoid function (where y = probability of passing, range: 0-1; if odds(y) is very big, y is approaching 1; if odds(y) is very small, y is approaching 0):$$y = prob_{pass} = p = (1-prob_{fail}) = (1-q) = \\frac{1}{1+e^{-z}}\\! = \\frac{odds(y)}{1+odds(y)}\\! = \\frac{e^z}{1+e^z}\\!$$"))
  })
  output$odds_of_y <- renderUI({
    withMathJax(sprintf("5. Odds_of_y (input (y) range: 0-1; output range: 0-Inf, since exponential func is always positive and up to infinity) (this relationship is important as it provides the link between log odds and odds_of_y):$$odds = \\frac{p}{q}\\! = odds(y) = \\frac{y}{1-y}\\! = \\frac{prob_{pass}}{prob_{fail}}\\! = e^z = e^{\\beta_0 + \\beta_1x}$$"))
  })
  output$log_odds <- renderUI({
    withMathJax(sprintf("6. Log odds (logit func; input (y) range: 0-1; output range: -Inf ~ +Inf):$$log\\,odds = ln(odds(y)) = ln(\\frac{y}{1-y}\\!) = ln(\\frac{prob_{pass}}{prob_{fail}}\\!) = ln(\\frac{p}{q}\\!) = ln(e^z) = z = \\beta_0 + \\beta_1x$$"))
  })
  output$odds_ratio <- renderUI({
    withMathJax(sprintf("7. Odds Ratio (OR; the effect size measure; for every 1-unit increase in x, the odds_of_y, (p/q), not just p, is multiplied by this factor) (this is true, even with multiple x's, assuming no interaction among x's). \nIn this example, for each 1 additional hr of study, the odds of the prob. of passing the exam will be OR = %.02f times as big:$$OR = \\frac{odds(y|_{x+1})}{odds(y|_x)}\\! = \\frac{e^{\\beta_0 + \\beta_1(x+1)}}{e^{\\beta_0 + \\beta_1x}}\\! = e^{\\beta_1} = e^{%.02f} = %.02f$$",exp(beta1()),beta1(),exp(beta1())))
  })
  output$aic <- renderUI({
    withMathJax(sprintf("8. Evaluate Performance: AIC (the lower the better):$$AIC = %.02f$$",my_model()$aic))
  })
  output$confusion_matrix <- renderPrint({
    cat(sprintf("9. Evaluate Performance: Confusion matrix:\n\n"))
    confusion_matrix <- table(raw_data$Pass, predicted() > 0.5)
    rownames(confusion_matrix) <- c('Actual Fail(0)','Actual Pass(1)')
    colnames(confusion_matrix) <- c('Predict Fail(0)','Predict Pass(1)')
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

    #confusionMatrix(as.factor(predicted() > 0.5), as.factor(raw_data$Pass > 0.5))
  })
  output$ROC_curve <- renderPlot({
    par(mfrow=c(1,1))
    par(pin=c(6,4))

    # https://stats.stackexchange.com/questions/10501/calculating-aupr-in-r
    fg <- predicted()[raw_data$Pass == 1]
    bg <- predicted()[raw_data$Pass == 0]
    # ROC Curve    
    roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
    plot(roc)
    
    # another workable version
    # https://www.analyticsvidhya.com/blog/2015/11/beginners-guide-on-logistic-regression-in-r/
    #ROCRpred <- prediction(predicted(), raw_data$Pass)
    #ROCRperf <- performance(ROCRpred, 'tpr', 'fpr')
    #plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2,1.7))
    #title(main=sprintf("10. Evaluate Performance: ROC curve (AUC = %.03f)", performance(ROCRpred, measure="auc")@y.values))
    #abline(a=0, b=1)
  })
  output$PR_curve <- renderPlot({
    par(mfrow=c(1,1))
    par(pin=c(6,4))
    # https://stats.stackexchange.com/questions/10501/calculating-aupr-in-r
    fg <- predicted()[raw_data$Pass == 1]
    bg <- predicted()[raw_data$Pass == 0]
    # PR Curve
    pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
    plot(pr)
  })
}

shinyApp(ui = ui, server = server)
