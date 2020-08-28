options(width = 200) # without this, shiny's htmlOutput() wraps the text around 80 characters

require("arules")
require("arulesViz")
require("shiny")
require("magrittr")
require("stringr")

data("Groceries")
data("Adult")

ui <- fluidPage(
  
  titlePanel("Association Rules"),
  tags$div(HTML("<h5>This demo uses the transactional datasets from the <a href='https://cran.r-project.org/web/packages/arules/index.html'>arules</a> package.</h5>")),
  hr(),
  sidebarLayout(
    position = "left",
    
    sidebarPanel(
      width = 2,
      HTML("<h4>Parameters for apriori:</h4>"),
      selectInput(inputId = "apriori_dataset",
                  label = "1. Data for apriori:",
                  choices = c("Groceries"),
                  selected = "Groceries",
                  multiple = FALSE,
                  selectize = TRUE),
      sliderInput(inputId = "apriori_sigma",
                  label = "2. Minimum support:",
                  min = 0.006,
                  max = 0.012,
                  value = 0.012,
                  step = 0.001),
      sliderInput(inputId = "apriori_gamma",
                  label = "3. Minimum confidence:",
                  min = 0.45,
                  max = 0.55,
                  value = 0.50,
                  step = 0.01),
      sliderInput(inputId = "apriori_n_rules_display",
                  label = "4. # rules to display:",
                  min = 1,
                  max = 20,
                  value = 10,
                  step = 1),
      HTML("<hr style='height: 1px; background: black;'>"),
      HTML("<h4>Parameters for Eclat:</h4>"),
      selectInput(inputId = "Eclat_dataset",
                  label = "1. Data for Eclat:",
                  choices = c("Adult"),
                  selected = "Adult",
                  multiple = FALSE,
                  selectize = TRUE),
      sliderInput(inputId = "Eclat_support",
                  label = "2. Minimum support:",
                  min = 0.3,
                  max = 0.7,
                  value = 0.5,
                  step = 0.05),
      sliderInput(inputId = "Eclat_n_rules_display",
                  label = "3. # rules to display:",
                  min = 1,
                  max = 20,
                  value = 10,
                  step = 1),
    ),
    
    mainPanel(
      width=10,
      tabsetPanel(type = "tabs",
                  tabPanel( title = "Output (apriori)",
                            fluidRow(
                              column(10, htmlOutput( outputId = "output_apriori_text", width="1000px")),
                              column(10, plotOutput( outputId = 'output_apriori_plot', width="640px", height = "480px")) 
                            )
                  ),
                  tabPanel( title = "Output (Eclat)",
                            fluidRow(
                              column(10, htmlOutput( outputId = "output_Eclat_text", width="1000px")),
                              column(10, plotOutput( outputId = 'output_Eclat_plot', width="640px", height = "480px")) 
                            )
                  ),
                  tabPanel( title = "About",
                            fluidRow(
                              column(10, htmlOutput( outputId = "about" ))
                            )
                  )
                  
      )#,  style='width: 1000px; height: 1000px'
    )
  )
)

server <- function(input, output, session) {
  
  capture.output()
  
  rules_from_apriori <- reactive({
    arules::apriori(Groceries, parameter = list(support = input$apriori_sigma, confidence = input$apriori_gamma, maxtime = 0 )) %>%
      arules::sort(by = "lift")
  })
  
  rules_from_Eclat <- reactive({
    arules::eclat(Adult, parameter = list(support = input$Eclat_support)) %>%
      ruleInduction() %>%
      arules::sort(by = "lift")
  })
  
  output$output_apriori_text <- renderText({
    rules <- rules_from_apriori()
    updateSliderInput( session = session, inputId = "apriori_n_rules_display", max = length(rules) )
    output_apriori <- capture.output(inspect(head(rules, input$apriori_n_rules_display))) %>%
      str_flatten(collapse = '<br/>') %>%
      str_replace_all(' ','&nbsp;')
    HTML(str_c('<div><br/>',
               '<h4 style="white-space:nowrap;">This [', input$apriori_dataset, '] transactional dataset has ', nrow(Groceries), ' transactions (rows) and ', ncol(Groceries), ' items (cols).</h4><hr>',
               '<h4>The analysis based on the <b><a href="https://www.rdocumentation.org/packages/arules/versions/1.6-6/topics/apriori">apriori</a></b> algorithm generates ', length(rules_from_apriori()), ' rule(s), sorted by lift:</h4>',
               '<table style="border: 0.5px solid black; background-color: rgb(255, 255, 255); font-family: \'Courier New\', Courier, monospace; font-size: 14px;">',
               '<tr><td style="white-space:nowrap; width:800px; padding: 15px;">', output_apriori, '</td></tr>',
               '</table>',
               '</div>')
    )
  })
  
  output$output_apriori_plot <- renderPlot({
    plot(rules_from_apriori())
  })
  
  output$output_Eclat_text <- renderText({
    rules <- rules_from_Eclat()
    updateSliderInput( session = session, inputId = "Eclat_n_rules_display", max = length(rules) )
    output_Eclat <- capture.output(inspect(head(rules, input$Eclat_n_rules_display))) %>%
      str_flatten(collapse = '<br/>') %>%
      str_replace_all(' ','&nbsp;')
    HTML(str_c('<div><br/>',
               '<h4 style="white-space:nowrap;">This [', input$Eclat_dataset, '] transactional dataset has ', nrow(Adult), ' transactions (rows) and ', ncol(Adult), ' items (cols).</h4><hr>',
               '<h4>The analysis based on the <b><a href="https://www.rdocumentation.org/packages/arules/versions/1.6-6/topics/eclat">Eclat</a></b> algorithm generates ', length(rules_from_Eclat()), ' rule(s), sorted by lift:</h4>',
               '<table style="border: 0.5px solid black; background-color: rgb(255, 255, 255); font-family: \'Courier New\', Courier, monospace; font-size: 14px;">',
               '<tr><td style="white-space:nowrap; width:800px; padding: 15px;">', output_Eclat, '</td></tr>',
               '</table>',
               '</div>')
    )
  })
  
  output$output_Eclat_plot <- renderPlot({
    plot(rules_from_Eclat())
  })
  
  output$about <- renderText({
    HTML('<br/>By Daniel Yang, Ph.D. (E-mail: <a href="mailto:daniel.yj.yang@gmail.com">daniel.yj.yang@gmail.com</a>)')
  })
}

shinyApp(ui = ui, server = server)