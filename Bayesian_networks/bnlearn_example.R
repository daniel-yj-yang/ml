# https://www.bnlearn.com/about/teaching/slides-bnshort.pdf

library(bnlearn)

# Age (A): young for individuals below 30 years old, adult for individuals between 30 and 60 years old, and old for people older than 60.
# Sex (S): male or female.
# Education (E): up to high school or university degree.
# Occupation (O): employee or self-employed.
# Residence (R): the size of the city the individual lives in, recorded as either small or big.
# Travel (T): the means of transport favoured by the individual, recorded either as car, train or other

# Setting individual arcs.
survey.dag = empty.graph(nodes = c("A", "S", "E", "O", "R", "T"))
survey.dag = set.arc(survey.dag, from = "A", to = "E")
survey.dag = set.arc(survey.dag, from = "S", to = "E")
survey.dag = set.arc(survey.dag, from = "E", to = "O")
survey.dag = set.arc(survey.dag, from = "E", to = "R")
survey.dag = set.arc(survey.dag, from = "O", to = "T")
survey.dag = set.arc(survey.dag, from = "R", to = "T")

# Setting the whole arc set at once
arc.set = matrix(c("A", "E",
                   "S", "E",
                   "E", "O",
                   "E", "R",
                   "O", "T",
                   "R", "T"),
                 byrow = TRUE, ncol = 2,
                 dimnames = list(NULL, c("from", "to")))
arcs(survey.dag) = arc.set

# Using the adjacency matrix representation of the arc set.
amat(survey.dag) =
  matrix(c(0L, 0L, 1L, 0L, 0L, 0L,
           0L, 0L, 1L, 0L, 0L, 0L,
           0L, 0L, 0L, 1L, 1L, 0L,
           0L, 0L, 0L, 0L, 0L, 1L,
           0L, 0L, 0L, 0L, 0L, 1L,
           0L, 0L, 0L, 0L, 0L, 0L),
         byrow = TRUE, nrow = 6, ncol = 6,
         dimnames = list(nodes(survey.dag), nodes(survey.dag)))

# Using the formula representation for the Bayesian network.
survey.dag = model2network("[A][S][E|A:S][O|E][R|E][T|O:R]")

# Finding the skeleton (the underlying undirected graph).
skeleton(survey.dag)

# Finding the moral graph.
moral(survey.dag)

# Plotting Graphs
hlight = list(nodes = c("E", "O"),
              arcs = c("E", "O"),
              col = "grey",
              textCol = "grey")
pp = graphviz.plot(survey.dag,
                   highlight = NULL) #hlight)

# Create a Discrete BN
A.lv = c("young", "adult", "old")
S.lv = c("M", "F")
E.lv = c("high", "uni")
O.lv = c("emp", "self")
R.lv = c("small", "big")
T.lv = c("car", "train", "other")
A.prob = array(c(0.30, 0.50, 0.20), dim = 3, dimnames = list(A = A.lv))
S.prob = array(c(0.60, 0.40), dim = 2, dimnames = list(S = S.lv))
O.prob = array(c(0.96, 0.04, 0.92, 0.08), dim = c(2, 2),
               dimnames = list(O = O.lv, E = E.lv))
R.prob = array(c(0.25, 0.75, 0.20, 0.80), dim = c(2, 2),
               dimnames = list(R = R.lv, E = E.lv))
E.prob = array(c(0.75, 0.25, 0.72, 0.28, 0.88, 0.12, 0.64,
                 0.36, 0.70, 0.30, 0.90, 0.10), dim = c(2, 3, 2),
               dimnames = list(E = E.lv, A = A.lv, S = S.lv))
T.prob = array(c(0.48, 0.42, 0.10, 0.56, 0.36, 0.08, 0.58,
                 0.24, 0.18, 0.70, 0.21, 0.09), dim = c(3, 2, 2),
               dimnames = list(T = T.lv, O = O.lv, R = R.lv))
cpt = list(A = A.prob, S = S.prob, E = E.prob, O = O.prob,
           R = R.prob, T = T.prob)
bn = custom.fit(survey.dag, cpt)

# Conditional Probability Tables
bn$T