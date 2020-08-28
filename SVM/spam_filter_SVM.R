rm(list=ls())
cat("\014")

# https://www.r-bloggers.com/support-vector-machines-with-the-mlr-package/

library(mlr)

library(tidyverse)

data(spam, package = "kernlab")

spamTib <- as_tibble(spam)

spamTib

spamTask <- makeClassifTask(data = spamTib, target = "type")

svm <- makeLearner("classif.svm")

getParamSet("classif.svm")

getParamSet("classif.svm")$pars$kernel$values

kernels <- c("polynomial", "radial", "sigmoid")

svmParamSpace <- makeParamSet(
  makeDiscreteParam("kernel", values = kernels),
  makeIntegerParam("degree", lower = 1, upper = 3),
  makeNumericParam("cost", lower = 0.1, upper = 10),
  makeNumericParam("gamma", lower = 0.1, 10))

randSearch <- makeTuneControlRandom(maxit = 100)

cvForTuning <- makeResampleDesc("Holdout", split = 2/3)

library(parallelMap)
library(parallel)

parallelStartSocket(cpus = detectCores())

tunedSvmPars <- tuneParams("classif.svm", task = spamTask,
                           resampling = cvForTuning,
                           par.set = svmParamSpace,
                           control = randSearch)

parallelStop()

tunedSvmPars

tunedSvmPars$x

tunedSvm <- setHyperPars(makeLearner("classif.svm"),
                         par.vals = tunedSvmPars$x)

tunedSvmModel <- train(tunedSvm, spamTask)

outer <- makeResampleDesc("CV", iters = 3)

svmWrapper <- makeTuneWrapper("classif.svm", resampling = cvForTuning,
                              par.set = svmParamSpace,
                              control = randSearch)

parallelStartSocket(cpus = detectCores())

cvWithTuning <- resample(svmWrapper, spamTask, resampling = outer)

parallelStop()