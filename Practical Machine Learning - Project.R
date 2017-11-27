
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(corrplot)

url1 <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(url1, destfile="pml-training.csv")
url2 <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url2, destfile="pml-testing.csv")
trainFile <- read.csv("pml-training.csv", header=TRUE)
testFile<- read.csv("pml-testing.csv", header=TRUE)

sum(complete.cases(trainRaw))

trainRaw <- trainRaw[, colSums(is.na(trainRaw)) == 0] 
testRaw <- testRaw[, colSums(is.na(testRaw)) == 0]

classe <- trainRaw$classe
trainRemove <- grepl("^X|timestamp|window", names(trainRaw))
trainRaw <- trainRaw[, !trainRemove]
trainCleaned <- trainRaw[, sapply(trainRaw, is.numeric)]
trainCleaned$classe <- classe

testRemove <- grepl("^X|timestamp|window", names(testRaw))
testRaw <- testRaw[, !testRemove]
testCleaned <- testRaw[, sapply(testRaw, is.numeric)]

dim(trainCleaned)

dim(testCleaned)

set.seed(12345) 
inTrain <- createDataPartition(trainCleaned$classe, p=0.70, list=F)
trainData <- trainCleaned[inTrain, ]
testData <- trainCleaned[-inTrain, ]

predictRf <- predict(modelRf, testData)
confusionMatrix(testData$classe, predictRf)

accuracy <- postResample(predictRf, testData$classe)
accuracy

oose <- 1 - as.numeric(confusionMatrix(testData$classe, predictRf)$overall[1])
oose

result <- predict(modelRf, testCleaned[, -length(names(testCleaned))])
result

corrPlot <- cor(trainData[, -length(names(trainData))])
corrplot(corrPlot, method="color")

treeModel <- rpart(classe ~ ., data=trainData, method="class")
prp(treeModel) 