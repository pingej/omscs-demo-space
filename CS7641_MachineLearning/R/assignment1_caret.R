install.packages(c("caret", "rpart", "nnet", "kernlab", "gbm", "e1071", "adabag"))

library(caret)

data(iris)

#' take X% of the data randomly to form your training and test set
set.seed(1)
rand.sample <- sample(1:nrow(iris), floor(nrow(iris))*.5)
train.data <- iris[rand.sample,]
test.data <- iris[-rand.sample,]

# setup cross validation under caret::train
# here we do 5-fold cv with 1 repeat
control <- trainControl(method="repeatedcv", number=5, repeats=1, 
                        index=createFolds(train.data$Species))

# check out ?caret::train to see how to control hyperparameters
DT <- train(Species ~., data=train.data, method="rpart", trControl=control)
NNET <- train(Species ~., data=train.data, method="nnet", trControl=control)
SVM <- train(Species ~., data=train.data, method="svmLinear", trControl=control)
KNN <- train(Species ~., data=train.data, method="knn", trControl=control)

iris.model <- list(DT=DT, NNET=NNET, BOOST=BOOST, SVM=SVM, KNN=KNN)

# plot the accuracy of the models based on the training set.
bwplot(resamples(iris.model))

# check against the test data (confusion matrix)
iris.confusionMatrix <- Map(function(model) {
  fit.test <- predict(model, newdata=test.data[,!(names(test.data) %in% c("Species"))])    
  cm <- confusionMatrix(fit.test, test.data[,names(test.data) %in% c("Species")])
  return(as.list(cm))
}, iris.model)

# you can extract the information from confusion matrix as follows:
iris.confusionMatrix$NNET$table


