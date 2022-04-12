library(moments)
library(fitdistrplus)
library(caret)
library(glmnet)
library(pls)

setwd("./data")

# Data Preprocessing
allData <- read.csv("data.csv")
attach(allData)
tickerFactor <- factor(allData$TICKER)
tickerLevel <- levels(tickerFactor)

## Use for loop to construct subset according to ticker
normalityTestResult <- matrix(0, nrow = 2, ncol = length(tickerLevel))
for (i in 1:length(tickerLevel)){
  subDataFrame <- allData[allData$TICKER == tickerLevel[i],]
  logReturn <- log(subDataFrame$RETN + 1)
  shapiroTest <- shapiro.test(logReturn)
  normalityTestResult[1,i] <- round(shapiroTest$p.value,3)
  if (shapiroTest$p.value < 0.1){
    normalityTestResult[2,i] <- 'Adequate'
  }
  else{
    normalityTestResult[2,i] <- 'Inadequate'
  }
}
colnames(normalityTestResult) <- tickerLevel
rownames(normalityTestResult) <- c('p-Value','Normality')
# normalityTestResult

## OLS Regression
### construct subdataframe of different tickers
for (i in 1:length(tickerLevel)) {
  subDf = allData[allData$TICKER == tickerLevel[i],]
  assign(tickerLevel[i], subDf)
}
### Standardize the covariate
Standardize <- function(x){
  x <- x/sd(x)
  return(x)
}
for (i in 1:length(tickerLevel)) {
  tempDf = get(tickerLevel[i])
  tempDfResponse <- tempDf[,c('TICKER','Month','RETN')]
  tempDfExplanatory <- tempDf[,!colnames(tempDf) %in% c('TICKER','Month','RETN')]
  tempDfExplanatoryStandardized <- apply(tempDfExplanatory, 2, Standardize)
  processedDf <- cbind(tempDfResponse, tempDfExplanatoryStandardized)
  assign(paste('standardized',tickerLevel[i], sep = ""), processedDf)
}

### Construct the regression DataFrame
preRegreDf <- data.frame(data.frame(matrix(0,1,ncol(allData))))
colnames(preRegreDf) <- colnames(standardizedAAPL)
for (i in 1:length(tickerLevel)) {
  dfName <- paste('standardized', tickerLevel[i], sep = "")
  preRegreDf <- rbind(preRegreDf, get(dfName))
}
preRegreDf <- preRegreDf[-1,]
regreDf <- preRegreDf[,-c(1,2)]
reg1 <- lm(RETN~., data = regreDf)
summary(reg1)

response <- allData$RETN
explanatory <- allData[,!colnames(allData) %in% c('TICKER','Month','RETN')]
response <- data.frame(allData$RETN)
colnames(response) <- 'RETN'
explantory1 <- apply(explanatory, 2, Standardize)
regDf <- cbind(response, explantory1)
reg2 <- lm(RETN~., data = regDf)
summary(reg2)

### K-fold Cross Validation
#### Way1 Create Folds
folds <- sample(1:10, dim(regDf), replace = TRUE)
Return.train <- regDf[folds != 10, ]
Return.test <- regDf[folds == 10, ]
fit.lm <- lm(RETN ~ ., data = Return.train)
pred.lm <- predict(fit.lm, Return.test)
testMSE <- mean((pred.lm - Return.test$RETN)^2)
#### Way2 Use Package Caret
set.seed(1)
train.control <- trainControl(method="cv",number=10)
reg_cv <- train(RETN~., data=regDf, method="lm", trControl=train.control)
print(reg_cv)
reg_cv$results
testMSE <- as.numeric(reg_cv$results[2])^2   # The estimated test MSE from CV
testMSE

## Lasso Regression
### Train Lasso Regression on the whole data set
# Create the X matrix for regression (excluding the constant column)
x <- model.matrix(RETN~.,regDf)[,-1]
# Create the Y vector for regression
y <- RETN
set.seed(1)
# Default is 10-fold CV. You can change the setting using argument nfolds. 
# There's no need to specify the vector lambda. The function cv.glmnet will choose a sequence of lambda automatically. But you can specify your own lambda.
# Use alpha=0 for ridge.
cv.out <- cv.glmnet(x,y,nfolds=10,alpha=1)    
plot(cv.out)
cv.out$cvm
min(cv.out$cvm)
bestlam <- cv.out$lambda.min
reg_lasso_best <- glmnet(x,y,alpha=1,lambda=c(bestlam))
coef(reg_lasso_best)
### Train Lasso Regression on the train set and predict it on the test set
folds <- sample(1:10, dim(regDf), replace = TRUE)
Return.train <- regDf[folds != 10, ]
Return.test <- regDf[folds == 10, ]
x.train <- model.matrix(RETN~.,Return.train)[,-1]
y.train <- Return.train$RETN
cv.out <- cv.glmnet(x.train, y.train, nfolds=10, alpha=1)
bestlam <- cv.out$lambda.min
fit.lasso <- glmnet(x.train, y.train, alpha=1,lambda=c(bestlam))
x.test <- model.matrix(RETN~.,Return.test)[,-1]
y.test <- Return.test$RETN
pred.lasso <- predict(fit.lasso, s = bestlam, newx = x.test)
testMSE <- mean((pred.lasso - y.test)^2)
testMSE