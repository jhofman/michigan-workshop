library(tidyverse)
library(broom)
library(tm)
library(Matrix)
library(glmnet)
library(ROCR)

# read business and world articles into one data frame
business <- read_tsv('business.tsv')
world <- read_tsv('world.tsv')
articles <- rbind(business, world)

head(articles)

# create a Corpus from the article snippets
corpus <- Corpus(VectorSource(articles$snippet))

# create a DocumentTermMatrix from the snippet Corpus
# remove punctuation and numbers
dtm <- DocumentTermMatrix(corpus, list(weight=weightBin,
                                       stopwords=T,
                                       removePunctuation=T,
                                       removeNumbers=T))
				       
# convert the DocumentTermMatrix to a sparseMatrix
X <- sparseMatrix(i=dtm$i, j=dtm$j, x=dtm$v, dims=c(dtm$nrow, dtm$ncol), dimnames=dtm$dimnames)

### NOTE: there was no fixed random seed, so results will vary due to this ###
# create a train / test split
set.seed(42)
ndx <- sample(nrow(X), floor(nrow(X) * 0.8))

# cross-validate logistic regression with cv.glmnet (family="binomial"), measuring misclassification error
cv <- cv.glmnet(X[ndx,], articles[ndx, ]$section, family="binomial", type.measure='class')

# plot the cross-validation curve
plot(cv)

# evaluate performance for the best-fit model
test_set <- data.frame(actual = articles[-ndx, ]$section,
                       predicted = as.character(predict(cv, X[-ndx,], s = "lambda.min", type = "class")),
                       prob = as.numeric(predict(cv, X[-ndx,], s = "lambda.min")))

# accuracy
summarize(test_set, mean(actual == predicted))

# confusion matrix
table(test_set$actual, test_set$predicted)

# ROC curve and AUC
pred <- prediction(test_set$prob, test_set$actual)
perf <- performance(pred, measure='tpr', x.measure='fpr')
plot(perf)
performance(pred, 'auc')

# look at highly-weighted words
tidy(coef(cv, s="lambda.min")) %>% arrange(value) %>% head
tidy(coef(cv, s="lambda.min")) %>% arrange(desc(value)) %>% head

