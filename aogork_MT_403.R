# Alessa Ogork
# G01191082
# Final
# CDS 403-001


# Set directory and load packages
setwd("C:/Users/aless/OneDrive/CDS_403")
library("tidyverse")
library("dplyr")
library("ggplot2")
library(class)
library(caret)

# ABOUT THE DATA SET: letter-recognition
# The objective is to identify each of a large number of black-and-white rectangular pixel displays as one of the 
# 26 capital letters in the English alphabet. The character images were based on 20 different fonts and each letter
# within these 20 fonts was randomly distorted to produce a file of 20,000 unique stimuli. Each stimulus was 
# converted into 16 primitive numerical attributes (statistical moments and edge counts) which were then scaled to 
# fit into a range of integer values from 0 through 15. 

# Attribites (17 in total)
# 1. lettr - capital letter (26 values from A to Z)
# 2. xbox - horizontal position of box (integer)
# 3. ybox - vertical position of box (integer)
# 4. width - width of box (integer)
# 5. height - height of box (integer)
# 6. onpix - total # on pixels (integer)
# 7. xbar - mean x of on pixels in box (integer)
# 8. ybar - mean y of on pixels in box (integer)
# 9. x2bar - mean x variance (integer)
# 10. y2bar - mean y variance (integer)
# 11. xybar - mean x y correlation (integer)
# 12. x2ybr - mean of x * x * y (integer)
# 13. xy2br - mean of x * y * y (integer)
# 14. x-ege - mean edge count left to right (integer)
# 15. xegvy - correlation of x-ege with y (integer)
# 16. yege - mean edge count bottom to top (integer)
# 17. yegvx - correlation of y-ege with x (integer)

# METHOD: KNN
# K Nearest Neighbor (KNN) is a simple and versatile machine learning algorithm that is used in many different areas, 
# such as handwriting detection, image recognition, and video recognition. The dataset that will be used in this 
# project is the letter-recognition multi-class classification dataset from UCI. We will be aiming to classify  
# black and white pixel images of capital letters from the English alphabet into one of 26 groups of letters.
# KNN is an algorithm, based on the local minimum of the target function which is used to learn an unknown function
# of desired precision and accuracy. The algorithm also finds the neighborhood of an unknown input, its range 
# or distance from it, and other parameters, which is accordance with the features of the letter-recognition dataset

# PROS AND CONS: KNN
# The pros of KNN include that it is a simple and intuitive algorithm, it is a non-assumptive algorithm (doesn't require any assumptions),
# does not require training steps, is adaptable to multi-class data sets (letter-recognition is multi-class), 
# is constantly evolving, and has a variety of distance criteria to choose from (Euclidean, etc.)
# The cons of KNN include that it is generally a slow algorithm and will take some time to run, becomes more 
# inefficient with a greater number of input variables, there is an optimal number of neighbors that must be 
# selected, KNN is sensitive to outliers, and KNN does not do well with missing values 

# Read and load the csv file
letters <- read.csv("letter-recognition.csv",na.strings=c("","NA"))


##########################
# PRE-PROCESSING THE DATA
##########################

# Preview the data
head(letters)
View(letters)

# Note that the target variable 'letter' is type: char and the remaining columns are type: int
# We will not discard any of the columns, as they are not identifying features that will interfere with the KNN algorithm
str(letters)

# Convert integer types to numeric (so we can create plots)
letters[2:17] <- lapply(letters[2:17], as.numeric)
df<-letters
#df$letter <- as.factor(df$letter)

# vector of letters (this is the target variable)
myletters <- c("A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z")

# convert letters to integers 1-26 (A=1,B=2,...,Z=26)
df <- df %>%
  mutate(letter2=match(letter,myletters))
df$letter<-df$letter2 # update letter column 
df<-df[-c(18)]# remove mutated column

# check data
str(df)
View(df)

# check features
summary(df)

# the outcome we want to predict
View(table(df$letter))

##################
# VISUALIZATIONS
##################
# Histogram for the count of observations (letters) faceted over each attribute
# Note that we have some right skewed plots (onpix, x2bar, xbox, yedge, xedge)
df%>%
  pivot_longer(cols=xbox:yedgex,names_to="measurement",values_to="value")%>%
  ggplot() +
  geom_histogram(mapping=aes(x=value))+
  facet_wrap(~ measurement, scales = "free_x") +
  labs(title="Distribution of observations (letters) faceted over each attribute")
# Bar plot for every attribute column faceted over 'letter'
# Note that we have some distributions that are bi-modal/multi-modal, indicating that there are certain letters that have distinct characteristics
df%>%
  pivot_longer(cols=xbox:yedgex,names_to="measurement",values_to="value")%>%
  ggplot() +
  geom_bar(mapping=aes(x=letter,y=value),stat='identity')+
  facet_wrap(~ measurement, scales = "free_x") +
  labs(title="Distribution of attributes per letter")

#######################
# NORMALIZING THE DATA 
#######################

# Note the slight differences in distribution of the values of the features (some attributes are skewed), so we will normalize the data
# create normalizing function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# normalize the data
#df_norm <- df
df_norm <- as.data.frame(lapply(df[,2:17], normalize))
#letter <- df$letter
#df_norm<-cbind(letter,df_norm) # use this only for the caret package and runs + visualization
View(df_norm)


# we can use the package class for KNN
library(class)

set.seed(1111)
#random selection of 70% data.
s1 <- sample(1:nrow(df_norm),size=nrow(df_norm)*0.7,replace = FALSE)

train.letter <- df[s1,] # 70% training data
test.letter <- df[-s1,] # remaining 30% test data

#Creating seperate dataframe for 'letter' feature which is our target.
train.letter_labels <- df[s1,1]
test.letter_labels <-df[-s1,1]

#Find the number of observation
NROW(train.letter_labels) 

# Note there are 1400 observations, and the square root of 1400 is approximately 38
# so we will create two models: 
# one with ‘K’ value as 37 and the other model with a ‘K’ value as 38.

knn.37 <- knn(train=train.letter, test=test.letter, cl=train.letter_labels, k=37)
knn.38 <- knn(train=train.letter, test=test.letter, cl=train.letter_labels, k=38)

#Calculate the proportion of correct classification for k = 37, 38
ACC.37 <- 100 * sum(test.letter_labels == knn.37)/NROW(test.letter_labels)
ACC.38 <- 100 * sum(test.letter_labels == knn.38)/NROW(test.letter_labels)

# accuracy
ACC.37 # overall accuracy of 97.06667%
ACC.38 # overall accuracy of 97.06667%

# confusion matrix with caret package
library(caret)
# note that class 10 = J has the lowest accuracy at 94.51%
# note that class 24= X has the highest accuracy at 99.97%
confusionMatrix(table(knn.38 ,test.letter_labels))

# most accurate 'K' value
i=1
k.optm=1
for (i in 1:38){
  knn.mod <- knn(train=train.letter, test=test.letter, cl=train.letter_labels, k=i)
  k.optm[i] <- 100 * sum(test.letter_labels == knn.mod)/NROW(test.letter_labels)
  k=i
  cat(k,'=',k.optm[i],'')}

# Accuracy plot according to K values
plot(k.optm, type="b", xlab="K- Value",ylab="Accuracy level")

# According to the plot, we see that with increasing K values, the accuracy decreases
# Over-fitting would occur at K=1, and according to the elbow trend, 
# it appears that a good K value would be approximately between K = 18 and K=20


#####################
# SVM Comparison
#####################
df_svm <- df
df_svm$letter <- as.factor(df$letter) # need to set letter as a factor variable

set.seed(1212)
letters_train <- df_svm[1:14000, ] # 70% training
letters_test <- df_svm[14001:20000, ] # 30% training

library(kernlab)
letter_classifier <- ksvm(letter ~ ., data = letters_train, kernel = "vanilladot") ## we specify the linear (that is, vanilla) kernel using the vanilladot option

letter_classifier # training error: 0.1293
## This information tells us very little about how well the model will perform in the real world. We need to examine its performance on the testing dataset to know whether it generalizes well to unseen data

letter_predictions <- predict(letter_classifier, letters_test)
head(letter_predictions)

table(letter_predictions, letters_test$letter) # we compare the predicted letters with the true letters
agreement <- letter_predictions == letters_test$letter
table(agreement) # classifier correctly identified the letter in 5,061 out of the 6,000 

prop.table(table(agreement)) # 84.35% accuracy

## Improving Model Performance
## A popular convention is to begin with the Gaussian RBF kernel, which has been shown to perform well for many types of data. We can train an RBF-based SVM.

letter_classifier_rbf <- ksvm(letter ~ ., data = letters_train, kernel = "rbfdot")
letter_predictions_rbf <- predict(letter_classifier_rbf, letters_test)
agreement_rbf <- letter_predictions_rbf == letters_test$letter
table(agreement_rbf) # predicted 5,573 out of 6,000 letters correctly
prop.table(table(agreement_rbf)) # 92.88% accuracy 

## By simply changing the kernel function, we were able to increase the accuracy of our character recognition model from 84 percent to 93 percent.


# Compared to the SVM algorithm, KNN performed significantly better across all values of K from 1-38.
# Perhaps because SVMs are most easily understood when used for binary classification

# KNN Accuracy: 97%  
# SVM Accuracy: 93% 

