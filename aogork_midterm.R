# Alessa Ogork
# G01191082
# Midterm
# CDS 403-001

# Set directory and load packages
setwd("C:/Users/aless/OneDrive/CDS_403")
library("tidyverse")
library("dplyr")
library("ggplot2")

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
# or distance from it, and other parameters, which is accorance with the features of the letter-recognition dataset

# PROS AND CONS: KNN
# The pros of KNN include that it is a simple and intuitive algorithm, it is a non-assumptive algorithm (doesn't require any assumptions),
# does not require training steps, is adaptable to multi-class data sets (letter-recognition is multi-class), 
# is constantly evolving, and has a variety of distance criteria to choose from (Euclidean, etc.)
# The cons of KNN include that it is generally a slow algorithm and will take some time to run, becomes more 
# inefficient with a greater number of input variables, there is an optimal number of neighbors that must be 
# selected, KNN is sensitive to outliers, and KNN does not do well with missing values 

# Read and load the csv file
letters <- read.csv("letter-recognition.csv",na.strings=c("","NA"))

# Preview the data
head(letters)
View(letters)

# Note that the target variable 'letter' is type: char and the remaining columns are type: int
# We will not discard any of the columns, as they are not identifying features that will interfere with the KNN algorithm
str(letters)

# Convert integer types to numeric (so we can create plots)
letters[2:17] <- lapply(letters[2:17], as.numeric)
df<-letters

# convert letter to factor (this is the target variable)
df$letter <- as.factor(df$letter)
str(df)
View(df)

# check features
summary(df)

# check that there are no NA values in the df
table(is.na(df))

# the outcome we want to predict
View(table(df$letter))

# Histogram for the count of observations (letters) faceted over each attribute
df%>%
  pivot_longer(cols=xbox:yedgex,names_to="measurement",values_to="value")%>%
  ggplot() +
  geom_histogram(mapping=aes(x=value))+
  facet_wrap(~ measurement, scales = "free_x") +
  labs(title="Distribution of observations (letters) faceted over each attribute")
# Bar plot for every attribute column faceted over 'letter'
df%>%
  pivot_longer(cols=xbox:yedgex,names_to="measurement",values_to="value")%>%
  ggplot() +
  geom_bar(mapping=aes(x=value,y=letter),stat='identity')+
  facet_wrap(~ measurement, scales = "free_x") +
  labs(title="Distribution of attributes per letter")


# Note the slight differences in distribution of the values of the features (some attributes are skewed), so we will normalize the data
# create normalizing function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# normalize the data
df_norm <- as.data.frame(lapply(df[2:17], normalize))
df_norm$letter<-df$letter # use this only for the caret package and runs + visualization
View(df_norm)

# Histogram for the count of observations (letters) faceted over each attribute (normalized data)
df_norm%>%
  pivot_longer(cols=xbox:yedgex,names_to="measurement",values_to="value")%>%
  ggplot() +
  geom_histogram(mapping=aes(x=value))+
  facet_wrap(~ measurement, scales = "free_x") +
  labs(title="Distribution of observations (letters) faceted over each attribute with normalized data")
# Bar plot for every attribute column faceted over 'letter' with normalized data
df_norm%>%
  pivot_longer(cols=xbox:yedgex,names_to="measurement",values_to="value")%>%
  ggplot() +
  geom_bar(mapping=aes(x=value,y=letter),stat='identity')+
  facet_wrap(~ measurement, scales = "free_x") +
  labs(title="Distribution of attributes per letter with normalized data")

# remove letter column
df_norm<-df_norm[-c(letter)]

# now we create training and testing data sets
l<-length(row.names(df))

# split data set into 70% train and 30% test
dataset_train <- df_norm[1:round(0.7*l), ] 
dataset_test <- df_norm[round((0.7*l)+1):l, ]
dataset_train_labels <- df[1:round(0.7*l), 1] 
dataset_test_labels <- df[round((0.7*l)+1):l, 1]

# We will use the caret package to split the data and for KNN
df_norm$letter<-df$letter
library("caret")
validation_index <- createDataPartition(df_norm$letter, p=0.70, list=FALSE) # this is an alternate way to split the dataset into testing and training
validation <- df_norm[-validation_index,]
dataset_new <- df_norm[validation_index,]

control <- trainControl(method="cv", number=10)
metric <- "Accuracy"


set.seed(42)
fit.knn <- train(letter~., data=df_norm, method="knn", metric=metric, trControl=control) 
fit.knn

# alternatively, we can use the package class for KNN
df_norm<-df_norm[-c(17)]
library(class)
# I am currently working through an error here, the CrossTable code will also not run due to this portion
dataset_test_pred <- knn(train = dataset_train, test = dataset_test, cl = dataset_train_labels, k = 9) 

# Evaluating model performance
library(gmodels)
CrossTable(x = dataset_test_labels, y = dataset_test_pred, prop.chisq=FALSE) # I am currently working through an error here

# We can improve the model performance through 2 ways
# (1) We create a different normalization function - based on z-score
# (2) We can vary the value of k

dataset_z <- as.data.frame(scale(df[-1]))

dataset_z_train <- dataset_z[1:round(0.7*l), ]
dataset_z_test <- dataset_z[round((0.7*l)+1):l, ]
dataset_z_train_labels <- df[1:round(0.7*l), 1]
dataset_z_test_labels <- df[round((0.7*l)+1):l, 1]

dataset_z_test_pred <- knn(train = dataset_z_train, test = dataset_z_test, cl = dataset_z_train_labels, k = 9) 
CrossTable(x = dataset_z_test_labels, y = dataset_z_test_pred, prop.chisq = FALSE)

### We can vary the value of k and check for improvements to the model performance 












