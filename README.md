# RStudio
RStudio for Analysis
```R
# CREDIT CARD FRAUD DECTATION
# DATA670 - DATA ANALYTICS CAPSTONE PROJECT
# STUDENT NAME - SEMA KEHDALA
# PROFESSORS NAME - JON McKEEBY
# ========================================

#library needed for analysis

install.packages('caret')
library(caret)

install.packages('ranger')
library(ranger)

install.packages('dplyr')
library(dplyr)

install.packages('ggplot2')
library(ggplot2)

install.packages('caTools')
library(caTools)

install.packages('ROSE')
library(ROSE)

install.packages('ROSE')
library(ROSE)

install.packages("smotefamily")
library(smotefamily)

install.packages('rpart')
library(rpart)

install.packages('rpart.plot')
library(rpart.plot)

install.packages('data.table')
library(data.table)

# --------------------------------------------------------

# opening the credit card file in R
credit_card <- read.csv(file.choose(), header=TRUE, sep=",", as.is=FALSE)


#View structure of the data
str(credit_card)

#convert class to factor variable
credit_card$Class <- factor(credit_card$Class, levels = c(0,1))

#get the summary of data
summary(credit_card)

#count the missing values
sum(is.na(credit_card))

# ----------------------------------------------------------

#get the distribution of fraud and legitimate transactions
table(credit_card$Class)

#get the percentage of fraud and legit transaction
prop.table(table(credit_card$Class))

#pie chart of credit card transactions
labels <- c("legit", "fraud")
labels <- paste(labels, round(100*prop.table(table(credit_card$Class)), 2))
labels <- paste0(labels, "%")

pie(table(credit_card$Class), labels, col = c("orange", "red"),
    main = "Pie chart of credit card transactions")

# ------------------------------------------------------------

#Creating a sample model without name (Scatter plot)
#No model predictions
predictions <- rep.int(0, nrow(credit_card))
predictions <- factor(predictions, levels = c(0, 1))

#install.packages('caret')
library(caret)
confusionMatrix(data = predictions, reference = credit_card$Class)

# -------------------------------------------------------------

install.packages('dplyr')
library(dplyr)

set.seed(1)
credit_card <- credit_card %>% sample_frac(0.1)

table(credit_card$Class)

install.packages('ggplot2')
library(ggplot2)

ggplot(data = credit_card, aes(x = V1, y = V2, col = Class)) +
  geom_point() +
  theme_bw() +
  scale_color_manual(values = c('dodgerblue2', 'red')) +
  ggtitle("Scatter Plot with Red as fraud and blue as legit")

# ---------------------------------------------------------------

#creating training and test sets for Fraud detection model

#install.packages('caTools')
library(caTools)

set.seed(123)

data_sample = sample.split(credit_card$Class,SplitRatio = 0.80)

train_data = subset(credit_card,data_sample==TRUE)

test_data = subset(credit_card,data_sample==FALSE)

dim(train_data)
dim(test_data)

# ---------------------------------------------------------------

# Balansing the dataset with the following method ROS & RUS
# Random over-sampling(ROS)

table(train_data$Class)

n_legit <- 22750
new_frac_legit <- 0.50
new_n_total <- n_legit/new_frac_legit # = 22750/0.50

#install.packages('ROSE')
library(ROSE)
oversampling_result <- ovun.sample(Class ~ .,
                                   data = train_data,
                                   method = "over",
                                   N = new_n_total,
                                   seed = 2019)

oversampled_credit <- oversampling_result$data

table(oversampled_credit$Class)

ggplot(data = oversampled_credit, aes(x = V1, y = V2, col = Class)) +
  geom_point(position = position_jitter(width = 0.1)) +
  theme_bw() +
  scale_color_manual(values = c('dodgerblue2', 'red'))

# -----------------------------------------------------------------

#Random under-sampling(RUS)

table(train_data$Class)

n_fraud <- 35
new_frac_fraud <- 0.50
new_n_total <- n_fraud/new_frac_fraud # = 35/0.50

#install.packages('ROSE')
library(ROSE)
undersampling_result <- ovun.sample(Class ~ .,
                                    data = train_data,
                                    method = "under",
                                    N = new_n_total,
                                    seed = 2019)

undersampled_credit <- undersampling_result$data

table(undersampled_credit$Class)

ggplot(data = undersampled_credit, aes(x = V1, y = V2, col = Class)) +
  geom_point() +
  theme_bw()+
  scale_color_manual(values = c('dodgerblue2', 'red'))

#----------------------------------------------------------------------

# Using both methods ROS and RUS

n_new <- nrow(train_data) # = 22785
fraction_fraud_new <- 0.50

sampling_result <- ovun.sample(Class ~ .,
                               data = train_data,
                               method = "both",
                               N = n_new,
                               p = fraction_fraud_new,
                               seed = 2019)

sampled_credit <- sampling_result$data

table(sampled_credit$Class)

prop.table(table(sampled_credit$Class))

ggplot(data = sampled_credit, aes(x = V1, y = V2, col = Class)) +
  geom_point(position = position_jitter(width = 0.2)) +
  theme_bw()+
  scale_color_manual(values = c('dodgerblue2', 'red'))

#------------------------------------------------------------------

#using the Synthetic Minority Oversampling Technique SMOTE to balance the dataset

#install.packages("smotefamily")
library(smotefamily)

table(train_data$Class)

#setting the number of fraud with the legitimate cases, and the desire percentage of legitimate cases

n0 <- 22750 #legitimate cases
n1 <- 35    #fraud cases
r0 <- 0.6   # the ratio after SMOTE

# calculate the value for the dup_size parameter of SMOTE
ntimes <- ((1 - r0) / r0) * (n0 / n1) -1

smote_output = SMOTE(X = train_data[ , -c(1, 31)],
                     target = train_data$Class, 
                     K = 5, 
                     dup_size = ntimes)

credit_smote <- smote_output$data

colnames(credit_smote)[30] <- "Class"

prop.table(table(credit_smote$Class))

#class distribution for original dataset
ggplot(train_data, aes(x = V1, y = V2, color = Class)) +
  geom_point() +
  scale_color_manual(values = c('dodgerblue2', 'red'))

#class distribution for over-sampled dataset using SMOTE
ggplot(credit_smote, aes(x = V1, y = V2, color = Class)) +
  geom_point() +
  scale_color_manual(values = c('dodgerblue2', 'red'))

#-----------------------------------------------------------

#Decision tree using the SMOTE dataset

#install.packages('rpart')
#install.packages('rpart.plot')

library(rpart)
library(rpart.plot)

CART_model <- rpart(Class ~ . , credit_smote)

#sample of the decision tree on column V14
rpart.plot(CART_model, extra = 0, type = 5, tweak = 1.2)

#Predict fraud classes
predicted_val <- predict(CART_model, test_data, type = 'class')

#Build confusion matrix
#install.packages('caret')

library(caret)

confusionMatrix(predicted_val, test_data$Class)

#-------------------------------------------------------------------

predicted_val <- predict(CART_model, credit_card[-1], type = 'class')
confusionMatrix(predicted_val, credit_card$Class)

#-------------------------------------------------------------------

#Decision tree on trained dataset without SMOTE

CART_model <- rpart(Class ~ . , train_data[,-1])

rpart.plot(CART_model, extra = 0, type = 5, tweak = 1.2)

#predict fraud classes
predicted_val <- predict(CART_model, test_data[-1], type = 'class')

library(caret)
confusionMatrix(predicted_val, test_data$Class)

#====================================================================

# Method 2 Logistic Regression Model on test data

# import dataset as data
data <- read.csv(file.choose(), header=TRUE, sep=",", as.is=FALSE)

data.table(data)
table(data$Class)
names(data)

summary(data$Amount)
sd(data$Amount)
IQR(data$Amount)
var(data$Amount)

# Data normalization

data$Amount <- scale(data$Amount)
data2 <- data[, -c(1)]  #Removing time column
head(data2)

set.seed(12)
library(caTools)

sample_data <- sample.split(data2$Class, SplitRatio = 0.80)

train_data <- subset(data2, sample_data == TRUE)
test_data <- subset(data2, sample_data == FALSE)

dim(train_data)
dim(test_data)

# Logistic Regression Model on test data

Logistic_Model <- glm(Class~. , test_data,family = binomial())
summary(Logistic_Model)

plot(Logistic_Model)

# Logistic Regression Model on train data

Logistic_Model1 <- glm(Class~. , train_data,family = binomial())
summary(Logistic_Model1)

plot(Logistic_Model1)

#Working on getting the ROC curve (about 90% accuracy)

library(pROC)
lr.predict <- predict(Logistic_Model1, test_data, probability = TRUE)
auc.gb <- roc(test_data$Class, lr.predict, plot = TRUE, col = "green")

#Working on getting the Decision tree

desicion_model <- rpart(Class ~ . , data, method = "class")
predict_val <- predict(desicion_model, type = "class")
probability <- predict(desicion_model, data, type = 'prob')

#plot the decision tree
rpart.plot(desicion_model, main = "Decision Tree Visualization", 
           cex = 0.5, 
           under.cex = 0.2)

#Working on getting the Neural Network NN (it took lots of time for the NN)
library(neuralnet)

# Prepare the training data
set.seed(123) # For reproducibility
train_data <- data.frame(
  Feature1 = rnorm(100),
  Feature2 = rnorm(100),
  Feature3 = rnorm(100),
  Feature4 = rnorm(100),
  Class = sample(0:1, 100, replace = TRUE))

# Train the neural network model with 18 hidden neurons 3 hidden layers
NN_model <- neuralnet(Class ~., data = train_data, 
                      hidden = c(6, 6, 6), linear.output = FALSE)

# Plot the neural network model
plot(NN_model)

# clear environment
rm(list=ls())
```
















