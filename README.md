# Credit Cards Fraud Detection Analysis

## Overview 
Credit card fraud detection is a critical issue faced by financial institutions, as fraud can lead to significant monetary losses. Using machine learning and statistical methods in R, my aims with this analysis is to detect fraudulent transactions by identifying patterns and anomalies in transaction data. The dataset was gotten from an open source website called [kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)

### Dataset 

The dataset i used typically consists of various features, including transaction time, amount, and anonymized variables (e.g., V1, V2, ..., V28), derived using Principal Component Analysis (PCA).
The target variable, "Class," indicates whether a transaction is fraudulent (1) or non-fraudulent (0).
The dataset is highly imbalanced, with fraudulent transactions comprising a small percentage of the total

![image](https://github.com/user-attachments/assets/1cad08c8-79fb-46b2-952c-3b9b8d53e99c)

Since fraud cases are rare, balancing techniques like oversampling (SMOTE), undersampling, or adjusting class weights are applied.
Below are some of the packages needed in this project.

### Packages used
```R
#library needed for this analysis

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
```
### Reading CSV file in RStudio
The are several ways to open CSV file in RStudio, 
i used the first opetion as it will open the file same as excel without converting factors to characters.
```R
# opening the credit card file in R without converting factors to characters
credit_card <- read.csv(file.choose(), header=TRUE, sep=",", as.is=FALSE)

# open a csv file in R while converting factors to characters
credit_card <- read.csv(file.choose(), header=T)
````
If required, this command reomve the ID column from dataset called credit_card.
```R
#If required, this command reomve the ID column from dataset called credit_card.
credit_card <- credit_card[, -which(names(credit_card) == "ID")]
```
![image](https://github.com/user-attachments/assets/1cad08c8-79fb-46b2-952c-3b9b8d53e99c)

### Viewing the structure and summary of the dataset
```R
#View structure of the data
str(credit_card)
```
![image](https://github.com/user-attachments/assets/03c0dfdb-658f-4eb8-905e-eb58ebe8163e)
```R
#convert class to factor variable
credit_card$Class <- factor(credit_card$Class, levels = c(0,1))
```
```R
#get the summary of data
summary(credit_card)
```
![image](https://github.com/user-attachments/assets/653f474c-c858-4e0a-8644-3a1b55255ff8)

```R
#count the missing values
sum(is.na(credit_card))
```
# ----------------------------------------------------------
```R
#get the distribution of fraud and legitimate transactions
table(credit_card$Class)
```
![image](https://github.com/user-attachments/assets/58be22f1-91b5-4606-afa2-02adb7a63f42)

```R
#get the percentage of fraud and legit transaction
prop.table(table(credit_card$Class))
```
![image](https://github.com/user-attachments/assets/60086b03-0613-44c9-9872-ce291fa2190b)

```R
#pie chart of credit card transactions
labels <- c("legit", "fraud")
labels <- paste(labels, round(100*prop.table(table(credit_card$Class)), 2))
labels <- paste0(labels, "%")

pie(table(credit_card$Class), labels, col = c("orange", "red"),
    main = "Pie chart of credit card transactions")
```
![image](https://github.com/user-attachments/assets/84087f35-9d86-4c68-81de-546fb872b218)

# ------------------------------------------------------------
```R
#Creating a sample model without name (Scatter plot)
#No model predictions
predictions <- rep.int(0, nrow(credit_card))
predictions <- factor(predictions, levels = c(0, 1))

#install.packages('caret')
library(caret)
confusionMatrix(data = predictions, reference = credit_card$Class)
```
![image](https://github.com/user-attachments/assets/0424d5e9-68e1-4b29-80e4-fdeff683a9ff)

# -------------------------------------------------------------
```R
install.packages('dplyr')
library(dplyr)

set.seed(1)
credit_card <- credit_card %>% sample_frac(0.1)

table(credit_card$Class)
```
![image](https://github.com/user-attachments/assets/4f1034c0-edf8-49b3-a92c-a4a8b30fb902)

```R
install.packages('ggplot2')
library(ggplot2)

ggplot(data = credit_card, aes(x = V1, y = V2, col = Class)) +
  geom_point() +
  theme_bw() +
  scale_color_manual(values = c('dodgerblue2', 'red')) +
  ggtitle("Scatter Plot with Red as fraud and blue as legit")
```
![image](https://github.com/user-attachments/assets/d11c3700-befa-45b5-bc9d-d10ffdfd0876)

# ---------------------------------------------------------------

### creating training and test sets for Fraud detection model

```R
#install.packages('caTools')
library(caTools)

set.seed(123)

data_sample = sample.split(credit_card$Class,SplitRatio = 0.80)

train_data = subset(credit_card,data_sample==TRUE)

test_data = subset(credit_card,data_sample==FALSE)
```
```R
dim(train_data)
```
![image](https://github.com/user-attachments/assets/b57734f0-6edd-44c0-9ca1-596bcb66eaa1)
```R
dim(test_data)
```
![image](https://github.com/user-attachments/assets/158ee4b8-906e-4c04-a9ef-dccaf72b3e7c)

# ---------------------------------------------------------------

## Balansing the dataset with the following method ROS & RUS
### Random over-sampling(ROS)
```R
table(train_data$Class)
```
![image](https://github.com/user-attachments/assets/ca2b31c9-bdf4-4e6b-b75c-6d538451f647)

```R
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
```
![image](https://github.com/user-attachments/assets/79714cc5-3970-4491-9ad9-e6e8691552aa)

```R
ggplot(data = oversampled_credit, aes(x = V1, y = V2, col = Class)) +
  geom_point(position = position_jitter(width = 0.1)) +
  theme_bw() +
  scale_color_manual(values = c('dodgerblue2', 'red'))
```
The dataset is now balanced, however this can not be clearly noticed as the red points are overlaping each other

![image](https://github.com/user-attachments/assets/b645e570-e5d5-4824-a614-819137abdf99)

# -----------------------------------------------------------------

### Random under-sampling(RUS)
```R
table(train_data$Class)
```
![image](https://github.com/user-attachments/assets/17450215-869c-4285-bbae-3d697763d676)
```R
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
```
![image](https://github.com/user-attachments/assets/498d35dd-319b-4ff7-8354-edff19ffc59b)

```R
ggplot(data = undersampled_credit, aes(x = V1, y = V2, col = Class)) +
  geom_point() +
  theme_bw()+
  scale_color_manual(values = c('dodgerblue2', 'red'))
```
![image](https://github.com/user-attachments/assets/d61e2f44-1846-4eb8-8bd9-651b5e74b362)


#----------------------------------------------------------------------

### Using both methods ROS and RUS
```R
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
```
![image](https://github.com/user-attachments/assets/f3cd4b05-85c7-4cbe-9536-94c556840bc6)

```R
prop.table(table(sampled_credit$Class))
```
![image](https://github.com/user-attachments/assets/4807de52-9d75-4106-a2e2-b8bc7b6ccd78)
```R
ggplot(data = sampled_credit, aes(x = V1, y = V2, col = Class)) +
  geom_point(position = position_jitter(width = 0.2)) +
  theme_bw()+
  scale_color_manual(values = c('dodgerblue2', 'red'))
```
![image](https://github.com/user-attachments/assets/041ac18b-0465-482b-acc9-ee04240dbacc)

#------------------------------------------------------------------

### Using the Synthetic Minority Oversampling Technique SMOTE to balance the dataset

```R
#install.packages("smotefamily")
library(smotefamily)

table(train_data$Class)
```
![image](https://github.com/user-attachments/assets/6536ef7d-3eba-4d93-b1cc-654c84e904c7)

### Setting the number of fraud with the legitimate cases, and the desire percentage of legitimate cases
```R 
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
```
```R
prop.table(table(credit_smote$Class))
```
![image](https://github.com/user-attachments/assets/66410074-0778-410a-a478-fbf47a117c46)

```R
### class distribution for original dataset
ggplot(train_data, aes(x = V1, y = V2, color = Class)) +
  geom_point() +
  scale_color_manual(values = c('dodgerblue2', 'red'))
```
![image](https://github.com/user-attachments/assets/ffbdec27-4929-468b-a23b-52a6666ba3e9)

```R
#class distribution for over-sampled dataset using SMOTE
ggplot(credit_smote, aes(x = V1, y = V2, color = Class)) +
  geom_point() +
  scale_color_manual(values = c('dodgerblue2', 'red'))
```
![image](https://github.com/user-attachments/assets/22df9d5e-2685-4f42-91a7-173e2533f131)

#-----------------------------------------------------------

### Decision tree using the SMOTE dataset

```R
#install.packages('rpart')
#install.packages('rpart.plot')

library(rpart)
library(rpart.plot)

CART_model <- rpart(Class ~ . , credit_smote)

#sample of the decision tree on column V14
rpart.plot(CART_model, extra = 0, type = 5, tweak = 1.2)
```
![image](https://github.com/user-attachments/assets/5bb66e56-804c-4e88-ac85-8a031a4f2ef6)

```R
#Predict fraud classes
predicted_val <- predict(CART_model, test_data, type = 'class')

#Build confusion matrix
#install.packages('caret')

library(caret)

confusionMatrix(predicted_val, test_data$Class)
```
#-------------------------------------------------------------------
```R
predicted_val <- predict(CART_model, credit_card[-1], type = 'class')
confusionMatrix(predicted_val, credit_card$Class)
```
![image](https://github.com/user-attachments/assets/115f38a9-8173-4e62-9a3c-a26f31ff8c85)

#-------------------------------------------------------------------

### Decision tree on trained dataset without SMOTE

```R
CART_model <- rpart(Class ~ . , train_data[,-1])

rpart.plot(CART_model, extra = 0, type = 5, tweak = 1.2)
```
![image](https://github.com/user-attachments/assets/1b9371c4-e9cd-43d4-888d-b3fcdf6478bf)

```R
#predict fraud classes
predicted_val <- predict(CART_model, test_data[-1], type = 'class')

library(caret)
confusionMatrix(predicted_val, test_data$Class)
```
![image](https://github.com/user-attachments/assets/9660b3d9-3113-4afd-a058-b3159e1a232d)

#====================================================================

## Method 2 Logistic Regression Model on test data

```R
#import dataset in a function called data
data <- read.csv(file.choose(), header=TRUE, sep=",", as.is=FALSE)
```
```R
data.table(data)
```
![image](https://github.com/user-attachments/assets/b0042d9e-5fe3-4e27-afa6-ae18f1af30e2)

```R
table(data$Class)
```
![image](https://github.com/user-attachments/assets/3a5eae3b-71d9-4e8c-a273-13071b538341)

```R
names(data)
```
![image](https://github.com/user-attachments/assets/56c7d58f-c920-4538-9e2c-a9fba9a1ba02)

```R
summary(data$Amount)
```
![image](https://github.com/user-attachments/assets/0b903f9b-caf4-463c-94dc-99d80d6b577b)

```R
sd(data$Amount)
```
![image](https://github.com/user-attachments/assets/1044f68a-d84b-41f3-923e-93b7c0bd2941)

```R
IQR(data$Amount)
```
![image](https://github.com/user-attachments/assets/4e47b807-fac2-4929-b93f-9793a189000d)

```R
var(data$Amount)
```
![image](https://github.com/user-attachments/assets/2a4598b6-674b-4c6a-a732-805b4d6671de)

### Data normalization

```R
data$Amount <- scale(data$Amount)
data2 <- data[, -c(1)]  #Removing time column
head(data2)
```
![image](https://github.com/user-attachments/assets/ac5eaa9d-c330-467b-a845-61370ef6fb93)

```R
set.seed(12)
library(caTools)

sample_data <- sample.split(data2$Class, SplitRatio = 0.80)

train_data <- subset(data2, sample_data == TRUE)
test_data <- subset(data2, sample_data == FALSE)
```
```R
dim(train_data)
```
![image](https://github.com/user-attachments/assets/5fb8d012-4fdb-4f31-abc2-9bcdb5ad36dc)

```R
dim(test_data)
```
![image](https://github.com/user-attachments/assets/ac664517-09a5-4e6f-835a-56e0d9bfbd77)

### Logistic Regression Model on test data

```R
Logistic_Model <- glm(Class~. , test_data,family = binomial())
summary(Logistic_Model)
```
![image](https://github.com/user-attachments/assets/d6dac3ad-0696-4a3d-8b61-9cab3dd801a7)

```R
plot(Logistic_Model)
```
(1) Residuals VS Fitted

![image](https://github.com/user-attachments/assets/12a23e3c-06c2-42fa-ba4f-3ba31556329e)

(2) Q-Q Residuals

![image](https://github.com/user-attachments/assets/9fd5d241-f85f-4a38-871c-12f8794c5309)

(3) Scale-Location

![image](https://github.com/user-attachments/assets/e6f343b1-e2bf-4184-ac6e-346c0638e46f)

(4) Residuals VS Leverage

![image](https://github.com/user-attachments/assets/f3a80617-c775-459c-894e-97cbf2150005)


### Logistic Regression Model on train data

```R
Logistic_Model1 <- glm(Class~. , train_data,family = binomial())
summary(Logistic_Model1)
```
![image](https://github.com/user-attachments/assets/8d9f8da6-ee60-4274-a8a1-cb4734ecc7b7)

```R
plot(Logistic_Model1)
```
(1) Residuals VS Fitted

![image](https://github.com/user-attachments/assets/506ddcff-e9c1-4ab0-b1ac-ee51184a492f)

(2) Q-Q Residuals

![image](https://github.com/user-attachments/assets/33523706-8326-4b4d-ac24-e3ff72f65404)

(3) Scale-Location

![image](https://github.com/user-attachments/assets/aab7fda2-964d-490b-8167-aa9407e191bb)

(4) Residuals VS Leverage

![image](https://github.com/user-attachments/assets/37bd5ad4-4255-4d18-8b29-ed84ad46e53b)


### Working on getting the ROC curve (about 90% accuracy)
```R
library(pROC)
lr.predict <- predict(Logistic_Model1, test_data, probability = TRUE)
auc.gb <- roc(test_data$Class, lr.predict, plot = TRUE, col = "green")
```
![image](https://github.com/user-attachments/assets/ea82b435-1b79-4c62-aeb9-52eadf9fbb09)

### Working on getting the Decision tree
```R
desicion_model <- rpart(Class ~ . , data, method = "class")
predict_val <- predict(desicion_model, type = "class")
probability <- predict(desicion_model, data, type = 'prob')

#plot the decision tree
rpart.plot(desicion_model, main = "Decision Tree Visualization", 
           cex = 0.5, 
           under.cex = 0.2)
```
![image](https://github.com/user-attachments/assets/d785c514-1e5d-4664-9567-a8b1a62ffb57)

### Working on getting the Neural Network NN (it took lots of time for the NN)

```R
library(neuralnet)

# Prepare the training data
set.seed(123) # For reproducibility
train_data <- data.frame(
  Feature1 = rnorm(100),
  Feature2 = rnorm(100),
  Feature3 = rnorm(100),
  Feature4 = rnorm(100),
  Class = sample(0:1, 100, replace = TRUE))
```
```R
# Train the neural network model with 18 hidden neurons 3 hidden layers
NN_model <- neuralnet(Class ~., data = train_data, 
                      hidden = c(6, 6, 6), linear.output = FALSE)

# Plot the neural network model
plot(NN_model)
```
![image](https://github.com/user-attachments/assets/d07d6083-ad77-48de-9bc5-95382b0d0222)

```R
# clear environment
rm(list=ls())
```
















