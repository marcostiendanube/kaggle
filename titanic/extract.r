#Clean memmory
rm(list = ls())
#Define working dirs
data_dir <- 'C:/Users/Marcos/Documents/Kaggle/titanic'

library('caret')
library('dplyr')

#Functions
drop_cols_ <- function(dt, drop_cols) {
  if (length(drop_cols) > 0){
      dt <- dt[, !colnames(dt) %in% drop_cols]
  }
  return (dt)
}

# DATA LOAD (build working data_set)
print('Load training data')
train_set <- read.csv(paste(data_dir,'/train.csv',sep=''), header = TRUE) #30% of data to fit the model
eval_set <- read.csv(paste(data_dir,'/test.csv',sep=''), header = TRUE) #30% of data to fit the model
str(train_set)
str(eval_set)

train_set$Survived <- as.factor(ifelse(train_set$Survived,'alive','dead'))
eval_set$Survived <- NA

data_set <- rbind(train_set,eval_set)
str(data_set)

# FEATURE ENGINEERING
print('FEATURE ENGINEERING')

data_set$cabin_letter <- as.factor(substr(data_set$Cabin,0,1))
data_set[data_set$cabin_letter == '','cabin_letter'] <- NA

data_set[data_set$Embarked == '','Embarked'] <- NA
data_set$Pclass <- as.factor(data_set$Pclass) 
data_set$family_size <- data_set$SibSp + data_set$Parch

data_set$title <- gsub('(.*, )|(\\..*)', '', data_set$Name)
data_set[!(data_set$title %in% c('Master','Miss','Mr','Mrs')),'title'] <- 'Other'
data_set$title <- as.factor(data_set$title)

print('drop unussed cols')
data_set <- drop_cols_(data_set,c('Name','Cabin','Ticket','SibSp','Parch'))
print('saving Data Set')

train_set <- data_set[!is.na(data_set$Survived),]
eval_set <- data_set[is.na(data_set$Survived),]

saveRDS(train_set, file = paste(data_dir,'/train_set.rds', sep = ''))
saveRDS(eval_set, file = paste(data_dir,'/eval_set.rds', sep = ''))
