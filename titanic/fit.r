# SETUP
#Clean memmory
rm(list = ls())

#Define working dirs
data_dir <- 'C:/Users/Marcos/Documents/Kaggle/titanic'
#Load required libraries
library(caret)
library(dplyr)
library(MLmetrics)
# DATA LOAD (build working data_set)
print('DATA LOAD')
train_set <- readRDS(paste(data_dir,'/train_set.rds', sep = ''))
str(train_set)

# MODEL FIT
print('MODEL FIT')
#CREO EL MODELO
fitControl <- trainControl(method="LGOCV",  # Esto da igual
                           number = 3,
                           verboseIter=TRUE,
                           returnData=FALSE, # No le pido que me devuelve el dataset OJO! dataset muy grandes
                           summaryFunction=twoClassSummary, #Te devuelve el area bajo la curva de ROC
                           classProbs=TRUE)

xgbGrid <- expand.grid(nrounds = c(100,150),
                      max_depth = c(1,2,5),
                      eta = c(0.2, 0.3, 0.4),
                      gamma = c(0,1,3,5,10),
                      colsample_bytree = seq(0.5, 1, by = 0.05),
                      min_child_weight = c(1, 2, 5,10),
                      subsample = seq(0.5, 1, by = 0.05))
#Armo todas las combinaciones posibles de hiperparametros -> Grilla.
# xgbGrid <- expand.grid(nrounds = c(100),
#                        max_depth = c(10),
#                        eta = c(0.2),
#                        gamma = c(1),
#                        colsample_bytree = c(0.75),
#                        min_child_weight = c(10),
#                        subsample = c(0.95))

# Entreno el modelo con una seleccion de 15 combinaciones de la grilla
xgbFit <- train(Survived ~ ., data=select(train_set,-PassengerId),
                method = "xgbTree",
                trControl = fitControl,
                tuneGrid = xgbGrid[sample(c(1:nrow(xgbGrid)), 60),], #Sampleo 15 puntos de la grilla de hiperparametros
                metric = "ROC",
                na.action = na.pass,
                allowParallel=TRUE)

# print('BEST FITTING RESULT')
# print('eta: %f | max_depth: %i | colsample_bytree: %f | min_child_weight: %i | subsample: %f | nrounds: %i | gamma: %i',
#           xgbFit$results[which.max(xgbFit$results$ROC),'eta'],xgbFit$results[which.max(xgbFit$results$ROC),'max_depth'],
#           xgbFit$results[which.max(xgbFit$results$ROC),'colsample_bytree'],xgbFit$results[which.max(xgbFit$results$ROC),'min_child_weight'],
#           xgbFit$results[which.max(xgbFit$results$ROC),'subsample'],xgbFit$results[which.max(xgbFit$results$ROC),'nrounds'],
#           xgbFit$results[which.max(xgbFit$results$ROC),'gamma'])
# print('ROC: %f ',xgbFit$results[which.max(xgbFit$results$ROC),'ROC'])
print(xgbFit$results[which.max(xgbFit$results$ROC),'ROC'])
print(varImp(xgbFit)$importance,row.names = TRUE)

# PREDICT
print('PREDICT')
eval_set <- readRDS(paste(data_dir,'/eval_set.rds', sep = ''))
# Hago la prediccion con los datos de evaluacion
preds_eval <- predict(xgbFit,
                      newdata = select(eval_set,-PassengerId),
                      type="raw", #Devolveme las probabilidades
                      na.action = na.pass)

# Lo paso a un archivo de texto
options(scipen=10) #para sacar la notacion cientifica
#Creo el data set para hacer submit en kaggle (id, prob)
submit <- data.frame(PassengerId=eval_set$PassengerId,
                    Survived= ifelse(preds_eval=='dead',0,1))

# F1_Score(y_pred = preds_eval, y_true = eval_set$Survived, positive = 'alive')
#Exporto a csv el dataset para subir a kaggle
write.table(submit,
            paste(data_dir,'/preds_submit_',format(Sys.time(), "%Y%m%d%H%M%S"),'.csv', sep = ''),
            sep=",", quote=FALSE, row.names=FALSE)

print('DONE')
