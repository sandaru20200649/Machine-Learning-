library(readxl)
library(dplyr) 
library(neuralnet)
library(Metrics)
library(grid)
library(gridExtra)

#Get the uow_consumption dataset loaded
data <- read_excel("/Users/sandaru/Desktop/ML/CW/ML1_CW/vehicles.xlsx", sheet = 1)

#----------------------------------------Preprocessing----------------------------------------------
names(data)[2] <- 'six'
names(data)[3] <- 'seven'
names(data)[4] <- 'eight'

# change date to numeric
date <-factor(data$date)
date <-as.numeric(date)
date

# create data frame
new_dataset_frame <- data.frame(data,data$'six',data$'seven',data$'eight')

# create dataframe for 8
eight_column <- c(new_dataset_frame$data.eight)
plot(eight_column, type = "l")

# create I/O matrix
main_time_delayed_matrix <- bind_cols(t7 = lag(eight_column,8),
                                      t4 = lag(eight_column,5),
                                      t3 = lag(eight_column,4),
                                      t2 = lag(eight_column,3),
                                      t1 = lag(eight_column,2),
                                      eightHour = eight_column) 

time_delayed_matrix <- na.omit(main_time_delayed_matrix)

#dividing the data into train and test
train_data <- time_delayed_matrix[1:380,]
test_data <- time_delayed_matrix[381:nrow(time_delayed_matrix),]

#max min values
min_val <- min(train_data)
max_val <- max(train_data)
dataset_output <- test_data$eightHour

#------------------------------------------normalize--------------------------------------------
#normalization function for min-max
normalize <- function(x){
  return ((x - min(x)) / (max(x) - min(x)))
}

# un-normalizing function
unnormalize <- function(x, min, max) {
  return( (max - min)*x + min )
}
#apply Normalization
time_delayedNorm <- as.data.frame(lapply(time_delayed_matrix[1:ncol(time_delayed_matrix)], normalize))

#after normalizing test data, separating the data
train_dataset_norm <- time_delayedNorm[1:380,]
test_dataNorm <- time_delayedNorm[381:nrow(time_delayed_matrix),]

head(time_delayed_matrix)

#before & after normalization
boxplot(time_delayed_matrix, main="data before normalization")
boxplot(time_delayedNorm, main="after data normalization")

#the generation of testing data for each timeDelay
t1_testing_data <- as.data.frame(test_dataNorm[, c("t1")])
t2_testing_data <- test_dataNorm[, c("t1", "t2")]
t3_testing_data <- test_dataNorm[, c("t1", "t2", "t3")]
t4_testing_data <- test_dataNorm[, c("t1", "t2", "t3", "t4")]
t7_testing_data<- test_dataNorm[, c("t1", "t2", "t3", "t4", "t7")]

#ability to train an AR model
trainModel <- function(formula, hiddenVal, isLinear, actFunc,inputs,hidden){
  
  my_text <- paste(inputs,"inputs and",length(hidden),"hidden layers","(",paste(hidden, collapse=","),") \n")
  
  set.seed(1234)
  nn <- neuralnet(formula,data = train_dataset_norm, hidden=hiddenVal, act.fct = actFunc, linear.output=isLinear)
  plot(nn)
  
  plot_panel <- grid.grab(wrap = TRUE)
  
  ## create a title grob:
  plot_title <- textGrob(my_text,
                         x = .5, y = .50,
                         gp = gpar(lineheight = 2,
                                   fontsize = 15, col = 'red',
                                   adj = c(1, 0)
                         )
  )
  
  #stack title and main panel, and plot:
  grid.arrange(
    grobs = list(plot_title,
                 plot_panel),
    heights = unit(c(.15, .85), units = "npc"),
    width = unit(1, "npc")
  )
  dev.new()
  dev.off()
  return(nn)
}

testing_Model <- function(nnModel, testing_df,inputs,hidden){
  cat("There are",inputs,"inputs and",length(hidden),"hidden layers","(",paste(hidden, collapse=","),") \n")
  my_text <- paste(inputs,"inputs and",length(hidden),"hidden layers","(",paste(hidden, collapse=","),") \n")
  
  nnresults <- compute(nnModel, testing_df)
  predicted <- nnresults$net.result
  unnormalised_predicted <- unnormalize(predicted, min_val, max_val)
  devia = ((dataset_output - unnormalised_predicted)/dataset_output) 
  modelAccuracy = 1 - abs(mean(devia))
  accuracy = round(modelAccuracy * 100 , digits = 2)
  
  plot( dataset_output,unnormalised_predicted, col = 'green', main = "Unnormalized Prediction Graph NARX", pch = 18, cex = 0.7)
  mtext(my_text,  side = 3, line = 2, cex = 0.8)
  abline(0,1,lwd=2)
  legend("bottomright", legend = 'NN', pch = 18, col = 'green')
  dev.new()
  dev.off()
  
  
  x = 1:length(dataset_output)
  plot(x, dataset_output, col = "red", type = "l", lwd=2,
       main = "concrete strength prediction")
  mtext(my_text,  side = 3, line = 2, cex = 0.8)
  lines(x, unnormalised_predicted, col = "blue", lwd=2)
  legend("topright", legend = c("original-strength", "predicted-strength"),
         fill = c("red", "blue"), col = 2:3, adj = c(0, 0.6))
  grid()
  dev.new()
  dev.off()
  
  
  rmse = rmse(dataset_output, unnormalised_predicted)
  mae = mae(dataset_output, unnormalised_predicted)
  mape = mape(dataset_output, unnormalised_predicted)
  smape = smape(dataset_output, unnormalised_predicted)
  
  cat("Model Accuracy:", accuracy, "%\n")
  cat("RMSE:", rmse, "\n")
  cat("MAE:", mae, "\n")
  cat("MAPE:", mape, "\n")
  cat("sMAPE:", smape, "\n")
  cat("\n\n")
  
  return(unnormalised_predicted)
}


#t1 Train models using various hidden layer sizes.
hidden_layers_count <- list( c(8),c(5, 3))

for (i in seq_along(hidden_layers_count)) {
  model <- trainModel(eightHour ~ t1, hidden_layers_count[[i]], isLinear = FALSE, "tanh",1,hidden_layers_count[[i]])
  pred <- testing_Model(model, t1_testing_data,1,hidden_layers_count[[i]])
  
}



# t2 Train models using various hidden layer sizes.
t2_training <- trainModel(eightHour ~ t1 + t2, c(8),isLinear = TRUE, "logistic",2,c(8))
test_t2_predict <- testing_Model(t2_training, t2_testing_data,2,c(8))



#t3 Train models using various hidden layer sizes.
hidden_layers_count <- list( c(4),c(7),c(6,8))

for (i in seq_along(hidden_layers_count)) {
  model <- trainModel(eightHour ~ t1 + t2 + t3 ,hidden_layers_count[[i]],isLinear = TRUE, "logistic",3,hidden_layers_count[[i]])
  
  pred <- testing_Model(model, t3_testing_data,3,hidden_layers_count[[i]])
  
}


#t4 Train models using various hidden layer sizes.
hidden_layers_count <- list( c(5),c(9),c(5,2),c(10,5))

for (i in seq_along(hidden_layers_count)) {
  model <- trainModel(eightHour ~ t1 + t2 + t3 + t4,hidden_layers_count[[i]],isLinear = TRUE, "logistic",4,hidden_layers_count[[i]])
  pred <- testing_Model(model, t4_testing_data,4,hidden_layers_count[[i]])
  
}


#t7 Train models using various hidden layer sizes.
hidden_layers_count <- list( c(5),c(10),c(6,2),c(10,8),c(7,6))

for (i in seq_along(hidden_layers_count)) {
  model <- trainModel(eightHour ~ t1 + t2 + t3 + t4 + t7,hidden_layers_count[[i]],isLinear = TRUE, "logistic",7,hidden_layers_count[[i]])
  pred <- testing_Model(model, t7_testing_data,7,hidden_layers_count[[i]])
  
}
#---------------------------------------------substask 02----------------------------------------#

#combining columns six and seven
time_delayed_matrix <- cbind(new_dataset_frame[,2:3], main_time_delayed_matrix)

time_delayed_matrix <- na.omit(time_delayed_matrix)

#dividing the data into train and test
train_data <- time_delayed_matrix[1:380,]
test_data <- time_delayed_matrix[381:nrow(time_delayed_matrix),]

#max min values
min_val <- min(train_data)
max_val <- max(train_data)
dataset_output <- test_data$eightHour

# apply Normalization
time_delayedNorm <- as.data.frame(lapply(time_delayed_matrix[1:ncol(time_delayed_matrix)], 
                                         normalize))
#splitting data after normalize test data
train_dataset_norm <- time_delayedNorm[1:380,]
test_dataNorm <- time_delayedNorm[381:nrow(time_delayed_matrix),]

#view before & after normalization
boxplot(time_delayed_matrix, main="data before normalization")
boxplot(time_delayedNorm, main="after data normalization") 

#the generation of testing data for each timeDelay
t1_testing_data <- test_dataNorm[, c("data.six","data.seven", "t1")]
t2_testing_data <- test_dataNorm[, c("data.six","data.seven", "t1", "t2")]
t3_testing_data <- test_dataNorm[, c("data.six","data.seven", "t1", "t2", "t3")]
t4_testing_data <- test_dataNorm[, c("data.six","data.seven", "t1", "t2", "t3", "t4")]
t7_testing_data<- test_dataNorm[, c("data.six","data.seven", "t1", "t2", "t3", "t4", "t7")]

#the input characteristics and the appropriate test data should be defined.
inputs <- c( "t1", "t2", "t3","t4","t7")
test_data <- list(t1_testing_data, t2_testing_data, t3_testing_data,t4_testing_data,t7_testing_data)


#Train models using various hidden layer sizes.

h1 <- list(c(9))
h2 <- list(c(7))
h3 <- list(c(7))
h4 <- list(c(4,5))
h5 <- list(c(6,5),c(5,3))

hidden_layers_count <- list(h1,h2,h3,h4,h5)


for (i in seq_along(inputs)) {
  for(j in seq_along(hidden_layers_count[[i]])){
    
    current_inputs <- inputs[1:i]
    # cat(current_inputs)
    
    # Get the current hidden layer configuration
    currentHidden <- hidden_layers_count[[i]][[j]]  
    
    # Train the model
    formula <- as.formula(paste("eightHour ~ data.six + data.seven +", paste(current_inputs, collapse="+")))
    
    
    model <- trainModel(formula, currentHidden,isLinear = TRUE, "logistic",current_inputs,currentHidden)
    
    # Test the model
    test_predict <- testing_Model(model, test_data[[i]],(length(current_inputs)+2),currentHidden)
    
  }
}
