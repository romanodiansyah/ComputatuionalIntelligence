#Penyiapan Data

#UNTUK PERCOBAAN HANYA DIRUN 1x SELAMA K FOLD
#---------------------BACA DATA--------------------
data <- read.csv("D:/DataCI_asli.csv", header = TRUE)
datasynth <- read.csv("D:/DataCI_Sintesis.csv", header = TRUE)

#--------------------PEMBAGIAN DATA-----------------------
#K FOLD VALIDATION
library(caret)
#ASLI
data_asli <- createFolds(data$age,k = 10, list = TRUE, returnTrain = FALSE)
#SINTESIS
data_sintesis <- createFolds(datasynth$age, k = 10,list = TRUE, returnTrain = FALSE)

#BAGIAN INI DIBAWAH DIRUN SEIRING GANTI FOLD
#PERGANTIAN FOLD DIBUAT MANUAL
#--------------------TRAIN TEST KFOLD----------
#data asli
x_train <- as.matrix(data[unlist(data_asli[-10],use.names = FALSE),-14])
y_train <- as.matrix(data[unlist(data_asli[-10],use.names = FALSE), 14])

x_test <- as.matrix(data[unlist(data_asli[10],use.names = FALSE),-14])
y_test <- as.matrix(data[unlist(data_asli[10],use.names = FALSE), 14])

#data sintesis
x_sintesis_train <- as.matrix(datasynth[unlist(data_sintesis[-10],use.names = FALSE),-14])
y_sintesis_train <- as.matrix(datasynth[unlist(data_sintesis[-10],use.names = FALSE), 14])

x_sintesis_test <- as.matrix(datasynth[unlist(data_sintesis[10],use.names = FALSE),-14])
y_sintesis_test <- as.matrix(datasynth[unlist(data_sintesis[10],use.names = FALSE), 14])

#--------------------NEURAL NETWORK MODEL NON SINTESIS--------------------
#MODEL 
Eksekusi <- c()
#Model
library('tensorflow')
library('keras')
# network and training
NB_EPOCH = 500
BATCH_SIZE = 1024
VERBOSE = 0
NB_CLASSES = 2   # Heart Disease diagnosis yes = 1, no = 0
N_HIDDEN = 128
TRAINING_SPLIT = 0.6 # how much from all of the data is split for training
VALIDATION_SPLIT=0.4 # how much in TRAIN is reserved for VALIDATION
DROPOUT = 0.5

model <- keras_model_sequential()
model %>%
  layer_dense(units = N_HIDDEN, input_shape = c(13), kernel_regularizer = regularizer_l2()) %>%
  layer_activation("relu") %>%
  layer_dropout(rate = DROPOUT) %>%
  layer_dense(units = N_HIDDEN) %>%
  layer_activation("relu") %>%
  layer_dropout(rate = DROPOUT) %>%
  layer_dense(units = N_HIDDEN) %>%
  layer_activation("relu") %>%
  layer_dropout(rate = DROPOUT) %>%
  layer_dense(units = N_HIDDEN) %>%
  layer_activation("relu") %>%
  layer_dropout(rate = DROPOUT) %>%
  layer_dense(units = NB_CLASSES) %>%
  layer_activation("softmax") 
summary(model)

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

model$metrics_names

start <- Sys.time()
history <- model %>% fit(
  x_train,to_categorical(y_train),
  batch_size = BATCH_SIZE, epochs = NB_EPOCH,
  verbose = VERBOSE, validation_split = VALIDATION_SPLIT
)
stop <- Sys.time()
Eksekusi <- c(Eksekusi,stop-start)

plot(history)

model %>% evaluate(x_test,to_categorical(y_test))
model %>% evaluate(x_sintesis_test,to_categorical(y_sintesis_test))

#--------------------NEURAL NETWORK MODEL SINTESIS--------------------
#MODEL 
Eksekusi <- c()
#Model
library('tensorflow')
library('keras')
# network and training
NB_EPOCH = 90
BATCH_SIZE = 1024
VERBOSE = 0
NB_CLASSES = 2   # Heart Disease diagnosis yes = 1, no = 0
N_HIDDEN = 128
TRAINING_SPLIT = 0.6 # how much from all of the data is split for training
VALIDATION_SPLIT=0.4 # how much in TRAIN is reserved for VALIDATION
DROPOUT = 0.5

model <- keras_model_sequential()
model %>%
  layer_dense(units = N_HIDDEN, input_shape = c(13), kernel_regularizer = regularizer_l2()) %>%
  layer_activation("relu") %>%
  layer_dropout(rate = DROPOUT) %>%
  layer_dense(units = N_HIDDEN) %>%
  layer_activation("relu") %>%
  layer_dropout(rate = DROPOUT) %>%
  layer_dense(units = N_HIDDEN) %>%
  layer_activation("relu") %>%
  layer_dropout(rate = DROPOUT) %>%
  layer_dense(units = N_HIDDEN) %>%
  layer_activation("relu") %>%
  layer_dropout(rate = DROPOUT) %>%
  layer_dense(units = NB_CLASSES) %>%
  layer_activation("softmax") 
summary(model)

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

model$metrics_names

start <- Sys.time()
history <- model %>% fit(
  x_sintesis_train,to_categorical(y_sintesis_train),
  batch_size = BATCH_SIZE, epochs = NB_EPOCH,
  verbose = VERBOSE, validation_split = VALIDATION_SPLIT
)
stop <- Sys.time()
Eksekusi <- c(Eksekusi,stop-start)

plot(history)

model %>% evaluate(x_sintesis_test,to_categorical(y_sintesis_test),batch_size=BATCH_SIZE)
model %>% evaluate(x_test,to_categorical(y_test))


#KELUARKAN CONFUSION MATRIX, BERLAKU UNTUK TRAINING ASLI MAUPUN SINTESIS
#confusion mat
hasil_sintesis <- model %>% predict(x_sintesis_test)
hasil_asli <- model %>% predict(x_test)

banding_sintesis <- c()
for (i in 1:nrow(hasil_sintesis)) {
  if(hasil_sintesis[i,1] >= hasil_sintesis[i,2])
  {
    banding_sintesis <- c(banding_sintesis,0)
  }
  else
  {
    banding_sintesis <- c(banding_sintesis,1)
  }
}

banding_asli <- c()
for (i in 1:nrow(hasil_asli)) {
  if(hasil_asli[i,1] >= hasil_asli[i,2])
  {
    banding_asli <- c(banding_asli,0)
  }
  else
  {
    banding_asli <- c(banding_asli,1)
  }
}

confusionmat_asli <- table(banding_asli,y_test)
confusionmat_sintesis <- table(banding_sintesis,y_sintesis_test)