#SELF ORGANIZING MAP (SOM)
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


#--------------------SOM-DATA ASLI--------------------
#Inisiasi Parameter
K <- 2 #Berapa vektor pewakil?
alpha <- 0.2
beta <- 0.3
iterasi <- 250
minalpha <- -1

#Inisiasi vektor pewakil
vektor_pewakil <- c()
for (j in 1:ncol(x_train)) {
  for (i in 1:K) {
    rand <- sample(ceiling(min(x_train[,j])):floor(max(x_train[,j])),1)
    vektor_pewakil <- c(vektor_pewakil,rand)
  }
}
vektor_pewakil <- matrix(vektor_pewakil,K,ncol(x_train))

#Fungsi euclid
euclid <- function(x,y)
{
  euclidean <- c()
  
  for (i in 1:length(x)) {
    euclidean <- c(euclidean, (x[i]-y[i])^2)
  }
  return(sqrt(sum(euclidean)))
}

#Perubahan parameter alpha
perubahan_alpha <- c(alpha)

#Cek hasil
hasil <- function(x,y)
{
  hasil <- c()
  for (i in 1:nrow(y)) {
    pewakil <- c()
    for (j in 1:nrow(x)) {
      pewakil <- c(pewakil, euclid(y[i,],x[j,]))
    }
    indeks_pewakilminimum <- which(pewakil == min(pewakil), arr.ind = TRUE)[1]
    
    hasil <- c(hasil, indeks_pewakilminimum)
  }
  return(hasil)
}

#Start time
start.time <- Sys.time()
#SOM
while (iterasi > 0) {
  for (i in 1:nrow(x_train)) {
    #Hitung jarak terhadap vektor pewakil
    pewakil <- c()
    for (j in 1:nrow(vektor_pewakil)) {
      pewakil <- c(pewakil, euclid(x_train[i,],x_train[j,]))
    }
    indeks_pewakilminimum <- which(pewakil == min(pewakil), arr.ind = TRUE)
    
    #Didekatkan
    for (j in 1:ncol(vektor_pewakil)) {
      vektor_pewakil[indeks_pewakilminimum,j] <- vektor_pewakil[indeks_pewakilminimum,j] + (alpha*(x_train[i,j] - vektor_pewakil[indeks_pewakilminimum,j])) 
    }
  }
  iterasi = iterasi - 1
  alpha <- alpha*beta
  perubahan_alpha <- c(perubahan_alpha, alpha)
  
  #ANOTHER STOPPING CONDITION
  if(alpha <= minalpha)
  {
    print("LEARNING RATE BREAK")
    return(0)
  }
}
end.time <- Sys.time()
timetaken <- end.time - start.time

#DATA TEST ASLI
hasilsom <- hasil(vektor_pewakil, x_test[,-ncol(x_test)])

confusionmat <- table(as.vector(y_test)+1,hasilsom)

akurasi <- c()
for (i in 1:K) {
  akurasi <- c(akurasi,max(confusionmat[i,]))
}
akurasi <- sum(akurasi)/nrow(x_test)

#DATA TEST SINTESIS

hasilsom_sintesis <- hasil(vektor_pewakil, x_sintesis_test)

confusionmat_sintesis <- table(as.vector(y_sintesis_test)+1,hasilsom_sintesis)

akurasi_sintesis <- c()
for (i in 1:K) {
  akurasi_sintesis <- c(akurasi_sintesis,max(confusionmat_sintesis[i,]))
}
akurasi_sintesis <- sum(akurasi_sintesis)/nrow(x_sintesis_test)

plot(perubahan_alpha,type = 'o', main = "PERUBAHAN NILAI ALPHA")
plot(x_test[,-5], col = hasilsom)
plot(x_test[,3:4], col = hasilsom)
points(vektor_pewakil[,3:4], col = 1:K, pch = 8, cex=2)


#--------------------SOM-DATA SINTESIS--------------------
#Inisiasi Parameter
K <- 2 #Berapa vektor pewakil?
alpha <- 0.5
beta <- 0.05
iterasi <- 250
minalpha <- -1

#Inisiasi vektor pewakil
vektor_pewakil <- c()
for (j in 1:ncol(x_sintesis_train)) {
  for (i in 1:K) {
    rand <- sample(ceiling(min(x_sintesis_train[,j])):floor(max(x_sintesis_train[,j])),1)
    vektor_pewakil <- c(vektor_pewakil,rand)
  }
}
vektor_pewakil <- matrix(vektor_pewakil,K,ncol(x_sintesis_train))

#Fungsi euclid
euclid <- function(x,y)
{
  euclidean <- c()
  
  for (i in 1:length(x)) {
    euclidean <- c(euclidean, (x[i]-y[i])^2)
  }
  return(sqrt(sum(euclidean)))
}

#Perubahan parameter alpha
perubahan_alpha <- c(alpha)

#Cek hasil
hasil <- function(x,y)
{
  hasil <- c()
  for (i in 1:nrow(y)) {
    pewakil <- c()
    for (j in 1:nrow(x)) {
      pewakil <- c(pewakil, euclid(y[i,],x[j,]))
    }
    indeks_pewakilminimum <- which(pewakil == min(pewakil), arr.ind = TRUE)[1]
    
    hasil <- c(hasil, indeks_pewakilminimum)
  }
  return(hasil)
}

#Start time
start.time <- Sys.time()
#SOM
while (iterasi > 0) {
  for (i in 1:nrow(x_sintesis_train)) {
    #Hitung jarak terhadap vektor pewakil
    pewakil <- c()
    for (j in 1:nrow(vektor_pewakil)) {
      pewakil <- c(pewakil, euclid(x_sintesis_train[i,],x_sintesis_train[j,]))
    }
    indeks_pewakilminimum <- which(pewakil == min(pewakil), arr.ind = TRUE)
    
    #Didekatkan
    for (j in 1:ncol(vektor_pewakil)) {
      vektor_pewakil[indeks_pewakilminimum,j] <- vektor_pewakil[indeks_pewakilminimum,j] + (alpha*(x_sintesis_train[i,j] - vektor_pewakil[indeks_pewakilminimum,j])) 
    }
  }
  iterasi = iterasi - 1
  alpha <- alpha*beta
  perubahan_alpha <- c(perubahan_alpha, alpha)
  
  #ANOTHER STOPPING CONDITION
  if(alpha <= minalpha)
  {
    print("LEARNING RATE BREAK")
    return(0)
  }
}
end.time <- Sys.time()
timetaken <- end.time - start.time

#DATA TEST SINTESIS
hasilsom <- hasil(vektor_pewakil, x_sintesis_test[,-ncol(x_sintesis_test)])

confusionmat <- table(as.vector(y_sintesis_test)+1,hasilsom)

akurasi <- c()
for (i in 1:K) {
  akurasi <- c(akurasi,max(confusionmat[i,]))
}
akurasi <- sum(akurasi)/nrow(x_sintesis_test)

#DATA TEST ASLI
hasilsom_asli <- hasil(vektor_pewakil, x_test)

confusionmat_asli <- table(as.vector(y_test)+1,hasilsom_asli)

akurasi_asli <- c()
for (i in 1:K) {
  akurasi_asli <- c(akurasi_asli,max(confusionmat_asli[i,]))
}
akurasi_asli <- sum(akurasi_asli)/nrow(x_test)

plot(perubahan_alpha,type = 'o', main = "PERUBAHAN NILAI ALPHA")
plot(x_sintesis_test[,-5], col = hasilsom)
plot(x_sintesis_test[,3:4], col = hasilsom)
points(vektor_pewakil[,3:4], col = 1:K, pch = 8, cex=2)