#LEARNING VECTOR OPTIMIZATION (LVQ)
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
y_train <- as.matrix(data[unlist(data_asli[-10],use.names = FALSE), 14])+1

x_test <- as.matrix(data[unlist(data_asli[10],use.names = FALSE),-14])
y_test <- as.matrix(data[unlist(data_asli[10],use.names = FALSE), 14])+1

#data sintesis
x_sintesis_train <- as.matrix(datasynth[unlist(data_sintesis[-10],use.names = FALSE),-14])
y_sintesis_train <- as.matrix(datasynth[unlist(data_sintesis[-10],use.names = FALSE), 14])+1

x_sintesis_test <- as.matrix(datasynth[unlist(data_sintesis[10],use.names = FALSE),-14])
y_sintesis_test <- as.matrix(datasynth[unlist(data_sintesis[10],use.names = FALSE), 14])+1

#--------------------LVQ Data Asli-----------------------
K <- 2 #Berapa vektor pewakil?
alpha <- 0.5
beta <- 0.05
iterasi <- 250
minalpha <- -1
minerror <- -1
errorconvergence <- -1

#Inisiasi vektor pewakil
vektor_pewakil <- c()
for (j in 1:ncol(x_train)) {
  for (i in 1:K) {
    rand <- sample(ceiling(min(x_train[,j])):floor(max(x_train[,j])),1)
    vektor_pewakil <- c(vektor_pewakil,rand)
  }
}
vektor_pewakil <- matrix(vektor_pewakil,K,ncol(x_train))
print(vektor_pewakil)

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
perubahan_error <- c()

#Cek hasil & Akurasi
hasil_akurasi <- function(x,y,z)
{
  akurasi <- 0
  hasil <- c()
  for (i in 1:nrow(y)) {
    pewakil <- c()
    for (j in 1:nrow(x)) {
      pewakil <- c(pewakil, euclid(y[i,],x[j,]))
    }
    indeks_pewakilminimum <- which(pewakil == min(pewakil), arr.ind = TRUE)[1]
    
    hasil <- c(hasil, indeks_pewakilminimum)
    if(indeks_pewakilminimum == z[i])
    {
      akurasi <- akurasi + 1
    }
  }
  return(c(hasil,akurasi/nrow(y)))
}

#Start time
start.time <- Sys.time()
#LVQ
while (iterasi > 0) {
  error <- 0
  akurasi <- 0
  hasil <- c()
  for (i in 1:nrow(x_train)) {
    #Hitung jarak terhadap vektor pewakil
    pewakil <- c()
    for (j in 1:nrow(vektor_pewakil)) {
      pewakil <- c(pewakil, euclid(x_train[i,],vektor_pewakil[j,]))
    }
    indeks_pewakilminimum <- which(pewakil == min(pewakil), arr.ind = TRUE)[1]
    
    if(indeks_pewakilminimum == y_train[i])
    {
      #Didekatkan
      for (j in 1:ncol(vektor_pewakil)) {
        vektor_pewakil[indeks_pewakilminimum,j] <- vektor_pewakil[indeks_pewakilminimum,j] + (alpha*(x_train[i,j] - vektor_pewakil[indeks_pewakilminimum,j])) 
      }
    }
    else
    {
      #Dijauhkan
      for (j in 1:ncol(vektor_pewakil)) {
        vektor_pewakil[indeks_pewakilminimum,j] <- vektor_pewakil[indeks_pewakilminimum,j] - (alpha*(x_train[i,j] - vektor_pewakil[indeks_pewakilminimum,j])) 
      }
      error <- error + 1
    }
  }
  iterasi = iterasi - 1
  alpha <- alpha*beta
  perubahan_alpha <- c(perubahan_alpha, alpha)
  
  err <- error/nrow(x_train)
  perubahan_error <- c(perubahan_error,err)
  
  #ANOTHER STOPPING CONDITION
  if (err <= minerror)
  {
    print("ERROR BREAK")
    return(0)
  }
  if(alpha <= minalpha)
  {
    print("LEARNING RATE BREAK")
    return(0)
  }
  if (length(perubahan_error)>1)
  {
    if(abs((perubahan_error[length(perubahan_error)]-perubahan_error[length(perubahan_error)-1])) <= errorconvergence)
    {
      print("ERROR CONVEGANCE BREAK")
      return(0)
    }
  }
}
end.time <- Sys.time()
#Uji coba data testing
hasil_akhir <- hasil_akurasi(vektor_pewakil,x_test,y_test)
akurasi <- hasil_akhir[length(hasil_akhir)]
hasil <- hasil_akhir[-length(hasil_akhir)]
timetaken <- end.time - start.time
confusionmat <- table(y_test,hasil_akhir[-length(hasil_akhir)])

#Uji coba data testing sintesis
hasil_akhir_sintesis <- hasil_akurasi(vektor_pewakil,x_sintesis_test,y_sintesis_test)
akurasi_sintesis <- hasil_akhir_sintesis[length(hasil_akhir_sintesis)]
hasil_sintesis <- hasil_akhir_sintesis[-length(hasil_akhir_sintesis)]
confusionmat_sintesis <- table(y_sintesis_test,hasil_akhir_sintesis[-length(hasil_akhir_sintesis)])

plot(perubahan_alpha,type = 'o', main = "PERUBAHAN NILAI ALPHA")
plot(perubahan_error, type = 'o',main = "PERUBAHAN NILAI ERROR")
plot(x_test[,-5], col = hasil)
plot(x_test[,3:4], col = hasil)
points(vektor_pewakil[,1:2], col = 1:K, pch = 8, cex=2)


#--------------------LVQ Data Sintesis-----------------------
K <- 2 #Berapa vektor pewakil?
alpha <- 0.5
beta <- 0.05
iterasi <- 250
minalpha <- -1
minerror <- -1
errorconvergence <- -1

#Inisiasi vektor pewakil
vektor_pewakil <- c()
for (j in 1:ncol(x_sintesis_train)) {
  for (i in 1:K) {
    rand <- sample(ceiling(min(x_sintesis_train[,j])):floor(max(x_sintesis_train[,j])),1)
    vektor_pewakil <- c(vektor_pewakil,rand)
  }
}
vektor_pewakil <- matrix(vektor_pewakil,K,ncol(x_sintesis_train))
print(vektor_pewakil)

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
perubahan_error <- c()

#Cek hasil & Akurasi
hasil_akurasi <- function(x,y,z)
{
  akurasi <- 0
  hasil <- c()
  for (i in 1:nrow(y)) {
    pewakil <- c()
    for (j in 1:nrow(x)) {
      pewakil <- c(pewakil, euclid(y[i,],x[j,]))
    }
    indeks_pewakilminimum <- which(pewakil == min(pewakil), arr.ind = TRUE)[1]
    
    hasil <- c(hasil, indeks_pewakilminimum)
    if(indeks_pewakilminimum == z[i])
    {
      akurasi <- akurasi + 1
    }
  }
  return(c(hasil,akurasi/nrow(y)))
}

#Start time
start.time <- Sys.time()
#LVQ
while (iterasi > 0) {
  error <- 0
  akurasi <- 0
  hasil <- c()
  for (i in 1:nrow(x_sintesis_train)) {
    #Hitung jarak terhadap vektor pewakil
    pewakil <- c()
    for (j in 1:nrow(vektor_pewakil)) {
      pewakil <- c(pewakil, euclid(x_sintesis_train[i,],vektor_pewakil[j,]))
    }
    indeks_pewakilminimum <- which(pewakil == min(pewakil), arr.ind = TRUE)[1]
    
    if(indeks_pewakilminimum == y_sintesis_train[i])
    {
      #Didekatkan
      for (j in 1:ncol(vektor_pewakil)) {
        vektor_pewakil[indeks_pewakilminimum,j] <- vektor_pewakil[indeks_pewakilminimum,j] + (alpha*(x_sintesis_train[i,j] - vektor_pewakil[indeks_pewakilminimum,j])) 
      }
    }
    else
    {
      #Dijauhkan
      for (j in 1:ncol(vektor_pewakil)) {
        vektor_pewakil[indeks_pewakilminimum,j] <- vektor_pewakil[indeks_pewakilminimum,j] - (alpha*(x_sintesis_train[i,j] - vektor_pewakil[indeks_pewakilminimum,j])) 
      }
      error <- error + 1
    }
  }
  iterasi = iterasi - 1
  alpha <- alpha*beta
  perubahan_alpha <- c(perubahan_alpha, alpha)
  
  err <- error/nrow(x_sintesis_train)
  perubahan_error <- c(perubahan_error,err)
  
  #ANOTHER STOPPING CONDITION
  if (err <= minerror)
  {
    print("ERROR BREAK")
    return(0)
  }
  if(alpha <= minalpha)
  {
    print("LEARNING RATE BREAK")
    return(0)
  }
  if (length(perubahan_error)>1)
  {
    if(abs((perubahan_error[length(perubahan_error)]-perubahan_error[length(perubahan_error)-1])) <= errorconvergence)
    {
      print("ERROR CONVEGANCE BREAK")
      return(0)
    }
  }
}
end.time <- Sys.time()
#Uji coba data testing sintesis
hasil_akhir <- hasil_akurasi(vektor_pewakil,x_sintesis_test,y_sintesis_test)
confusionmat <- table(hasil_akhir[-length(hasil_akhir)],y_sintesis_test)
akurasi <- hasil_akhir[length(hasil_akhir)]
hasil <- hasil_akhir[-length(hasil_akhir)]
timetaken <- end.time - start.time

#Ujicoba data testing non sintesis
hasil_akhir_asli <- hasil_akurasi(vektor_pewakil,x_test,y_test)
confusionmat_asli <- table(hasil_akhir_asli[-length(hasil_akhir_asli)],y_test)
akurasi_asli <- hasil_akhir_asli[length(hasil_akhir_asli)]
hasil_asli <- hasil_akhir_asli[-length(hasil_akhir_asli)]

plot(perubahan_alpha,type = 'o', main = "PERUBAHAN NILAI ALPHA")
plot(perubahan_error, type = 'o',main = "PERUBAHAN NILAI ERROR")
plot(x_test[,-5], col = hasil)
plot(x_test[,3:4], col = hasil)
points(vektor_pewakil[,1:2], col = 1:K, pch = 8, cex=2)
