#Penyiapan Data

#---------------------BACA DATA--------------------
df <- read.table("D:/processed.cleveland.data.txt", sep = ",")
names(df) <- c("age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","num")

#EKSPLORASI DATA
summary(df)
str(df)

#PRAPROSES
#Hapus data missing (hanya ada 6 dari seluruh data)
df <- df[-(which(df$ca == '?' | df$thal == '?')),]

#Membenarkan tipe data agar bisa masuk package neuralnet (harus seluruhnya)
df$sex <- as.numeric(df$sex)
df$cp <- as.numeric(df$cp)
df$fbs <- as.numeric(df$fbs)
df$restecg <- as.numeric(df$restecg)
df$exang <- as.numeric(df$exang)
df$slope <- as.numeric(df$slope)
df$num <- as.numeric(df$num)
df$ca <- as.numeric(df$ca)
df$thal <- as.numeric(df$thal)

#GABUNG CLASS
df$num <- as.character(df$num)
df$num[(which(df$num == '1' | df$num == '2' | df$num == '3' | df$num == '4'))] <- 1
df$num <- as.numeric(df$num)


#--------------------SINTESIS DATA---------------------
#Aktifkan library
library(neuralnet)
library(synthpop)
library(BBmisc)

synthstratified <- syn.strata(df, strata = 'num', minstratumsize = 100,k = 60000)

#Normalisasi
#data asli
data <- normalize(df,method="range",range=c(0,1))

#data sintesis
datasynth <- normalize(synthstratified$syn,method = "range", range = c(0,1))

write.csv(data,"D:/DataCI_asli.csv")
write.csv(datasynth,"D:/DataCI_Sintesis.csv")