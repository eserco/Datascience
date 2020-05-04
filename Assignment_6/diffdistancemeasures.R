#install packages
install.packages("amap")
library(amap)


#read csv data
data <- read.csv("data3.csv")


# Compute manhattan wss for k = 25 to k = 40.
k.max <- 40
Manwss <- sapply(25:k.max, 
              function(k){Kmeans(data, k, nstart=5,iter.max = 20, method="manhattan" )})


# Compute manhattan wss for k = 25 to k = 40.
k.max <- 40
Eucwss <- sapply(25:k.max, 
              function(k){Kmeans(data, k, nstart=5,iter.max = 20, method="euclidean" )})

