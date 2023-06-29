#part 1

library(readxl)
library(dplyr)
library(fpc)
library(MASS)
library(ggplot2)
library(ggcorrplot)
library(caret)
library(flexclust)
library(factoextra)
library(NbClust)
library(caret)
library(ggfortify)
library(FactoMineR)
library(readxl)
library(cluster)

# Read the XLSX file
vehicles_data <- read_xlsx("/Users/sandaru/Desktop/ML/CW/ML1_CW/vehicles.xlsx")

#Output variable removed (class attribute)
vehicles_data <- vehicles_data[, 2:19]
boxplot(vehicles_data, main = "Before Outlier Removal", outcol="red")
View(vehicles_data)

#Create a function to eliminate outliers from a single column using the boxplot method
remove_outliers <- function(x) {
  bp <- boxplot.stats(x)$stats
  x[x < bp[1] | x > bp[5]] <- NA
  return(x)
}

#Apply the function to each data frame column
vehicles_data <- apply(vehicles_data, 2, remove_outliers)

#Eliminate any rows with empty values (remove)
vehicles_data <- na.omit(vehicles_data)
boxplot(vehicles_data, main = "After Outlier Removal", outcol="red")

#Data scaling 
scaled_vehicles_data <- scale(vehicles_data)
boxplot(scaled_vehicles_data, main = "Scale the data set")
head(scaled_vehicles_data)

set.seed(1234)
NBcluster <- NbClust(scaled_vehicles_data, min.nc = 2,max.nc = 10, method = "kmeans")
table(NBcluster$Best.n[1,])


#--------------------------------------------------------------------------------------------------------------
#elbow_method
fviz_nbclust(scaled_vehicles_data,kmeans,method = "wss")

#silhouette_method
fviz_nbclust(scaled_vehicles_data,kmeans,method = "silhouette")

#gap_static_method
fviz_nbclust(scaled_vehicles_data,kmeans,method = "gap_stat")

#--------------------------------------------------------------------------------------------------------------
# kmean %2 clusters
k2 <-kmeans(scaled_vehicles_data, 2)
k2
autoplot(k2,scaled_vehicles_data,frame=TRUE)

#Extract relevant data when k = 2.
# Cluster centers
clus_center <- k2$centers

#clustered outcomes
clus_assi <- k2$cluster 

#Calculation BSS over TSS when k = 2
BSSk2 <- k2$betweenss #BSS
WSSk2 <- k2$tot.withinss #WSS
TSSk2 <- BSSk2 + WSSk2 #TSS

# BSS/TSS ratio
BSS_TSS_ratiok2 <- BSSk2 / TSSk2 

#Explained variance as a percentage
percent_vark2 <- round(BSS_TSS_ratiok2 * 100, 3) 

#Output results
cat("Cluster centers:\n", clus_center, "\n\n")
cat("Cluster assignments:\n", clus_assi, "\n\n")
cat("BSS/TSS ratio: ", round(BSS_TSS_ratiok2, 3), "\n\n")
cat("BSS: ", round(BSSk2, 3), "\n\n")
cat("WSS: ", round(WSSk2, 3), "\n\n")
cat("Explained variance as a percentage: ", percent_vark2, "%\n")

#3 clusters 
k3 <-kmeans(scaled_vehicles_data, 3)
k3
autoplot(k3,scaled_vehicles_data,frame=TRUE)

#Extract relevant data when k = 3
# Cluster centers
clus_center <- k3$centers

#clustered outcomes
clus_assi <- k3$cluster 

#Calculation BSS over TSS when k = 3
BSSk3 <- k3$betweenss #BSS
WSSk3 <- k3$tot.withinss #WSS
TSSk3 <- BSSk3 + WSSk3 #TSS

# BSS/TSS ratio
BSS_TSS_ratiok3 <- BSSk3 / TSSk3 

#Explained variance as a percentage
percent_vark3 <- round(BSS_TSS_ratiok3 * 100, 3) 

# Output results
cat("Cluster centers:\n", clus_center, "\n\n")
#cat("Cluster assignments:\n", cluster_assignments, "\n\n")
cat("Cluster assignments:\n", clus_assi, "\n\n")
cat("BSS/TSS ratio: ", round(BSS_TSS_ratiok3, 3), "\n\n")
cat("BSS: ", round(BSSk3, 3), "\n\n")
cat("WSS: ", round(WSSk3, 3), "\n\n")
cat("Explained variance as a percentage: ", percent_vark3,"%\n")

#Fit k-means model with k=2
k <- 2
kmeans_model <- kmeans(scaled_vehicles_data, centers = k, nstart = 25)

#Create a silhouette plot
silhouette_plot <- silhouette(kmeans_model$cluster, dist(scaled_vehicles_data))

#Compute the average silhouette's width
avg_sil_width <- mean(silhouette_plot[, 3])

#Plot the silhouette plot
plot(silhouette_plot, main = paste0("Silhouette Plot for k =", k),
     xlab = "Silhouette Width", ylab = "Cluster", border = NA)

#As a vertical line, add the average silhouette width
abline(v = avg_sil_width, lty = 2, lwd =2,col="red")

#Fit k-means model with k=3
k <- 3
kmeans_model <- kmeans(scaled_vehicles_data, centers = k, nstart = 25)

#Generate silhouette plot
silhouette_plot <- silhouette(kmeans_model$cluster, dist(scaled_vehicles_data))

#Calculate average silhouette width
avg_sil_width <- mean(silhouette_plot[, 3])

#Plot the silhouette plot
plot(silhouette_plot, main = paste0("Silhouette Plot for k =", k),
     xlab = "Silhouette Width", ylab = "Cluster", border = NA)

#Add average silhouette width as vertical line
abline(v = avg_sil_width, lty = 2, lwd =2,col="red")

#-----------------------------------------------------------------------------------------------------------------------------------
#part_02 
pca <- prcomp(scaled_vehicles_data)

#The eigenvalues and eigenvectors in print
print(summary(pca))

#Calculate the total score for each of the principal components (PC)
pca_var <- pca$sdev^2
pca_var_prop <- pca_var / sum(pca_var)
pca_var_cumprop <- cumsum(pca_var_prop)

#Plot cumulative score per PC
plot(pca_var_cumprop, xlab = "Principal Component", ylab = "Cumulative Proportion of Variance Explained",
     ylim = c(0, 1), type = "b")

#Convert an existing dataset and add attributes that correspond to the primary components
pca_trans <- predict(pca, newdata = scaled_vehicles_data)

#Choose PCs that provide at least cumulative score > 92%
selected_pcs <- which(pca_var_cumprop > 0.92)
transformed_data <- pca_trans[, selected_pcs]

boxplot(transformed_data)

set.seed(1234)
NBcluster <- NbClust(transformed_data, min.nc = 2,max.nc = 10, method = "kmeans")
table(NBcluster$Best.n[1,])


#---------------------------------------------------------------------------------------------------------------------------------
#elbow_method
fviz_nbclust(transformed_data,kmeans,method = "wss")

#silhouette_method
fviz_nbclust(transformed_data,kmeans,method = "silhouette")

#gap static_method
fviz_nbclust(transformed_data,kmeans,method = "gap_stat")

#Perform k-means clustering with k=2
set.seed(1234)
kmeans_model_2 <- kmeans(transformed_data, centers = 2, nstart = 25)

#The k-means output in print
print(kmeans_model_2)
autoplot(kmeans_model_2,transformed_data,frame=TRUE)

#Calculate the within-cluster sum of squares (WSS)
wss_2 <- sum(kmeans_model_2$withinss)

#Calculate the between-cluster sum of squares (BSS)
bss_2 <- sum(kmeans_model_2$size * dist(rbind(kmeans_model_2$centers, colMeans(transformed_data)))^2)

#Calculate the total sum of squares (TSS)
tss_2 <- sum(dist(transformed_data)^2)

#Calculate the ratio of BSS to TSS
bss_tss_ratio_2 <- bss_2 / tss_2

#Print the WSS, BSS, TSS, and ratio of BSS to TSS
cat("For k=2:\n")
cat("Sum of squares within a cluster (WSS): ", wss_2, "\n")
cat("Between-cluster square sum (BSS): ", bss_2, "\n")
cat("sum of all squares (TSS): ", tss_2, "\n")
cat("Ratio of BSS to TSS: ", bss_tss_ratio_2, "\n\n")

#Perform k-means clustering with k=3
set.seed(1234)
kmeans_model_3 <- kmeans(transformed_data, centers = 3, nstart = 25)

#The k-means output in print
print(kmeans_model_3)
autoplot(kmeans_model_3,transformed_data,frame=TRUE)

#Calculate the squares' within-cluster sum(WSS)
wss_3 <- sum(kmeans_model_3$withinss)

#Calculate the squares' between-cluster sum (BSS)
bss_3 <- sum(kmeans_model_3$size * dist(rbind(kmeans_model_3$centers, colMeans(transformed_data)))^2)

#Calculate the sum of all the squares (TSS)
tss_3 <- sum(dist(transformed_data)^2)

#Calculate the BSS to TSS ratio.
bss_tss_ratio_3 <- bss_3 / tss_3

#Print out the WSS, BSS, TSS, and the BSS/TSS ratio
cat("For k=3:\n")
cat("Sum of squares within a cluster (WSS): ", wss_3, "\n")
cat("Between-cluster square sum (BSS): ", bss_3, "\n")
cat("sum of all squares (TSS): ", tss_3, "\n")
cat("Ratio of BSS to TSS: ", bss_tss_ratio_3, "\n")


#Fit k-means model with k=2
k <- 2
kmeans_Model <- kmeans(transformed_data, centers = k, nstart = 25)

#Create a silhouette plot
silhouette_plot <- silhouette(kmeans_Model$cluster, dist(transformed_data))

#Calculate average silhouette width
avg_sil_width <- mean(silhouette_plot[, 3])

#Plot the silhouette plot
plot(silhouette_plot, main = paste0("Silhouette Plot for k =", k),
     xlab = "Silhouette Width", ylab = "Cluster", border = NA)

#As a vertical line, add the average silhouette width
abline(v = avg_sil_width, lty = 2, lwd =2,col="red")

# Fit k-means model with k=3
k <- 3
kmeans_model <- kmeans(transformed_data, centers = k, nstart = 25)

#Create a silhouette plot
silhouette_plot <- silhouette(kmeans_model$cluster, dist(transformed_data))

#Calculate average silhouette width
avg_sil_width <- mean(silhouette_plot[, 3])

#Plot the silhouette plot
plot(silhouette_plot, main = paste0("Silhouette Plot for k =", k),
     xlab = "Silhouette Width", ylab = "Cluster", border = NA)

#Add average silhouette width as vertical line
abline(v = avg_sil_width, lty = 2, lwd =2,col="red")

#Calculate Calinski-Harabasz Index k =2 
calinski_harabasz_pca <- function(cluster_result, data) {
  k2 <- length(unique(cluster_result$cluster))
  n2 <- nrow(data)
  BSS2 <- cluster_result$betweenss
  WSS2 <- cluster_result$tot.withinss
  
  ch_index2 <- ((n2 - k2) / (k2 - 1)) * (BSS2 / WSS2)
  return(ch_index2)
}

ch_index_pca_2 <- calinski_harabasz_pca(kmeans_Model, transformed_data)
ch_index_pca_2


#Calculate Calinski-Harabasz Index k =3 
calinski_harabasz_pca <- function(cluster_result, data) {
  k3 <- length(unique(cluster_result$cluster))
  n3 <- nrow(data)
  BSS3 <- cluster_result$betweenss
  WSS3 <- cluster_result$tot.withinss
  
  ch_index3 <- ((n3 - k3) / (k3 - 1)) * (BSS3 / WSS3)
  return(ch_index3)
}

ch_index_pca_3 <- calinski_harabasz_pca(kmeans_model, transformed_data)
ch_index_pca_3