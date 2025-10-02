Original_data<-read.csv("hcvdat0.csv")     ## Loading  the dataset
str(Original_data)                         ### Data structure:The data has 615 observations and 14 variables
##With a response variable having four levels namely Blood donor ,Cirrhosis,Fibrosis and Hepatitis  


class(Original_data)##forms a data frame
Original_data
dim(Original_data)## The data has 14 attributes and 615 cases 
################ LOADING RELEVANT PACKAGES
install.packages("party")
library(party)
install.packages("ROCR")
library(ROCR)
library(ggplot2)
library(rpart)
install.packages("rattle")
library(rattle)
install.packages("rpart.plot")
library(rpart.plot)
library(rpart.plot)
library(RColorBrewer)
library(dplyr)
library(plyr)
library(caTools)
library(lattice)
library(DMwR)
library(TTR)
library(xts)
library(quantmod)
library(kernlab)
library(randomForest)
library(factoextra)
library(ggvis)
install.packages("k")
library(VIM)
library(mice)
library(pysch)
library(caret)
library(caretEnsemble)
library(shiny)
library(shinydashboard)
library(ggplot2)
################################### DATA CLEANING
summary(na.omit(Original_data))
view(Original_data)
any(is.na(Original_data))#####There are NAs in the data frame
sum(is.na(Original_data))###There are 31 NAs values in our data frame original data
table(is.na(Original_data))##########Using a table to record the number of missing values i.e 31
sapply(Original_data,function(x) sum(is.na(x)))###Tabling the missing values per feature
missinf_plot<-aggr(Original_data,col=c("navyblue","yellow"),numbers=TRUE,sortVars=TRUE,labels=names(Original_data)
                   ,cex.axis=0.5,gap=3,ylab=c("Missing Data","Pattern"))##The distribution of missing values in each of the features
my_data<-Original_data
#####Converting factors (Sex and Category) into dummy codes
dmy<-dummyVars("~.",data=my_data,fullRank=T)
my_data<-data.frame(predict(dmy,newdata=my_data))
str(my_data)
##Using the predictive modelling from the mice package this method uses regression to impute missing values based on other values in the data
######Imputation of missing values :since we dealing with numeric missing values im using predictive mean matching to imput my missing values
imputed_values<-mice(my_data,m=4,maxit=20,method="pmm",seed=500)###Creates four data sets (m=4) and maxit specifies the iteration
summary(imputed_values)
selected_data<-complete(imputed_values,2)#######Selects the second data set
table(is.na(selected_data))#########Check if there are still any missing values in our data set
##################Handling of Outliers
########Cook"s distance 
mod<-glm(Original_data$Category~.
         ,data=my_data)
cooksd<-cooks.distance(mod)
Influential<-as.numeric(names(cooksd)[(cooksd>3*mean(cooksd,na.rm = T))])
final_data<-imputed_values[-influential,]####removing influential points in our data
#######Normalization of our features 
normalize<-function(x){return((x-mean(x))/sd(x))}###The Z score normalization
lap<-as.data.frame(final_data[,c(3,5:14)],normalize)
dataset<-final_data$dataset
Gender<-final_data$Sex
####Binding dataset and Gender to normalized data
lap$Gender<-Gender
lap$dataset<dataset
new_data<-na.omit(Original_data)
count(is.na(new_data))
Bilrubin<-Original_data$BIL
Age<-Original_data$Age
Cholestrol<-Original_data$CHOL
Protein<-Original_data$PROT
Sex<-Original_data$Sex
Albumin<-Original_data$ALB
AST<-Original_data$AST
ALP<-Original_data$ALP
CHE<-Original_data$CHE
Creatinine<-Original_data$CREA
GGT<-Original_data$GGT
ALT<-Original_data$ALT
sum(is.na(as.numeric(Original_data$)))
par(mfrow=c(1,1))
boxplot(GGT)




################################### EXPLORATORY DATA ANALYSIS
library(plyr)
hist(ALT)
hist(flights$dep_delay)
boxplot(flights$dep_time)
head(flights)
flights$arr_delay_delay<-flights$Arr_delay
head(flights)
BasicSummary(Original_data)
################################### PERFOMING VARIABLE SELECTION
data=Original_data[,-4]### Removing Sex a categorical variable
view(data)
numeric_data<-data[,-2]##############Removing Category
clean_data<-na.omit(numeric_data)### Removing all Nas in the data and all non numeric variables
pca<-prcomp(clean_data,center = TRUE,scale. = TRUE)
print(pca)
summary(pca)
plot(pca)
pca$
fviz_eig(pca)
featurePlot(x=data_f[,-4],y=data_or$Category,plot="ellipse")
featurePlot(x=data_f[,-4],y=data_or$Category,plot="box")
scales<-list(x=list(relation="free"),y=list(relation="free"))
featurePlot(x=data_f[,-4],y=data_or$Category,plot="density",scales=scales)
control<-trainControl(method = "cv",number = 10)
################################## MODEL FITTING
train<-readingSkills[1:105,]
test<-readingSkills[106:200,]
tree<-rpart(nativeSpeaker~age+shoeSize+score,data=train,method = "class")
fancyRpartPlot(tree)
tree<-rpart(nativeSpeaker~age+shoeSize+score,data=train,method = "class",control = rpart.control(cp=0.01))
fancyRpartPlot(tree)
prune<-prune(tree,cp=0.01)######Pruning let the tree grow to any complexity then cut off branches in a bottom up fashion
###############cp is the complexity parameter specifies the minimum reduction in error.

validation_index<-createDataPartition(data_or$Category,p=0.80,list=FALSE)
Train<-data_or[validation_index]
Smoted_data<-SMOTE(Category~.,datatrain,perc.over=1500,k=5,perc.under=900)
validation<-data_or[-validation_index,]
control<-trainControl(method = "cv",number = 10)
data$Xmetric<-"Accuracy"
set.seed(1364584)
fit.lda<-train(Category~.,data =Smoted_data,method="lda",metric=metric,trControl=control)
set.seed(1364584)
fit.cart<-train(Category~.,data=Smoted_data,method="rpart",metric=metric,trControl=control)
set.seed(1364584)
fit.knn<-train(Category~.,data=Smoted_data,method="knn",metric=metric,trControl=control)
set.seed(1364584)
fit.svm<-train(Category~.,data=Smoted_data,method="svmRadial",metric=metric,trControl=control)
set.seed(1364584)
fit.rf<-train(Category~.,data=Smoted_data,method="rf",metric=metric,trControl=control)
results<-resamples(list(lda=fit.lda,cart=fit.cart,knn=fit.knn,svm=fit.svm,rf=fit.rf))
summary(results)
dotplot(results)
print(fit.svm)
print(fit.rf)
prediction<-predict(fit.rf,validation)
confusionMatrix(prediction,validation$Category)
confusionMatrix(prediction,validation$Category)
####################Validation
