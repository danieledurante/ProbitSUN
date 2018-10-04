
Application to the Cancer SAGE Dataset
================
Daniele Durante

Description
-----------
As described in the [`README.md`](https://github.com/danieledurante/ProbitSUN/blob/master/README.md) file, this tutorial contains general guidelines and code to perform the analyses for the [Cancer SAGE](http://www.i3s.unice.fr/~pasquier/web/?Research_Activities___Dataset_Downloads___Cancer_SAGE) application in **Section 3** of the paper. In particular, you will find information on how to **download and clean the data**, detailed **`R` code to implement the different methods for posterior inference** discussed in Section 3 and **guidelines to reproduce Table 1 along with Figures 2 and 3** in the paper.

Upload and Clean the Cancer SAGE Dataset
--------------------------------------
As discussed in **Section 3** of the paper, the focus is on learning how gene expression (monitored at `p - 1 = 516` tags) relates to the probability of a cancerous tissue. Data are available for `n = 74` measurements and can be downloaded at [Cancer SAGE](http://www.i3s.unice.fr/~pasquier/web/?Research_Activities___Dataset_Downloads___Cancer_SAGE) by clicking [here](http://www.i3s.unice.fr/~pasquier/web/userfiles/downloads/datasets/SAGE_filtered_small_dataset.zip).
 
The download provides a directory `SAGE_filtered_small_dataset` which contains several datasets. Here the focus is on `dataset_74-516.csv`. To **clean this dataset**, first set the working directory where `dataset_74-516.csv` is placed. Once this has been done, **clean the workspace, and load the data along with useful libraries**.

``` r
rm(list=ls())
library(arm)

dataset_gene <- read.csv("dataset_74-516.csv",header=TRUE,sep="")
```

The dataframe  `dataset_gene` contains information on the **response variable** in the first column, and on the **covariates** in the remaining ones. More specifically, the first column `dataset_gene[,1]` contains names of tissues followed by a letter which is either `N` (normal) or `C` (cancerous). Exploiting this information, **let us create the response by hand**.
   
``` r
y_data <- c(0,0,1,1,1,1,1,1,0,0,1,1,1,1,1,0,1,1,1,1,1,1,0,1,1,1,1,
            1,1,1,1,1,1,0,0,1,0,0,1,1,0,1,0,0,0,1,1,0,1,1,0,0,0,0,
            0,1,1,1,1,0,0,1,1,0,1,0,1,1,1,1,1,1,1,1)
```

The design matrix comprising the covariates can be easily obtained by extracting the remaining columns in `dataset_gene`. Following [Gelman et al. (2008)](https://projecteuclid.org/euclid.aoas/1231424214), **such covariates are also rescaled and an intercept term is added**.

``` r
X_data <- apply(dataset_gene,2,rescale)
X_data <- cbind(rep(1,dim(X_data)[1]),X_data)
```
According to the discussion in Section 3 of the paper, **posterior inference** relies on `50` randomly chosen observations, whereas the remaining `24` are held-out to assess performance also in **out-of-sample classification via the posterior predictive distribution**. Let us, therefore, create these training and test sets.

``` r
set.seed(1)

#Indicators of units comprising the training set
sel_set <- sample(c(1:dim(X_data)[1]),50,replace=FALSE)

#Training data
y <- y_data[sel_set]
X <- X_data[sel_set,]

#Test data
y_new <- y_data[-sel_set]
X_new <- X_data[-sel_set,]
```
Finally, **save the relevant quantities in the file** `gene_data.RData`.

``` r
save(y,X,y_new,X_new,sel,file="gene_data.RData")
```


