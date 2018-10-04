
Description of the Cancer SAGE Application
================
As described in the [`README.md`](https://github.com/danieledurante/ProbitSUN/blob/master/README.md) file, this tutorial contains general guidelines and code to perform the analyses for the [Cancer SAGE](http://www.i3s.unice.fr/~pasquier/web/?Research_Activities___Dataset_Downloads___Cancer_SAGE) application in **Section 3** of the paper. In particular, you will find information on how to **download and clean the data**, detailed **`R` code to implement the different methods for posterior inference** discussed in Section 3 and **guidelines to reproduce Table 1 along with Figures 2 and 3** in the paper.

Upload and Clean the Cancer SAGE Dataset
================
As discussed in **Section 3** of the paper, the focus is on learning how gene expression (monitored at `p - 1 = 516` tags) relates to the probability of a cancerous tissue. Data are available for `n = 74` measurements and can be downloaded at [Cancer SAGE](http://www.i3s.unice.fr/~pasquier/web/?Research_Activities___Dataset_Downloads___Cancer_SAGE) by clicking [here](http://www.i3s.unice.fr/~pasquier/web/userfiles/downloads/datasets/SAGE_filtered_small_dataset.zip).
 
The download provides a directory `SAGE_filtered_small_dataset` which contains several datasets. Here the focus is on `dataset_74-516.csv`. To **clean this dataset**, first set the working directory where `dataset_74-516.csv` is placed. Once this has been done, **clean the workspace, and load the data along with useful `R` packages**.

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

# Indicators of units comprising the training set
sel_set <- sample(c(1:dim(X_data)[1]),50,replace=FALSE)

# Training data
y <- y_data[sel_set]
X <- X_data[sel_set,]

# Test data
y_new <- y_data[-sel_set]
X_new <- X_data[-sel_set,]
```
Finally, **save the relevant quantities in the file** `gene_data.RData`.

``` r
save(y,X,y_new,X_new,sel,file="gene_data.RData")
```
Implementation of the Different Sampling Schemes
================
This section contains codes to implement the algorithm which provides **i.i.d. samples from the unified skew-normal posterior** (`Algorithm 1` in the paper), as well as state-of-the-art Markov Chain Monte Carlo (MCMC) competitors. These include the **data augmentation MCMC** by [Albert and Chib (1993)](https://www.jstor.org/stable/2290350) (`R` package `bayesm`), the **adaptive Metropolis-Hastings** from [Haario et al. (2001)](https://projecteuclid.org/euclid.bj/1080222083) (`R` package `LaplacesDemon`) and the **Hamiltonian no u-turn sampler** by [Hoffman and Gelman (2014)](http://jmlr.org/papers/v15/hoffman14a.html) (`R` package `rstan`).

i.i.d. sampling from the unified skew-normal posterior
------------------
This subsection implements the **i.i.d. sampler from the unified skew-normal posterior** which relies on the novel results in [Durante (2018). *Conjugate Bayes for probit regression via unified skew-normals*](https://arxiv.org/abs/1802.09565). This scheme is described in detail in Section 2.4 of the paper. A pseudo-code is also provided in `Algorithm 1`. 

To implement this routine, **let us re-start a new `R` session and set the working directory where `gene_data.RData` is placed**. Once this has been done, load the file `gene_data.RData` along with useful `R` packages, and set the model dimensions togheter with the number of i.i.d. samples to draw.

``` r
rm(list=ls())
library(mvtnorm)
library(ggplot2)
library(coda)
library(TruncatedNormal)
library(arm)

# Load the data
load("gene_data.RData")

# Set model dimensions
n <- dim(X)[1]
p <- dim(X_data)[2] 

# Number of i.i.d. samples from the posterior
N_sampl <- 20000
```
Once the above steps have been done, let us first **define the key quantities to implement `Algorithm 1`** in [Durante (2018). *Conjugate Bayes for probit regression via unified skew-normals*](https://arxiv.org/abs/1802.09565).

``` r
# Relevant parameters of the Gaussian prior
Omega <- diag(16,p,p)
omega <- sqrt(diag(Omega[cbind(1:p,1:p)],p,p))
bar_Omega <- solve(omega)%*%Omega%*%solve(omega)
xi <- matrix(0,p,1)

# Relevant parameters of the SUN posterior useful for sampling
D <- diag(2*y-1,n,n)%*%X
s <- diag(sqrt((D%*%Omega%*%t(D)+diag(1,n,n))[cbind(1:n,1:n)]),n,n)
gamma_post <- solve(s)%*%D%*%xi
Gamma_post <- solve(s)%*%(D%*%Omega%*%t(D)+diag(1,n,n))%*%solve(s)

# Other useful quantities for sampling
coef_V1 <- omega%*%bar_Omega%*%omega%*%t(D)%*%solve(D%*%Omega%*%t(D)+diag(1,n,n))%*%s
coef_V0 <- omega

Var_V0 <- bar_Omega-bar_Omega%*%omega%*%t(D)%*%solve(D%*%Omega%*%t(D)+diag(1,n,n))%*%D%*%omega%*%bar_Omega
Var_V0 <- 0.5*(Var_V0+t(Var_V0))
```
Finally **let us implement `Algorithm 1`**. This requires calculating linear combinations of samples from *p*-variate Gaussians and *n*-variate truncated normals (using the methods in [Botev (2017)](https://rss.onlinelibrary.wiley.com/doi/10.1111/rssb.12162)). Note that also the running-time is monitored in order to compare it with those of the MCMC competitors implemented in the upcoming subsections.

``` r
time_SUN <- system.time({
set.seed(123)

V_0 <- t(rmvnorm(N_sampl,mean=rep(0,p),sigma=Var_V0))
V_0_scale_plus_xi <- apply(V_0,2,function(x) xi+coef_V0%*%x)

V_1 <- mvrandn(-gamma_post,rep(Inf,n),Gamma_post,N_sampl)
V_1_scale <- apply(V_1,2,function(x) coef_V1%*%x)

beta_SUN <- V_0_scale_plus_xi+V_1_scale

})
```

Let us finally **calculate the posterior mean of the regression coefficients and the posterior predictive probabilities for the `24` held-out units**. The quantities are obtained here via Monte Carlo integration using the samples of the posterior and will be used in the performance comparisons with state-of-the-art competitors (see Figures 2 and 3 in the paper).

``` r
# Posterior means via Monte Carlo
SUN_means<-apply(beta_SUN,1,mean)

# Posterior predictive probabilities via Monte Carlo
pred_SUN <- rep(0,dim(X_new)[1])
beta_SUN <- t(beta_SUN)
for (i in 1:dim(X_new)[1]){
pred_SUN[i] <- mean(pnorm((beta_SUN%*%X_new[i,]),0,1))
print(i)}
```
Finally let us save the output in the file `SUN_output.RData`
``` r
save(time_SUN,beta_SUN,SUN_means,pred_SUN,file="SUN_output.RData")
```
