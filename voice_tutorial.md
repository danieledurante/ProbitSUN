
Description of the Voice Rehabilitation Application
================
As described in the [`README.md`](https://github.com/danieledurante/ProbitSUN/blob/master/README.md) file, this tutorial contains general guidelines and code to perform the analyses for the [Voice Rehabilitation](https://archive.ics.uci.edu/ml/datasets/LSVT+Voice+Rehabilitation) application. Here, the goal is to provide additional insights compared to those discussed in Section 3 of [Durante (2018). *Conjugate Bayes for probit regression via unified skew-normals*](https://arxiv.org/abs/1802.09565) for the genomic study at [`genes_tutorial.md`](https://github.com/danieledurante/ProbitSUN/blob/master/genes_tutorial.md). Indeed, as mentioned in Section 2, when *p* decreases and *n* increases, the MCMC methods in `bayesm`, `rstan` and `LaplacesDemon`  are expected to progressively improve performance, whereas `Algorithm 1` may face more evident issues in computational time. This behavior is quantitatively assessed in a [Voice Rehabilitation](https://archive.ics.uci.edu/ml/datasets/LSVT+Voice+Rehabilitation) study which has almost doubled `n` and almost halved `p` when compared to the application in the article.

Below, you will find information on how to **download and clean the data**, detailed `R` **code to implement the different methods for posterior inference** and **guidelines to derive tables and figures** similar to those in [`genes_tutorial.md`](https://github.com/danieledurante/ProbitSUN/blob/master/genes_tutorial.md). For implementation purposes, **execute the code below considering the same order in which is presented**.

Upload and Clean the Voice Rehabilitation Dataset
================
The focus here is on learning how `p - 1 = 309` dysphonia measures relate to the probability of an acceptable phonation. Data are available for `n = 126` measurements and can be downloaded at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/LSVT+Voice+Rehabilitation) by clicking at this [link](https://archive.ics.uci.edu/ml/machine-learning-databases/00282/). For more information on this study, refer [to Tsanas et al. (2014)](https://ieeexplore.ieee.org/document/6678640).
 
The download provides a directory `LSVT_voice_rehabilitation` which contains a description `LSVT_feature_names.txt` and the dataset of interest `LSVT_voice_rehabilitation.xlsx`. To **clean this dataset**, first set the working directory where `LSVT_voice_rehabilitation.xlsx` is placed. Once this has been done, **clean the workspace**,**load useful** `R` **packages**  and **upload the predictors matrix along with the response vector**. Following [Gelman et al. (2008)](https://projecteuclid.org/euclid.aoas/1231424214), **the predictors are also rescaled and an intercept term is added**.

``` r
rm(list=ls())
library(gdata)
library(arm)

# Predictors
audio_measures <- as.matrix(read.xls("LSVT_voice_rehabilitation.xlsx",sheet=1,header=TRUE)[,-128])
X_data <- apply(audio_measures,2,rescale)
X_data <- cbind(rep(1,dim(X_data)[1]),X_data) 

# Response
y_data <- c(read.xls("LSVT_voice_rehabilitation.xlsx",sheet=2,header=TRUE)[,1])-1
```

In this application, **posterior inference** relies on `100` randomly chosen observations, whereas the remaining `26` are held-out to assess performance also in **out-of-sample classification via the posterior predictive distribution**. Let us, therefore, create these training and test sets.

``` r
set.seed(1)

# Indicators of units comprising the training set
sel_set <- sample(c(1:dim(X_data)[1]),100,replace=FALSE)

# Training data
y <- y_data[sel_set]
X <- X_data[sel_set,]

# Test data
y_new <- y_data[-sel_set]
X_new <- X_data[-sel_set,]
```
Finally, **save the relevant quantities in the file** `voice_data.RData`.

``` r
save(y,X,y_new,X_new,sel_set,file="voice_data.RData")
```
Implementation of the Different Sampling Schemes
================
This section contains codes to implement the algorithm which provides **i.i.d. samples from the unified skew-normal posterior** (`Algorithm 1` in the paper), as well as state-of-the-art Markov Chain Monte Carlo (MCMC) competitors. These include the **data augmentation Gibbs sampler** by [Albert and Chib (1993)](https://www.jstor.org/stable/2290350) (`R` package `bayesm`), the **Hamiltonian no u-turn sampler** by [Hoffman and Gelman (2014)](http://jmlr.org/papers/v15/hoffman14a.html) (`R` package `rstan`) and the **adaptive Metropolis-Hastings** in [Haario et al. (2001)](https://projecteuclid.org/euclid.bj/1080222083) (`R` package `LaplacesDemon`).

Direct sampling from the unified skew-normal posterior
------------------
This subsection implements the **i.i.d. sampler from the unified skew-normal posterior** which relies on the novel results in [Durante (2018). *Conjugate Bayes for probit regression via unified skew-normals*](https://arxiv.org/abs/1802.09565). This scheme is described in detail in Section 2.4 of the paper. A pseudo-code is also provided in `Algorithm 1`. 

To implement this routine, **let us re-start a new** `R` **session** and **set the working directory where** `voice_data.RData` **is placed**. Once this has been done, load the file `voice_data.RData` along with useful `R` packages, and set the model dimensions (`p`,`n`) together with the desired number `N_sampl` of i.i.d. samples.

``` r
rm(list=ls())
library(mvtnorm)
library(TruncatedNormal)

# Load the data
load("voice_data.RData")

# Set model dimensions
n <- dim(X)[1]
p <- dim(X)[2] 

# Number of i.i.d. samples from the posterior
N_sampl <- 20000
```
Once the above steps have been done, let us first **define the key quantities to implement** `Algorithm 1` in [Durante (2018). *Conjugate Bayes for probit regression via unified skew-normals*](https://arxiv.org/abs/1802.09565).

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
Finally, **let us implement** `Algorithm 1`. This requires calculating linear combinations of samples from *p*-variate Gaussians and *n*-variate truncated normals—using the methods in [Botev (2017)](https://rss.onlinelibrary.wiley.com/doi/10.1111/rssb.12162). Note that also the running-time is monitored in order to compare it with those of the MCMC competitors implemented in the upcoming subsections.

``` r
start_time <- Sys.time()
set.seed(123)

V_0 <- t(rmvnorm(N_sampl,mean=rep(0,p),sigma=Var_V0))
V_0_scale_plus_xi <- apply(V_0,2,function(x) xi+coef_V0%*%x)
V_1 <- mvrandn(-gamma_post,rep(Inf,n),Gamma_post,N_sampl)
V_1_scale <- apply(V_1,2,function(x) coef_V1%*%x)

beta_SUN <- V_0_scale_plus_xi+V_1_scale
end_time <- Sys.time()

time_SUN <- difftime(end_time, start_time, units=("secs"))[[1]]
```

Let us now **calculate the posterior mean of the regression coefficients and the posterior predictive probabilities for the** `24` **held-out units**. Such quantities are obtained here via Monte Carlo integration using the samples from the posterior, and will be used in the comparisons with state-of-the-art competitors (see Figures 2 and 3 in the paper).

``` r
# Posterior means via Monte Carlo
SUN_means <- apply(beta_SUN,1,mean)

# Posterior predictive probabilities via Monte Carlo
pred_SUN <- rep(0,dim(X_new)[1])
beta_SUN <- t(beta_SUN)

for (i in 1:dim(X_new)[1]){
pred_SUN[i] <- mean(pnorm((beta_SUN%*%X_new[i,]),0,1))
print(i)}
```
Finally, **save the output in the file** `SUN_output.RData`
``` r
save(time_SUN,beta_SUN,SUN_means,pred_SUN,file="SUN_output.RData")
```

Data augmentation Gibbs sampler
------------------
Let us focus now on the **data augmentation Gibbs sampler** by [Albert and Chib (1993)](https://www.jstor.org/stable/2290350) (`R` package `bayesm`). To implement this routine, **re-start again a new** `R` **session** (this is useful to ensure that each algorithm is run with fully clean memory, thus guaranteeing fair comparisons on computational time). **Set also the working directory where** `voice_data.RData` **is placed**. Once this has been done, load again the file `voice_data.RData` along with useful `R` packages, and set the model dimensions (`p`,`n`) together with the desired number `N_sampl` of MCMC samples and the requested burn-in `burn` period.

``` r
rm(list=ls())
library(bayesm)

# Load the data
load("voice_data.RData")

# Set model dimensions
n <- dim(X)[1]
p <- dim(X)[2] 

# Number of MCMC samples from the posterior and burn-in
N_sampl <- 25000
burn <- 5000
```
Note that, **differently from the i.i.d. sampler, MCMC methods do not sample from the posterior since the beginning, but require a burn-in period** to reach convergence.  Let us now **define the key quantities to implement** the data augmentation Gibbs sampler.

``` r
# Data array
Data_GIBBS = list(y=as.matrix(y,c(n,1)),X=X)

# Prior settings
Prior_GIBBS = list(betabar=matrix(rep(0,p),c(p,1)),A=diag(1/16,p,p))

# MCMC settings
Mcmc_GIBBS = list(R=N_sampl,keep=1,nprint=0)
```
Finally, **let us implement** the Gibbs sampler by [Albert and Chib (1993)](https://www.jstor.org/stable/2290350). This requires the `R` package `bayesm`. Note that also here the running-time is monitored for performance comparisons.

``` r
start_time <- Sys.time()

set.seed(123)
GIBBS_Samples <- rbprobitGibbs(Data=Data_GIBBS, Prior=Prior_GIBBS, Mcmc=Mcmc_GIBBS)

end_time <- Sys.time()

beta_GIBBS <- t(GIBBS_Samples$betadraw[(burn+1):N_sampl,])
time_GIBBS <- difftime(end_time, start_time, units=("secs"))[[1]]
```

Let us now **calculate the posterior mean of the regression coefficients and the posterior predictive probabilities for the** `24` **held-out units**. Such quantities are obtained here via Monte Carlo integration using the MCMC samples from the data augmentation Gibbs sampler.

``` r
# Posterior means via Monte Carlo
GIBBS_means <- apply(beta_GIBBS,1,mean)

# Posterior predictive probabilities via Monte Carlo
pred_GIBBS <- rep(0,dim(X_new)[1])
beta_GIBBS <- t(beta_GIBBS)

for (i in 1:dim(X_new)[1]){
pred_GIBBS[i] <- mean(pnorm((beta_GIBBS%*%X_new[i,]),0,1))
print(i)}
```
Finally, **save the output in the file** `GIBBS_output.RData`
``` r
save(time_GIBBS,beta_GIBBS,GIBBS_means,pred_GIBBS,file="GIBBS_output.RData")
```

Hamiltonian no u-turn sampler
------------------
Let us consider now the **Hamiltonian no u-turn sampler** by [Hoffman and Gelman (2014)](http://jmlr.org/papers/v15/hoffman14a.html) (`R` package `rstan`). To implement this routine, **re-start again a new** `R` **session** and **set also the working directory where** `voice_data.RData` **is placed**. Once this has been done, load the file `voice_data.RData` along with useful `R` packages, and set the model dimensions (`p`,`n`) together with the desired number `N_sampl` of MCMC samples and the requested burn-in `burn`.

``` r
rm(list=ls())
library(rstan)

# Load the data
load("voice_data.RData")

# Set model dimensions
n <- dim(X)[1]
p <- dim(X)[2] 

# Number of MCMC samples from the posterior and burn-in
N_sampl <- 25000
burn <- 5000
```
Let us now **define the key quantities to implement** the Hamiltonian no u-turn sampler.

``` r
# Model structure
probmodel<-'data{
	  int<lower=0> K;
	  int<lower=0> N;
	  int<lower=0,upper=1> Y[N];
	  matrix[N,K] X;
}
parameters {
	vector[K] beta;
}
model {
	for(i in 1:K)
	    beta[i]~normal(0,sqrt(16));
	for(n in 1:N)
		Y[n] ~ bernoulli(Phi(X[n]*beta));
}'

# Data array
data_prob <- list(N=n,K=p,Y=as.vector(y),X=X)
```
Finally, **let us implement** the Hamiltonian no u-turn sampler by [Hoffman and Gelman (2014)](http://jmlr.org/papers/v15/hoffman14a.html). This requires the `R` package `rstan`. Note that also here the running-time is monitored for performance comparisons.

``` r
HMC_Samples <- stan(model_code = probmodel, data = data_prob, iter = N_sampl,warmup=burn,chains = 1,init="0",algorithm="NUTS",seed=123)

beta_HMC <- t(extract(HMC_Samples)$beta)
time_HMC <- get_elapsed_time(HMC_Samples)[1] + get_elapsed_time(HMC_Samples)[2]
```

Let us now **calculate the posterior mean of the regression coefficients and the posterior predictive probabilities for the** `24` **held-out units**. Such quantities are obtained here via Monte Carlo integration using the MCMC samples from the Hamiltonian no u-turn sampler.

``` r
# Posterior means via Monte Carlo
HMC_means <- apply(beta_HMC,1,mean)

# Posterior predictive probabilities via Monte Carlo
pred_HMC <- rep(0,dim(X_new)[1])
beta_HMC <- t(beta_HMC)

for (i in 1:dim(X_new)[1]){
pred_HMC[i] <- mean(pnorm((beta_HMC%*%X_new[i,]),0,1))
print(i)}
```
Finally, **save the output in the file** `HMC_output.RData`
``` r
save(time_HMC,HMC_Samples,HMC_means,pred_HMC,file="HMC_output.RData")
```

Adaptive Metropolis-Hastings
------------------
Let us finally consider the **adaptive Metropolis-Hastings** by [Haario et al. (2001)](https://projecteuclid.org/euclid.bj/1080222083) (`R` package `LaplacesDemon`). To implement this routine, **re-start again a new** `R` **session** and **set also the working directory where** `voice_data.RData` **is placed**. Once this has been done, load the file `voice_data.RData` along with useful `R` packages, and set the model dimensions (`p`,`n`) together with the desired number `N_sampl` of MCMC samples and the requested burn-in `burn`.

``` r
rm(list=ls())
library(LaplacesDemon)
library(EPGLM)

# Load the data
load("voice_data.RData")

# Set model dimensions
n <- dim(X)[1]
p <- dim(X)[2] 

# Number of MCMC samples from the posterior and burn-in
N_sampl <- 25000
burn <- 5000
```
As already discussed in the [`README.md`](https://github.com/danieledurante/ProbitSUN/blob/master/README.md) file, this routine is also carefully initialized via expectation propagation estimates for the location and scale of the posterior. Such quantities can be obtained via the `R` package `EPGLM`. Let us calculate them.
``` r
set.seed(123)
EPgene <- EPprobit(X = X, Y = y, s = 16)
```
Let us now **define the key quantities to implement** the adaptive Metropolis-Hastings.

``` r
# Data structure
mon.names <- "LP"
parm.names <- as.parm.names(list(beta=rep(0,p)))
PGF <- function(Data) {
beta <- rnorm(Data$p)
return(beta)
}
MyData <- list(p=p, PGF=PGF, X=X, mon.names=mon.names, parm.names=parm.names, y=y)

# Model structure
Model <- function(parm, Data)
{
### Parameters
beta <- parm[1:Data$p]
### Log-Prior
beta.prior <- sum(dnormv(beta, 0, 16, log=TRUE))
### Log-Likelihood
mu <- tcrossprod(Data$X, t(beta))
probit_prob <- pnorm(mu)
LL <- sum(dbern(Data$y, probit_prob, log=TRUE))
### Log-Posterior
LP <- LL + beta.prior
Modelout <- list(LP=LP, Dev=-2*LL, Monitor=LP, yhat=rbern(length(probit_prob), probit_prob), parm=parm)
return(Modelout)
}
```
Finally, **let us implement** the adaptive Metropolis-Hastings by [Haario et al. (2001)](https://projecteuclid.org/euclid.bj/1080222083). This requires the `R` package `LaplacesDemon`. Note that also here the running-time is monitored for performance comparisons.

``` r
start_time <- Sys.time()

set.seed(123)
MH_Samples <- LaplacesDemon(Model,Data=MyData,Initial.Values=c(EPgene$m),Covar=(2.38^2/p)*EPgene$V,Iterations=N_sampl,Thinning=1,Algorithm="AM", Specs=list(Adaptive=burn, Periodicity=100))

end_time <- Sys.time()

beta_MH <- t(MH_Samples$Posterior1[(burn+1):N_sampl,])
time_MH <- difftime(end_time, start_time, units=("secs"))[[1]]
```

Let us now **calculate the posterior mean of the regression coefficients and the posterior predictive probabilities for the** `24` **held-out units**. Such quantities are obtained here via Monte Carlo integration using the MCMC samples from the adaptive Metropolis-Hastings.

``` r
# Posterior means via Monte Carlo
MH_means <- apply(beta_MH,1,mean)

# Posterior predictive probabilities via Monte Carlo
pred_MH <- rep(0,dim(X_new)[1])
beta_MH <- t(beta_MH)

for (i in 1:dim(X_new)[1]){
pred_MH[i] <- mean(pnorm((beta_MH%*%X_new[i,]),0,1))
print(i)}
```
Finally, **save the output in the file** `MH_output.RData`
``` r
save(time_MH,beta_MH,MH_means,pred_MH,file="MH_output.RData")
```
Implementation of Exact Inference
================
As discussed in Section 3 of the paper, besides comparing the above algorithms in terms of **computational efficiency**, there is also an interest in understanding **to what extent the Monte Carlo estimates produced by the aforementioned sampling schemes can recover those provided by the exact expressions** presented in Section 2.3 of [Durante (2018). *Conjugate Bayes for probit regression via unified skew-normals*](https://arxiv.org/abs/1802.09565). Here the focus is on the **posterior mean of the regression coefficients** and the **posterior predictive probabilities for the** `24` **held-out units**.

To address the above goal, let us **re-start again a new** `R` **session** and **set the working directory where** `voice_data.RData` **is placed**. Once this has been done, load the file `voice_data.RData` along with useful `R` packages, and set the model dimensions (`p`,`n`).

``` r
rm(list=ls())
library(mvtnorm)
library(TruncatedNormal)

# Load the data
load("voice_data.RData")

# Set model dimensions
n <- dim(X)[1]
p <- dim(X)[2] 
```
In order to compute the posterior mean of the regression coefficients (**via equation (6) in the article**) and the posterior predictive probabilities for the `24` held-out units (**via equation (7) in the article**), let us first define the required quantities.
``` r
# Relevant parameters of the Gaussian prior
Omega <- diag(16,p,p)
omega <- sqrt(diag(Omega[cbind(1:p,1:p)],p,p))
bar_Omega <- solve(omega)%*%Omega%*%solve(omega)
xi <- matrix(0,p,1)

# Relevant parameters of the SUN posterior useful for (6) and (7)
D <- diag(2*y-1,n,n)%*%X
s <- diag(sqrt((D%*%Omega%*%t(D)+diag(1,n,n))[cbind(1:n,1:n)]),n,n)
gamma_post <- solve(s)%*%D%*%xi
Gamma_post <- solve(s)%*%(D%*%Omega%*%t(D)+diag(1,n,n))%*%solve(s)
```
To compute the **posterior mean of the coefficients** via equation (6) in the article, execute the code below.
``` r
set.seed(123)

# Normalizing constant
Norm_const <- mvNcdf(l=rep(-Inf,n),u=gamma_post,Sig=Gamma_post,10^4)$prob

# Eta vector in equation (6)
eta <- matrix(0,n,1)
for (i in 1:n){
eta[i,1] <- dnorm(gamma_post[i])*mvNcdf(l=rep(-Inf,n-1),u=gamma_post[-i]-(Gamma_post[,i])[-i]*gamma_post[i],Sig=(Gamma_post[,-i])[-i,]-(Gamma_post[,i])[-i]%*%t((Gamma_post[,i])[-i]),10^4)$prob
print(i)}

# Posterior means without sampling from the SUN posterior
NUMERICAL_means <- xi+Omega%*%t(D)%*%solve(s)%*%(eta/Norm_const)
```

The **posterior predictive probabilities for the** `24` **held-out units**—calculated via equation (7) in the article—can be instead obtained from the code below.

``` r
set.seed(123)

# Normalizing constant for training data
Norm_const_obs <- mvNcdf(l=rep(-Inf,n),u=gamma_post,Sig=Gamma_post,10^4)$prob

# Vector containing the posterior predictive probabilities for the 24 units
pred_NUMERICAL <- rep(0,dim(X_new)[1])

# Calculate these posterior predictive probabilities as in (7) without sampling from the SUN posterior
for (i in 1:dim(X_new)[1]){

D_new <- rbind(D,X_new[i,])
s_new <- diag(sqrt((D_new%*%Omega%*%t(D_new)+diag(1,n+1,n+1))[cbind(1:(n+1),1:(n+1))]),n+1,n+1)
gamma_new <- solve(s_new)%*%D_new%*%xi
Gamma_new <- solve(s_new)%*%(D_new%*%Omega%*%t(D_new)+diag(1,n+1,n+1))%*%solve(s_new)	

pred_NUMERICAL[i] <- mvNcdf(l=rep(-Inf,n+1),u=gamma_new,Sig=Gamma_new,10^4)$prob/Norm_const_obs

print(i)}
```
Finally, let us **save the output in the file** `NUMERICAL_output.RData`
``` r
save(NUMERICAL_means,pred_NUMERICAL,file="NUMERICAL_output.RData")
```
Performance Assessments
================
This section concludes the analysis by providing **codes to reproduce Table 1 along with Figures 2 and 3** in Section 3 of the paper [Durante (2018). *Conjugate Bayes for probit regression via unified skew-normals*](https://arxiv.org/abs/1802.09565). More specifically, **Table 1** compares the computational performance of the sampling schemes implemented above, whereas **Figures 2 and 3** assess to what extent the Monte Carlo estimates produced by the aforementioned sampling schemes can recover those provided by the exact expressions presented in Section 2.3 of [Durante (2018). *Conjugate Bayes for probit regression via unified skew-normals*](https://arxiv.org/abs/1802.09565)—with a focus on posterior means and posterior predictive probabilities.

Before providing the codes to reproduce Table 1 along with Figures 2 and 3, **re-start again a new** `R` **session** and **set the working directory where** `voice_data.RData`, `SUN_output.RData`, `GIBBS_output.RData`, `HMC_output.RData`, `MH_output.RData` and `NUMERICAL_output.RData` **are placed**. Once this has been done, load these data files along with useful `R` packages.

``` r
rm(list=ls())

library(ggplot2)
library(coda)
library(rstan)
library(RColorBrewer)
library(MASS) 
library(reshape)
library(knitr)

# Load the data
load("voice_data.RData")

# Load the output of the sampling algorithms
load("SUN_output.RData")
load("GIBBS_output.RData")
load("HMC_output.RData")
load("MH_output.RData")

# Load the estimates obtained without sampling from the posterior
load("NUMERICAL_output.RData")
```

To **reproduce Table 1**, create an empty matrix `Table_perf` and set the total number of samples provided by the algorithms implemented above. 

``` r
N_sampl_SUN <- 20000
N_sampl <- 25000

Table_perf <- matrix(0,4,4)
rownames(Table_perf) <- c("Unified skew-normal Sampler", "Gibbs sampler", "Hamiltonian no-turn sampler", "Adaptive Metropolis-Hastings Sampler")
colnames(Table_perf) <- c("Iterations per second", "Min ESS", "Q1 ESS", "Median ESS")
```
**Note** that the three MCMC methods produce `N_sampl <- 25000` samples—since a burn-in of 5000 is required—whereas `Algorithm 1` draws directly i.i.d. from the unified skew-normal posterior, and hence, only `N_sampl_SUN <- 20000` samples are required. Let us now calculate the key quantities in **Table 1** and display it.
``` r
#----------------
# Unified skew-normal Sampler
#----------------

# Summaries for the effective sample sizes (ESS)
# (Being and independent sampler it has always ESS = N_sampl_SUN)
Table_perf[1,c(2:4)] <- N_sampl_SUN

# Iterations per second
Table_perf[1,1] <- N_sampl_SUN/time_SUN

#----------------
# Gibbs sampler
#----------------

# Summaries for the effective sample sizes (ESS)
Table_perf[2,c(2:4)] <- summary(apply(beta_GIBBS,2,effectiveSize))[1:3]

# Iterations per second
Table_perf[2,1] <- N_sampl/time_GIBBS

#----------------
# Hamiltonian no-turn sampler
#----------------

# Summaries for the effective sample sizes (ESS)
Table_perf[3,c(2:4)] <- summary((summary(HMC_Samples)$summary)[1:dim(X)[2],9])[1:3]

# Iterations per second
Table_perf[3,1] <- N_sampl/time_HMC

#----------------
# Adaptive Metropolis-Hastings Sampler
#----------------

# Summaries for the effective sample sizes (ESS)
Table_perf[4,c(2:4)] <- summary(apply(beta_MH,2,effectiveSize))[1:3]

# Iterations per second
Table_perf[4,1] <- N_sampl/time_MH


kable(Table_perf)
```
|                                     | Iterations per sec.  |     Min ESS|     Q1 ESS|  Median ESS|
|:------------------------------------|---------------------:|-----------:|----------:|-----------:|
|Unified skew-normal Sampler          |             886.64268| 20000.00000| 20000.0000| 20000.00000|
|Gibbs sampler                        |              13.47774|    55.46406|  2417.3770|  3687.17645|
|Hamiltonian no-turn sampler          |              15.95125| 20000.00000| 20000.0000| 20000.00000|
|Adap.    Metropolis-Hastings Sampler |              19.33543|    28.55497|    49.2213|    59.07417|

Refer to Section 3 in [Durante (2018). *Conjugate Bayes for probit regression via unified skew-normals*](https://arxiv.org/abs/1802.09565) for detailed comments on the above results. 

To obtain **Figure 2**, consider instead the code below.

``` r
# Unified skew-normal sampler
data_matrix_SUN_plot <- c(SUN_means-NUMERICAL_means)
data_matrix_SUN_plot <- melt(data_matrix_SUN_plot)
data_matrix_SUN_plot$method <- c("Unified skew-normal sampler")

# Gibbs sampler
data_matrix_GIBBS_plot <- c(GIBBS_means-NUMERICAL_means)
data_matrix_GIBBS_plot <- melt(data_matrix_GIBBS_plot)
data_matrix_GIBBS_plot$method <- c("Gibbs sampler")

# Hamiltonian no u-turn sampler
data_matrix_HMC_plot <- c(HMC_means-NUMERICAL_means)
data_matrix_HMC_plot <- melt(data_matrix_HMC_plot)
data_matrix_HMC_plot$method <- c("Hamiltonian no u-turn sampler")

# Adaptive Metropolis-Hastings sampler
data_matrix_MH_plot <- c(MH_means-NUMERICAL_means)
data_matrix_MH_plot <- melt(data_matrix_MH_plot)
data_matrix_MH_plot$method <- c("Adaptive Metropolis-Hastings sampler")

# Some graphical settings
data_final_plot <- rbind(data_matrix_GIBBS_plot,data_matrix_SUN_plot,data_matrix_HMC_plot,data_matrix_MH_plot)
data_final_plot$method <- factor(data_final_plot$method,levels=c("Unified skew-normal sampler","Gibbs sampler","Hamiltonian no u-turn sampler","Adaptive Metropolis-Hastings sampler"))
data_final_plot$description <- "Quality in posterior mean calculation via Monte Carlo methods"

# Figure 2
set.seed(123)
ggplot(data_final_plot, aes(x=method, y=value))+geom_boxplot()+theme_bw()+ geom_jitter(width = 0.2,alpha=0.1,size=0.5)+ylab("Error for posterior means")+xlab("Sampling scheme")+theme(axis.title.x = element_text(size=10),axis.title.y = element_text(size=10),strip.text = element_text(size=12))+ylim(-2.5,2.5)+ facet_wrap( ~ description)
```
![](https://raw.githubusercontent.com/danieledurante/probitSUN/master/img/F_moments_genes.png)

Finally, the code for **Figure 3** can be found below.

``` r
# Unified skew-normal Sampler
data_matrix_SUN_plot<-c(pred_SUN-pred_NUMERICAL)
data_matrix_SUN_plot<-melt(data_matrix_SUN_plot)
data_matrix_SUN_plot$method<-c("Unified skew-normal sampler")

# Gibbs sampler
data_matrix_GIBBS_plot<-c(pred_GIBBS-pred_NUMERICAL)
data_matrix_GIBBS_plot<-melt(data_matrix_GIBBS_plot)
data_matrix_GIBBS_plot$method<-c("Gibbs sampler")

# Hamiltonian no-turn sampler
data_matrix_HMC_plot<-c(pred_HMC-pred_NUMERICAL)
data_matrix_HMC_plot<-melt(data_matrix_HMC_plot)
data_matrix_HMC_plot$method<-c("Hamiltonian no u-turn sampler")

# Adaptive Metropolis-Hastings Sampler
data_matrix_MH_plot<-c(pred_MH-pred_NUMERICAL)
data_matrix_MH_plot<-melt(data_matrix_MH_plot)
data_matrix_MH_plot$method<-c("Adaptive Metropolis-Hastings sampler")

# Some graphical settings
data_final_plot <- rbind(data_matrix_GIBBS_plot,data_matrix_SUN_plot,data_matrix_HMC_plot,data_matrix_MH_plot)
data_final_plot$method <- factor(data_final_plot$method,levels=c("Unified skew-normal sampler","Gibbs sampler","Hamiltonian no u-turn sampler","Adaptive Metropolis-Hastings sampler"))
data_final_plot$description <- "Quality in posterior predictive probability calculation via Monte Carlo methods"

# Figure 3
set.seed(123)
ggplot(data_final_plot, aes(x=method, y=value))+geom_boxplot()+theme_bw()+ geom_jitter(width = 0.1,alpha=0.1,size=1)+ylab("Error for posterior predictive probability")+xlab("Sampling scheme")+theme(axis.title.x = element_text(size=10),axis.title.y = element_text(size=10),strip.text = element_text(size=12))+ylim(-0.4,0.4)+ facet_wrap( ~ description)
```
![](https://raw.githubusercontent.com/danieledurante/probitSUN/master/img/F_predict_genes.png)

Also in this case, refer to Section 3 in [Durante (2018). *Conjugate Bayes for probit regression via unified skew-normals*](https://arxiv.org/abs/1802.09565) for detailed comments on the above Figures. 
