# Conjugate Bayes for Probit Regression via Unified Skew-Normal Random Variables

This repository is associated with the article [Durante (2018). *Conjugate Bayes for Probit Regression via Unified Skew-Normals*](https://arxiv.org/abs/1802.09565). The **key contribution of this paper is outlined below**.

> When the focus in on Bayesian probit regression with Gaussian priors for the coefficients, the posterior is available and belongs to the class of unified skew-normal random variables. The same is true more generally when the prior is a  unified skew-normal.

This repository provides **codes and tutorials to implement the inference methods associated with such a new result and presented in Section 2**. Here the focus is on two illustrative applications. One case-study is outlined in Section 3 of the paper, whereas the other is meant to provide further insights. More information can be found below.

- [`genes_tutorial.md`](https://github.com/danieledurante/ProbitSUN/blob/master/genes_tutorial.md). This tutorial is discussed in Section 3 of the paper and focuses on a large *p* and small *n* genomic study available at [Cancer SAGE](http://www.i3s.unice.fr/~pasquier/web/?Research_Activities___Dataset_Downloads___Cancer_SAGE). The goal is to compare the `Algorithm 1` proposed in the paper—which provides **independent and identically distributed samples from the unified skew-normal posterior**—with state-of-the-art Markov Chain Monte Carlo (MCMC) competitors. These include the **data augmentation Gibbs sampler** by [Albert and Chib (1993)](https://www.jstor.org/stable/2290350) (`R` package `bayesm`), the **Hamiltonian no u-turn sampler** by [Hoffman and Gelman (2014)](http://jmlr.org/papers/v15/hoffman14a.html) (`R` package `rstan`) and the **adaptive Metropolis-Hastings** in [Haario et al. (2001)](https://projecteuclid.org/euclid.bj/1080222083) (`R` package `LaplacesDemon`). This last algorithm is also tuned via expectation propagation estimates obtained from the `R` package `EPGLM` (version 1.1.2) which needs to be downloaded at [`CRAN Archive`](https://cran.r-project.org/src/contrib/Archive/EPGLM/) and then installed locally. 

- [`voice_tutorial.md`](https://github.com/danieledurante/ProbitSUN/blob/master/voice_tutorial.md). This tutorial implements the algorithms for posterior inference discussed above on a dataset with lower *p* and larger *n*. Specifically, the illustrative application considered here refers to a voice rehabilitation study available at [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/LSVT+Voice+Rehabilitation). As discussed in the article, when *p* decreases and *n* increases, the MCMC methods in `bayesm`, `rstan` and `LaplacesDemon`  are expected to progressively improve performance, whereas `Algorithm 1` may face more evident issues in computational time. This behavior is partially observed in this tutorial, although `Algorithm 1` is still competitive.

In [`genes_tutorial.md`](https://github.com/danieledurante/ProbitSUN/blob/master/genes_tutorial.md) and [`voice_tutorial.md`](https://github.com/danieledurante/ProbitSUN/blob/master/voice_tutorial.md), the inference performance based on sampling from the posterior is also compared with the **exact methods** proposed in Section 2.3.

All the analyses are performed with a **MacBook Pro (OS X El Capitan, version 10.11.6)**, using a `R` version 3.4.1. 
