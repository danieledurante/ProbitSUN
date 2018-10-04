# Conjugate Bayes for Probit Regression via Unified Skew-Normal Random Variables

This repository is associated with the article [Durante (2018). *Conjugate Bayes for Probit Regression via Unified Skew-Normals*](https://arxiv.org/abs/1802.09565), and provides **detailed codes and tutorials to implement the inference methods presented in Section 2**. Here the focus is on two illustrative applications. One case-study is outlined in detail in Section 3 of the paper, whereas the other is meant to provide additional insights. More details can be found below.

- [`genes_tutorial.md`](https://github.com/danieledurante/ProbitSUN/blob/master/genes_tutorial.md). This tutorial is discussed in detail in Section 3 of the paper and focuses on a large *p* and small *n* genomic study available at [Cancer SAGE](http://www.i3s.unice.fr/~pasquier/web/?Research_Activities___Dataset_Downloads___Cancer_SAGE). The goal is to compare the `Algorithm 1` proposed in the paper—which provides **independent and identically distributed samples from the unified skew-normal posterior**—with state-of-the-art Markov Chain Monte Carlo (MCMC) competitors. These include the **data augmentation MCMC** by [Albert and Chib (1993)](https://www.jstor.org/stable/2290350) (`R` package `bayesm`), the **adaptive Metropolis-Hastings** from [Haario et al. (2001)](https://projecteuclid.org/euclid.bj/1080222083) (`R` package `LaplacesDemon`) and the **Hamiltonian no u-turn sampler** by [Hoffman and Gelman (2014)](http://jmlr.org/papers/v15/hoffman14a.html) (`R` package `rstan`). Note that the adaptive Metropolis-Hastings is also carefully tuned via expectation propagation estimates obtained from the `R` package `EPGLM` (version 1.1.2) which needs to be downloaded at the [`CRAN Archive`](https://cran.r-project.org/src/contrib/Archive/EPGLM/) and then installed locally. 

- [`voice_tutorial.md`](https://github.com/danieledurante/ProbitSUN/blob/master/voice_tutorial.md). This tutorial implements the algorithms for posterior inference discussed above in a dataset with lower *p* and larger *n*. Specifically, the illustrative application considered here refers to a voice rehabilitation study available at [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/LSVT+Voice+Rehabilitation). As discussed in the article, when *p* decreases and *n* increases the MCMC methods in `bayesm`, `LaplacesDemon` and `rstan` are expected to progressively improve performance, whereas `Algorithm 1` should face more evident issues in computational time. This behavior is partially observed in this tutorial, although `Algorithm 1` is still competitive.

In [`genes_tutorial.md`](https://github.com/danieledurante/ProbitSUN/blob/master/genes_tutorial.md) and [`voice_tutorial.md`](https://github.com/danieledurante/ProbitSUN/blob/master/voice_tutorial.md), the performance of posterior inference based on sampling from the posterior is also compared with the **exact methods** proposed in Section 2.3.

All the analyses are performed with a **MacBook Pro (OS X El Capitan, version 10.11.6)**, using a `R` version 3.4.1. 
