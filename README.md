Supports:

- Bipartite Configuration Model (BiCM) [1]
- Enhanced Configuration Model (ECM) [2]
- Directed Enhanced Configuration Model (DECM) [2]
- Bipartite Weighted Configuration Model (BWCM) [3]
- Bipartite Enhanced Configuration Model (BiECM) [3]
- Reciprocal Configuration Model (RCM) [5]

See the `examples` folder for Jupyter notebook examples of usage. There is an example of using BiCM to form a projection of a bipartite network of Senators and bills. And, there is an example of using ECM to filter edges from a graph of relations between Game of Thrones characters.

The implementation of the pmf for the Poisson-Binomial distribution (used for assessing the statistical significance of the presence of V-motifs in the BiCM) is based on [4]

Todo:

- Parallelize further?
- Use degree multiplicity to speed up dc_fft using a fast binomial pmf?
- Make poibin selection more flexible
- Multiple comparison correction
- Suppress or work around some warnings

My implementation of the BiCM was inspired by https://github.com/mat701/BiCM and https://github.com/tsakim/bicm. See also: https://github.com/nicoloval/NEMtropy. One of these might meet your needs better.

---
[1] Saracco, Fabio, Mika J Straka, Riccardo Di Clemente, Andrea Gabrielli, Guido Caldarelli, and Tiziano Squartini. “Inferring Monopartite Projections of Bipartite Networks: An Entropy-Based Approach.” New Journal of Physics 19, no. 5 (May 17, 2017): 053022. https://doi.org/10.1088/1367-2630/aa6b38.

[2] Gemmetto, Valerio, Alessio Cardillo, and Diego Garlaschelli. “Irreducible Network Backbones: Unbiased Graph Filtering via Maximum Entropy.” ArXiv:1706.00230 [Physics], June 9, 2017. http://arxiv.org/abs/1706.00230.

[3] Di Gangi, Domenico, Fabrizio Lillo, and Davide Pirino. “Assessing Systemic Risk Due to Fire Sales Spillover through Maximum Entropy Network Reconstruction.” Journal of Economic Dynamics and Control 94 (September 2018): 117–41. https://doi.org/10.1016/j.jedc.2018.07.001.

[4] Biscarri, William, Sihai Dave Zhao, and Robert J. Brunner. “A Simple and Fast Method for Computing the Poisson Binomial Distribution Function.” Computational Statistics & Data Analysis 122 (June 2018): 92–100. https://doi.org/10.1016/j.csda.2018.01.007.

[5] Squartini, Tiziano, and Diego Garlaschelli. “Analytical Maximum-Likelihood Method to Detect Patterns in Real Networks.” New Journal of Physics 13, no. 8 (August 3, 2011): 083001. https://doi.org/10.1088/1367-2630/13/8/083001.
