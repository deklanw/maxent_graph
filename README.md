Supports:

- Bipartite Configuration Model (BiCM) [1]
- Enhanced Configuration Model (ECM) [2]
- Directed Enhanced Configuration Model (DECM) [2]
- Bipartite Weighted Configuration Model (BWCM) [3]
- Bipartite Enhanced Configuration Model (BiECM) [3]
- Reciprocal Configuration Model (RCM) [5]
- Binary Configuration Model, undirected and directed (UBCM, DBCM) [5]

See the `examples` folder for Jupyter notebook examples of usage. There is an example of using BiCM to form a projection of a bipartite network of Senators and bills. And, there is an example of using ECM to filter edges from a graph of relations between Game of Thrones characters.

## Count-valued models

`maxent_graph.counts` holds a second family of null models for integer edge
weights. They are all dyad-independent with a product-form mean, so they share
one base class over an arbitrary dyad set and expose the same interface --
`fit()`, `mean()`, `var()`, `pmf()`, `cdf()`, `sf()`, `loglik()`, `sample()` --
in bipartite, undirected and directed flavours.

| | bipartite | undirected | directed | dyad distribution |
| --- | --- | --- | --- | --- |
| Poisson | `BIPCM` | `UPCM` | `DPCM` | Poisson, so `var == mean` |
| Negative binomial | `BINBCM` | `UNBCM` | `DNBCM` | `var == mean + mean^2 / r` |
| Hurdle Poisson | `BIHPCM` | `UHPCM` | `DHPCM` | Bernoulli presence, shifted Poisson weight |

```python
from maxent_graph import BIPCM, aggregate_blocks

model = BIPCM(B).fit()
model.pvalues(B.nonzero())        # P(weight >= observed) per edge
aggregate_blocks(model, row_blocks, col_blocks)
```

`BIPCM(B, exact=True)` switches to microcanonical stub matching, in which every
strength is fixed exactly rather than in expectation: the dyad marginal becomes
hypergeometric, samples are drawn by shuffling stubs, and block totals are
hypergeometric too.

`aggregate_blocks` is what makes a node-level fit answer block-level questions.
Given a partition of each side it reports, per cell, the observed and expected
totals, the variance, an enrichment ratio, a z-score, both tail probabilities
and the per-side coverage. Cell tails are exact where the family has a closed
form for them, otherwise the dyad pmfs are convolved, falling back to a normal
approximation once a cell is too large -- which for the overdispersed families
is noticeably worse, so prefer the convolution where you can afford it. It also
works on a solved BiCM or BiECM through `from_bicm` / `from_biecm`, and with a
singleton partition it reproduces the BiECM's own edge p-values.

The negative binomial models are the Gamma-Poisson mixture over the Poisson
one: the means stay at `s_i * s_a / W` and `r` measures how far dyad intensities
vary beyond Poisson noise, which turns "Poisson or geometric?" into an estimated
quantity. `fit_info` carries the evidence directly:

```python
model = BINBCM(W).fit()
model.r, model.r_std_error
model.fit_info["loglik"], model.fit_info["loglik_poisson"]
model.fit_info["overdispersion_lr"], model.fit_info["overdispersion_p"]
```

The Poisson model is the boundary of the family, so that p-value is against an
even mixture of a point mass at zero and a chi-square on one degree of freedom,
not a plain chi-square. Pass `fit(constrain_strengths=False)` for the
unrestricted maximum likelihood fit instead, whose weighted score equations
leave the means free and so do *not* reproduce the strengths.

The hurdle models' positive part is a shifted Poisson -- `w - 1 ~ Poisson(lam)`
-- matching the BiECM's shifted geometric. `positive="ztp"` gives a
zero-truncated Poisson instead, which is what `kind="zip"` implies; the shifted
form is the default because its conditional mean `1 + lam` is linear in the
rate, which makes the fit a plain Poisson fit on `w - 1` with targets
`s_i - k_i`.

Two more things worth knowing before reading the numbers:

- The hurdle models fit their positive half on the observed edges only, as the
  factorised likelihood implies, so their *joint* expected strength -- which
  sums over absent dyads too -- is below the observed one.
- On sparse networks the hurdle models' positive half can sit near a boundary
  the product form cannot reach, set by which dyads are present rather than by
  unit weights. The fit then stops on the constraint residual, so check
  `fit_info["positive_strength_error"]` against the strengths.
- The binary configuration models ignore self-loops, so the hurdle models do
  too. The Poisson and negative binomial models take `self_loops=True`.

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
