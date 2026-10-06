# GORIN: Genome-Resolved Inference on MAG Network

GORIN estimates metagenome-assembled genome (MAG) abundances from amplicon sequence variant (ASV) abundances using genomic copy-number information and network regularization.

This repository contains the estimator used in the associated study and a Jupyter notebook demonstrating the complete workflow.

- `mag_abundance.py` — MAG-abundance estimator
- `estimate_mag_abundance.ipynb` — data preparation, estimation, restoration of missing time points, and export

The estimator is intentionally strict: it does not silently intersect ASVs, reorder rows, remove missing samples, subset MAGs, or modify the input data. These preparation steps are handled explicitly in the notebook.

## Associated study

**High-resolution temporal profiling reveals synchronized dynamics of the mouse gut microbiome**  
bioRxiv: https://doi.org/10.64898/2026.03.26.714232

The analysis used the following parameters:

| Parameter | Value |
| --- | ---: |
| `lambda_penalty` | `1e-8` |
| `epsilon` | `1e-4` |
| `delta` | `1e-4` |
| `max_iter` | `50` |
| `tol` | `1e-6` |

## Example data

The repository includes `copy_number_matrix.csv` and `asv_abundance.csv` as a small synthetic example for running `estimate_mag_abundance.ipynb`. Replace these files or edit the paths in the notebook to use your own data.

## Inputs

The notebook reads two CSV files whose paths are set in its first code cell.

### 1. ASV × MAG copy-number matrix

Rows are ASVs, columns are candidate MAGs, and values are nonnegative integer genomic copy numbers.

```text
        MAG1  MAG2  MAG3
ASV1       1     1     0
ASV2       0     1     0
ASV3       0     0     2
```

The columns should represent the intended **candidate MAG set**. Changing this set can change the inference when ASVs are shared among MAGs. If only selected MAG trajectories are needed for downstream analysis, subset the estimated result rather than removing candidate MAGs before fitting.

### 2. ASV × sample/time-point abundance matrix

Rows are ASVs and columns are samples or time points. Values must be nonnegative. A completely missing sample/time point may be represented by an all-`NaN` column. Partial missingness within a column is not supported.

## Workflow

The notebook makes all data preparation explicit before running the estimator:

1. Validate the two input tables.
2. Remove ASVs that map to none of the candidate MAGs.
3. Intersect ASVs between the two tables and report how many are retained.
4. Enforce identical ASV row order.
5. Verify that every candidate MAG still has at least one ASV.
6. Remove completely missing samples/time points and reject partial missingness.
7. Estimate MAG abundances with `estimate_mag_abundance()`.
8. Restore the original sample/time-point axis, leaving missing columns as `NaN`.
9. Save the MAG × sample/time-point result as CSV.

`estimate_mag_abundance()` accepts only already-prepared inputs. In particular, the ASV IDs and their row order must match exactly between the two input matrices, and no `NaN` or infinite values may be passed to the estimator.

## Method summary

GORIN uses iterative Poisson-weighted nonnegative least squares with network regularization.

For one sample/time point, let `A` be the ASV × MAG copy-number matrix, `y` the ASV-abundance vector, and `x >= 0` the MAG-abundance vector. The expected ASV abundance is `mu = A x`.

Each iteratively reweighted least squares (IRLS) step solves:

```text
x^(k+1) = argmin_{x >= 0} {
    sum_i [y_i - (A x)_i]^2 / max(mu_i^(k), delta)
    + lambda * ||R x||_2^2
}
```

Here, `lambda` corresponds to `lambda_penalty`, and the initialization is `mu^(0) = y`. After each step, `mu^(k+1) = A x^(k+1)`. The subproblem is solved using `scipy.optimize.nnls`.

The ASV–MAG bipartite graph is decomposed into connected components. For rank-deficient components, MAG pairs are weighted by the number of ASV types they share, giving a graph-Laplacian smoothing term.

If at least one rank-deficient component exists, the implementation uses Cholesky factorization in the full MAG space:

```text
R^T R = L + epsilon * I
```

`L` contains smoothing edges only for rank-deficient components. The diagonal `epsilon * I` term applies to all MAGs but does not connect otherwise independent components. If all connected components have full column rank, the regularization term is omitted.

Samples/time points are estimated independently. Sampling time, sample order, light/dark labels, and neighboring time points are not used by the estimator.

Regularization stabilizes rank-deficient systems but does not add identifying observations, so MAG-level estimates within those components should be interpreted with that assumption in mind.

Estimates are obtained numerically and may vary depending on convergence, finite-precision arithmetic, and the computing environment.

## License

See `LICENSE`.
