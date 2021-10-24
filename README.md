## Spectral Mixture Kernel

A minimal working example of the spectral mixture kernel introduced in:
> Wilson, A., & Adams, R. (2013). Gaussian process kernels for pattern
> discovery and extrapolation. In *International conference on machine learning*
> (pp. 1067-1075). PMLR.

This a single-file implementation of the spectral mixture kernel meant to
highlight the basic ideas. Notation is consistent with the paper.

#### Usage:
```bash
$ python script.py
```

#### Dependencies
* [PyTorch](https://pytorch.org) (>=1.9)


#### Example Output

<img src="out.pdf" width="800" align="middle" alt="SM GP output">


#### See Also
* [GPyTorch](https://github.com/cornellius-gp/gpytorch) for a better, more
  practical implementation of the spectral mixture kernel.
