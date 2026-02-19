```
 █████╗ ██████╗        █████╗ ███████╗ ██████╗ ███╗   ███╗ ██████╗
██╔══██╗██╔══██╗      ██╔══██╗██╔════╝██╔═══██╗████╗ ████║██╔════╝
███████║██║  ██║█████╗███████║█████╗  ██║   ██║██╔████╔██║██║
██╔══██║██║  ██║╚════╝██╔══██║██╔══╝  ██║▄▄ ██║██║╚██╔╝██║██║
██║  ██║██████╔╝      ██║  ██║██║     ╚██████╔╝██║ ╚═╝ ██║╚██████╗
╚═╝  ╚═╝╚═════╝       ╚═╝  ╚═╝╚═╝      ╚══▀▀═╝ ╚═╝     ╚═╝ ╚═════╝
```

An end-to-end differentiable Auxiliary Field Quantum Monte Carlo (AFQMC) code based on Jax.

## Usage

The code can be installed as a package using pip:

```
  pip install .
```

For use on GPUs with CUDA, install as:

```
  pip install .[gpu]
```

CPU calculations are only parallelized with multithreading (no MPI).

This code is interfaced with pyscf for electronic integral evaluation. Examples can be found in the examples/ directory.
