# Examples

These examples show how to run AFQMC with different trial wave functions and with both

- Phaseless AFQMC
- Free projection AFQMC

The examples include mean field and coupled cluster based trial wave functions. The `staging/` subdirectory contains examples for writing trial and Hamiltonian data to disk and running from staged data, including examples for multi-GPU parallelization.

`ucisdt.py` and `ucisdtq.py` use `ccpy` to build higher-order unrestricted CI trials from CC amplitudes. Run them in a `ccpy`-capable environment (for example `ccpy_env_xing_2`).
