from pyscf import gto, scf

from trot.afqmc import Afqmc

mol = gto.M(
    atom="""
    O        0.0000000000      0.0000000000      0.0000000000
    H        0.9562300000      0.0000000000      0.0000000000
    H       -0.2353791634      0.9268076728      0.0000000000
    """,
    basis="6-31g",
    verbose=3,
)

mf = scf.RHF(mol)
mf.kernel()

af = Afqmc(mf)
mean, err = af.kernel()
