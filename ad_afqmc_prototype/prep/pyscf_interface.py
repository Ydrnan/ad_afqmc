import jax
import jax.numpy as jnp
import numpy as np
from pyscf import ao2mo
from . import integrals

from pyscf.scf.hf import RHF
from pyscf.scf.rohf import ROHF
from pyscf.scf.uhf import UHF
from pyscf.scf.ghf import GHF

from pyscf.cc.ccsd import CCSD
from pyscf.cc.gccsd import GCCSD

def get_integrals(mf):
    if not isinstance(mf, (RHF, ROHF, GHF)):
        raise TypeError(f"Expected RHF, ROHF or GHF, but found {type(mf)}.")
    if not hasattr(mf, "mo_coeff"):
        raise ValueError(f"mo_coeff not found, you may not have run the scf kernel.")

    mol = mf.mol
    h0 = mf.energy_nuc()
    h1 = np.array(mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff)
    #eri = np.array(ao2mo.kernel(mol, mf.mo_coeff))
    #eri = ao2mo.restore(4, eri, mol.nao)
    #chol = integrals.modified_cholesky(eri, max_error=1e-6)
    chol_vec = integrals.chunked_cholesky(mol)  

    nchol = chol_vec.shape[0]
    norb = mf.mo_coeff.shape[0]
    chol = np.zeros((nchol, norb, norb))
    for i in range(nchol):
        chol_i = chol_vec[i].reshape(norb, norb)
        chol[i] = mf.mo_coeff.T.conj() @ chol_i @ mf.mo_coeff

    h0=jnp.array(h0)
#    jax.debug.print("h0 : {a}", a=h0)
    h1=jnp.array(h1)
#    jax.debug.print("h1 : {a}", a=h1)
    chol=jnp.array(chol)
#    jax.debug.print("chol : {a}", a=chol)
#    jax.debug.print("chol shape : {a}", a=len(chol))

    return h0, h1, chol

def get_cisd(cc):
    if not isinstance(cc, (CCSD, GCCSD)):
        raise TypeError(f"Expected CCSD or GCCSD, but found {type(cc)}.")
    if not hasattr(cc, "t1") or not hasattr(cc, "t2"):
        raise ValueError(f"amplitudes not found, you may not have run the cc kernel.")

    ci2 = cc.t2 + np.einsum("ia,jb->ijab", np.array(cc.t1), np.array(cc.t1))
    ci2 = ci2.transpose(0, 2, 1, 3)
    ci2 = jnp.array(ci2)
    ci1 = jnp.array(cc.t1)
    return ci1, ci2
