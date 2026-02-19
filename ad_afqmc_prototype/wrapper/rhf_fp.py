import numpy as np
import jax
import jax.numpy as jnp

from .. import driver, config
from ..prep import integrals
from ..core.system import System
from ..ham.chol import HamChol
from ..meas.rhf import make_rhf_meas_ops
from ..prop.afqmc_fp import make_prop_ops_fp
from ..prop.blocks import block_fp
from ..prop.types import QmcParams
from ..trial.rhf import RhfTrial, make_rhf_trial_ops
from ..prep.pyscf_interface import get_integrals

class Rhf_fp:
    def __init__(self, mf):
        config.setup_jax()

        mol = mf.mol
        h0, h1, chol = get_integrals(mf)

        sys = System(norb=mol.nao, nelec=mol.nelec, walker_kind="restricted")
        ham_data = HamChol(h0, h1, chol)
        self.trial_data = RhfTrial(jnp.eye(mol.nao, mol.nelectron // 2))
        self.trial_ops = make_rhf_trial_ops(sys=sys)
        self.meas_ops = make_rhf_meas_ops(sys=sys)
        self.prop_ops = make_prop_ops_fp(ham_data, sys.walker_kind, sys=sys)
        self.params = QmcParams(
            n_eql_blocks=0, n_ene_blocks=100,n_blocks=51, n_prop_steps=40,dt=0.005, ene0 = mf.e_tot,n_walkers=200, seed=10)#np.random.randint(0, int(1e6))
        #)
        self.block_fn = block_fp
        self.sys = sys
        self.ham_data = ham_data

    def kernel(self):
        return driver.run_qmc_energy_fp(
            sys=self.sys,
            params=self.params,
            ham_data=self.ham_data,
            trial_ops=self.trial_ops,
            trial_data=self.trial_data,
            meas_ops=self.meas_ops,
            prop_ops=self.prop_ops,
            block_fn=self.block_fn,
        )
