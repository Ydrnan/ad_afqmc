import jax.numpy as jnp
import numpy as np
from ad_afqmc import wavefunctions, sampling, propagation
import pickle

from jax import config
config.update("jax_enable_x64", True)

class Block():
    def __init__(self, ham, ham_data, propagator, prop_data, trial, wave_data, sampler):
        self.ham = ham
        self.ham_data = ham_data
        self.propagator = propagator
        self.prop_data = prop_data
        self.trial = trial
        self.wave_data = wave_data
        self.sampler = sampler

def from_chkpt(fname):

    with open(fname, "rb") as f:
        ham, ham_data, propagator, prop_data, trial, wave_data, sampler = pickle.load(f)

    b = Block(ham, ham_data, propagator, prop_data, trial, wave_data, sampler)

    return b

def print_diff(a, b):
    n_a = np.linalg.norm(a)
    n_b = np.linalg.norm(b)
    diff = np.linalg.norm(np.abs(n_a)-np.abs(n_b))
    err = diff / n_b
    print(f"diff={diff:.2e}, err={err:.2e}")

def read_block(n_block, n_chk):
    # 1
    print("\n### "+str(n_chk)+" ###")

    b1_uhf = from_chkpt(uhf_path+"chk"+str(n_chk)+"_"+str(n_block)+".pkl")
    b1_rghf = from_chkpt(rghf_path+"chk"+str(n_chk)+"_"+str(n_block)+".pkl")
    b1_cghf = from_chkpt(cghf_path+"chk"+str(n_chk)+"_"+str(n_block)+".pkl")

    ## Walkers
    print("\n### Walkers ###")
    print_diff(b1_cghf.prop_data["walkers"], b1_rghf.prop_data["walkers"])

    ## Overlap
    print("\n### Overlap ###")
    uhf_o = np.asarray([
        b1_uhf.trial._calc_overlap(walker_up, walker_dn, b1_uhf.wave_data) for walker_up, walker_dn in zip(b1_uhf.prop_data["walkers"][0], b1_uhf.prop_data["walkers"][1])
    ])

    rghf_o = np.asarray([
        b1_rghf.trial._calc_overlap_restricted(walker, b1_rghf.wave_data) for walker in b1_rghf.prop_data["walkers"]
    ])

    cghf_o = np.asarray([
        b1_cghf.trial._calc_overlap_restricted(walker, b1_cghf.wave_data) for walker in b1_cghf.prop_data["walkers"]
    ])

    print_diff(rghf_o, uhf_o)
    print_diff(cghf_o, rghf_o)

    ## Energy
    print("\n### Energy ###")
    uhf_e = np.asarray([
        b1_uhf.trial._calc_energy(walker_up, walker_dn, b1_uhf.ham_data, b1_uhf.wave_data) for walker_up, walker_dn in zip(b1_uhf.prop_data["walkers"][0], b1_uhf.prop_data["walkers"][1])
    ])

    rghf_e = np.asarray([
        b1_rghf.trial._calc_energy_restricted(walker, b1_rghf.ham_data, b1_rghf.wave_data) for walker in b1_rghf.prop_data["walkers"]
    ])
    
    cghf_e = np.asarray([
        b1_cghf.trial._calc_energy_restricted(walker, b1_cghf.ham_data, b1_cghf.wave_data) for walker in b1_cghf.prop_data["walkers"]
    ])
    
    print_diff(rghf_e, uhf_e)
    print_diff(cghf_e, rghf_e)

    ## Force bias
    print("\n### Force bias ###")
    uhf_fb = np.asarray([
        b1_uhf.trial._calc_force_bias(walker_up, walker_dn, b1_uhf.ham_data, b1_uhf.wave_data) for walker_up, walker_dn in zip(b1_uhf.prop_data["walkers"][0], b1_uhf.prop_data["walkers"][1])
    ])

    rghf_fb = np.asarray([
        b1_rghf.trial._calc_force_bias_restricted(walker, b1_rghf.ham_data, b1_rghf.wave_data) for walker in b1_rghf.prop_data["walkers"]
    ])

    cghf_fb = np.asarray([
        b1_cghf.trial._calc_force_bias_restricted(walker, b1_cghf.ham_data, b1_cghf.wave_data) for walker in b1_cghf.prop_data["walkers"]
    ])

    print_diff(rghf_fb, uhf_fb)
    print_diff(cghf_fb, rghf_fb)

    if n_chk == 1:
        block_energy_n, prop_data = b1_uhf.sampler.propagate_phaseless(
            b1_uhf.ham, b1_uhf.ham_data, b1_uhf.propagator, b1_uhf.prop_data, b1_uhf.trial, b1_uhf.wave_data
        )
        block_weight_n = np.array([jnp.sum(prop_data["weights"])], dtype="float32")
        print(block_energy_n)
        print(block_weight_n)
        block_energy_n, prop_data = b1_rghf.sampler.propagate_phaseless(
            b1_rghf.ham, b1_rghf.ham_data, b1_rghf.propagator, b1_rghf.prop_data, b1_rghf.trial, b1_rghf.wave_data
        )
        block_weight_n = np.array([jnp.sum(prop_data["weights"])], dtype="float32")
        print(block_energy_n)
        print(block_weight_n)
        block_energy_n, prop_data = b1_cghf.sampler.propagate_phaseless(
            b1_cghf.ham, b1_cghf.ham_data, b1_cghf.propagator, b1_cghf.prop_data, b1_cghf.trial, b1_cghf.wave_data
        )
        block_weight_n = np.array([jnp.sum(prop_data["weights"])], dtype="float32")
        print(block_energy_n)
        print(block_weight_n)

blocks = [6113, 6114, 6115, 6116]

uhf_path = "real_debug/uhf/"
rghf_path = "real_debug/ad/"
cghf_path = "complex_debug/ad/"

for n_block in blocks:
    print("\n### Block "+str(n_block)+" ###")
    read_block(n_block, 1)
    read_block(n_block, 2)
    read_block(n_block, 3)
    read_block(n_block, 4)
