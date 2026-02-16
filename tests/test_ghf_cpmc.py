import pytest
from ad_afqmc import config

from ad_afqmc import (
    lattices,
    wavefunctions, 
    propagation,
    pyscf_interface, 
    launch_script, 
    run_afqmc, 
    driver,
    hf_guess
)
from ad_afqmc.walkers import GHFWalkers, UHFWalkers

from pyscf import gto, scf, ao2mo
import numpy as np
import scipy.linalg as la
import jax.numpy as jnp
import os

tmpdir="tmp"
chol_cut = 1e-8

if not os.path.exists(tmpdir):
    os.makedirs(tmpdir)

def make_env():
    old_cfg = dict(config.afqmc_config)

    config.afqmc_config["use_mpi"] = False
    config.setup_jax()
    comm = config.setup_comm()

    return {
        "comm": comm,
        "old_cfg": old_cfg,
    }

def teardown_env(env):
    config.afqmc_config.clear()
    config.afqmc_config.update(env["old_cfg"])

@pytest.fixture(scope="module")
def env():
    # Save old config so we can restore it
    old_cfg = dict(config.afqmc_config)

    config.afqmc_config['use_mpi'] = False
    config.setup_jax()
    comm = config.setup_comm()

    yield {
        "comm": comm,
    }

    # restore config after tests in this module
    config.afqmc_config.clear()
    config.afqmc_config.update(old_cfg)

def random_orthogonal_real(n):
    rng = np.random.default_rng()
    A = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    Q *= np.sign(np.diag(R))
    return Q

options = {
"dt": 0.005,
"n_eql": 1,
"n_ene_blocks": 1,
"n_sr_blocks": 5,
"n_blocks": 4,
"n_prop_steps": 50,
"n_walkers": 4,
"seed": 8,
"trial": "",
"walker_type": "",
}

def check_hf(mf, integrals, options, comm):
    nmo = np.shape(mf.mo_coeff)[-1]
    n_elec = mf.mol.nelec
    nocc = sum(n_elec)
    hcore_ao = integrals["h1"]
    n_ao = hcore_ao.shape[-1]
    n_walkers = options["n_walkers"]

    # RHF/UHF
    if isinstance(mf, scf.rhf.RHF):
        options["trial"] = "rhf"
        options["walker_type"] = "restricted"
    elif isinstance(mf, scf.uhf.UHF):
        options["trial"] = "uhf"
        options["walker_type"] = "unrestricted"
    
    pyscf_interface.prep_afqmc(
        mf, basis_coeff=np.eye(n_ao), integrals=integrals, chol_cut=chol_cut, tmpdir=tmpdir
    )
    ham_data, ham, prop, trial, wave_data, sampler, observable, options = launch_script.setup_afqmc(
        options, tmp_dir=tmpdir
    )
    trial = wavefunctions.uhf_cpmc(n_ao, n_elec)
    wave_data["mo_coeff"] = [
        mf.mo_coeff[0][:, :n_elec[0]], 
        mf.mo_coeff[1][:, :n_elec[1]]
    ]
    wave_data["rdm1"] = mf.make_rdm1()
    ham_data["u"] = integrals["u"]
    prop = propagation.propagator_cpmc(
        dt=options["dt"],
        n_walkers=options["n_walkers"],
        walker_type=options["walker_type"]
    )
    init_walkers = UHFWalkers([
        jnp.array([mf.mo_coeff[0][:, : n_elec[0]]] * n_walkers),
        jnp.array([mf.mo_coeff[1][:, : n_elec[1]]] * n_walkers),
    ])
    ene1, err1 = driver.afqmc(
        ham_data, ham, prop, trial, wave_data, sampler, observable, options, comm, tmpdir=tmpdir,
        init_walkers=init_walkers
    )
    
    # -------------------------------------------------------------------------
    # RHF/UHF based GHF
    options["trial"] = "ghf"
    options["walker_type"] = "generalized"   
    integrals_g = integrals.copy()
    integrals_g["h1"] = la.block_diag(hcore_ao, hcore_ao)
    
    # Build GHF object.
    gmf = scf.GHF(mf.mol)
    gmf.get_hcore = lambda *args: integrals_g["h1"]
    gmf.get_ovlp = lambda *args: np.eye(2*n_ao)
    gmf._eri = ao2mo.restore(8, integrals["h2"], n_ao)
    gmf.mo_coeff = np.zeros((2*n_ao, 2*n_ao))
    gmf.mo_coeff[:n_ao, :n_ao] = mf.mo_coeff[0]
    gmf.mo_coeff[n_ao:, n_ao:] = mf.mo_coeff[1]
    gmf.mo_occ = np.zeros(2*n_ao)
    gmf.mo_occ[:n_ao] = mf.mo_occ[0]
    gmf.mo_occ[n_ao:] = mf.mo_occ[1]
    dm = gmf.make_rdm1()

    pyscf_interface.prep_afqmc(
        gmf, basis_coeff=np.eye(2*n_ao), integrals=integrals_g, chol_cut=chol_cut, tmpdir=tmpdir
    )
    ham_data, ham, prop, trial, wave_data, sampler, observable, options = launch_script.setup_afqmc(
        options, tmp_dir=tmpdir
    )
    trial = wavefunctions.ghf_cpmc(n_ao, n_elec)
    wave_data["mo_coeff"] = gmf.mo_coeff[:, gmf.mo_occ>0]
    wave_data["rdm1"] = dm
    ham_data["u"] = integrals_g["u"]
    prop = propagation.propagator_cpmc(
        dt=options["dt"],
        n_walkers=options["n_walkers"],
        walker_type=options["walker_type"]
    )
    init_walkers = GHFWalkers(jnp.array([gmf.mo_coeff[:, gmf.mo_occ>0]] * n_walkers))
    ene2, err2 = driver.afqmc(
        ham_data, ham, prop, trial, wave_data, sampler, observable, options, comm, tmpdir=tmpdir,
        init_walkers=init_walkers
    )
    
    assert np.isclose(ene1, ene2, atol=1e-6), f"{ene1} {ene2}"
    assert np.isclose(err1, err2, atol=1e-8), f"{err1} {err2}"
    

def check_hf_slow(mf, integrals, options, comm):
    nmo = np.shape(mf.mo_coeff)[-1]
    n_elec = mf.mol.nelec
    nocc = sum(n_elec)
    hcore_ao = integrals["h1"]
    n_ao = hcore_ao.shape[-1]
    n_walkers = options["n_walkers"]

    # RHF/UHF
    if isinstance(mf, scf.rhf.RHF):
        options["trial"] = "rhf"
        options["walker_type"] = "restricted"
    elif isinstance(mf, scf.uhf.UHF):
        options["trial"] = "uhf"
        options["walker_type"] = "unrestricted"
    
    pyscf_interface.prep_afqmc(
        mf, basis_coeff=np.eye(n_ao), integrals=integrals, chol_cut=chol_cut, tmpdir=tmpdir
    )
    ham_data, ham, prop, trial, wave_data, sampler, observable, options = launch_script.setup_afqmc(
        options, tmp_dir=tmpdir
    )
    trial = wavefunctions.uhf_cpmc(n_ao, n_elec)
    wave_data["mo_coeff"] = [
        mf.mo_coeff[0][:, :n_elec[0]], 
        mf.mo_coeff[1][:, :n_elec[1]]
    ]
    wave_data["rdm1"] = mf.make_rdm1()
    ham_data["u"] = integrals["u"]
    prop = propagation.propagator_cpmc_slow(
        dt=options["dt"],
        n_walkers=options["n_walkers"],
        walker_type=options["walker_type"]
    )
    init_walkers = UHFWalkers([
        jnp.array([mf.mo_coeff[0][:, : n_elec[0]]] * n_walkers),
        jnp.array([mf.mo_coeff[1][:, : n_elec[1]]] * n_walkers),
    ])
    ene1, err1 = driver.afqmc(
        ham_data, ham, prop, trial, wave_data, sampler, observable, options, comm, tmpdir=tmpdir,
        init_walkers=init_walkers
    )
    
    # -------------------------------------------------------------------------
    # RHF/UHF based GHF
    options["trial"] = "ghf"
    options["walker_type"] = "generalized"   
    integrals_g = integrals.copy()
    integrals_g["h1"] = la.block_diag(hcore_ao, hcore_ao)
    
    # Build GHF object.
    gmf = scf.GHF(mf.mol)
    gmf.get_hcore = lambda *args: integrals_g["h1"]
    gmf.get_ovlp = lambda *args: np.eye(2*n_ao)
    gmf._eri = ao2mo.restore(8, integrals["h2"], n_ao)
    gmf.mo_coeff = np.zeros((2*n_ao, 2*n_ao))
    gmf.mo_coeff[:n_ao, :n_ao] = mf.mo_coeff[0]
    gmf.mo_coeff[n_ao:, n_ao:] = mf.mo_coeff[1]
    gmf.mo_occ = np.zeros(2*n_ao)
    gmf.mo_occ[:n_ao] = mf.mo_occ[0]
    gmf.mo_occ[n_ao:] = mf.mo_occ[1]
    dm = gmf.make_rdm1()

    pyscf_interface.prep_afqmc(
        gmf, basis_coeff=np.eye(2*n_ao), integrals=integrals_g, chol_cut=chol_cut, tmpdir=tmpdir
    )
    ham_data, ham, prop, trial, wave_data, sampler, observable, options = launch_script.setup_afqmc(
        options, tmp_dir=tmpdir
    )
    trial = wavefunctions.ghf_cpmc(n_ao, n_elec)
    wave_data["mo_coeff"] = gmf.mo_coeff[:, gmf.mo_occ>0]
    wave_data["rdm1"] = dm
    ham_data["u"] = integrals_g["u"]
    prop = propagation.propagator_cpmc_slow(
        dt=options["dt"],
        n_walkers=options["n_walkers"],
        walker_type=options["walker_type"]
    )
    init_walkers = GHFWalkers(jnp.array([gmf.mo_coeff[:, gmf.mo_occ>0]] * n_walkers))
    ene2, err2 = driver.afqmc(
        ham_data, ham, prop, trial, wave_data, sampler, observable, options, comm, tmpdir=tmpdir,
        init_walkers=init_walkers
    )
    
    assert np.isclose(ene1, ene2, atol=1e-6), f"{ene1} {ene2}"
    assert np.isclose(err1, err2, atol=1e-8), f"{err1} {err2}"
    

# Hubbard
def test_hubbard(env):
    check_hubbard(env)

def check_hubbard(env):
    comm = env["comm"]

    U = 12.0
    nup, ndown = 8, 8
    n_elec = (nup, ndown)
    nx, ny = 4, 4
    nwalkers = 50
    bc = 'xc'
    
    # Lattice
    lattice = lattices.triangular_grid(nx, ny, boundary=bc)
    n_sites = lattice.n_sites
    nocc = sum(n_elec)

    # Integrals
    integrals = {}
    integrals["h0"] = 0.0
    h1 = -1.0 * lattice.create_adjacency_matrix()
    integrals["h1"] = h1
    h2 = np.zeros((n_sites, n_sites, n_sites, n_sites))
    for i in range(n_sites): h2[i, i, i, i] = U
    integrals["h2"] = ao2mo.restore(8, h2, n_sites)
    integrals["u"] = U

    # Make dummy molecule
    mol = gto.Mole()
    mol.nelectron = nocc
    mol.incore_anyway = True
    mol.spin = abs(n_elec[0] - n_elec[1])
    mol.build()

    # Prep trial.
    mf = scf.UHF(mol)
    mf.get_hcore = lambda *args: integrals["h1"]
    mf.get_ovlp = lambda *args: np.eye(n_sites)
    mf._eri = ao2mo.restore(8, integrals["h2"], n_sites)
    psi_init = hf_guess.get_ghf_neel_guess(lattice)
    dm_init = [
        psi_init[:n_sites] @ psi_init[:n_sites].T.conj(),
        psi_init[n_sites:] @ psi_init[n_sites:].T.conj(),
    ]
    mf.kernel(dm_init)

    check_hf(mf, integrals, options, comm)
    check_hf_slow(mf, integrals, options, comm)


if __name__ == "__main__":
    env = make_env()

    try:
        check_hubbard(env)
    
    finally:
        teardown_env(env)
