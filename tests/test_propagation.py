import numpy as np
import scipy as sp

from ad_afqmc import config

config.setup_jax()
from jax import numpy as jnp
from jax import scipy as jsp
from jax import random

from ad_afqmc import hamiltonian, propagation, wavefunctions
from ad_afqmc.walkers import GHFWalkers

# -----------------------------------------------------------------------------
# Fixed Hamiltonian objects.
seed = 102
np.random.seed(seed)
n_walkers, norb, nelec, nchol = 10, 10, (5, 5), 5

h0 = np.random.rand(1)[0]
h1 = jnp.array(np.random.rand(2, norb, norb))
chol = jnp.array(np.random.rand(nchol, norb * norb))
chol_g = np.zeros((nchol, 2*norb, 2*norb))

for i in range(nchol):
    chol_i = chol[i].reshape((norb, norb))
    chol_g[i] = jsp.linalg.block_diag(chol_i, chol_i)

chol_g = chol_g.reshape(nchol, -1)

ham_handler = hamiltonian.hamiltonian(norb)

fields = random.normal(
    random.PRNGKey(seed), shape=(n_walkers, chol.shape[0])
)

# -----------------------------------------------------------------------------
# RHF propagator.
trial = wavefunctions.rhf(norb, nelec)
prop_handler = propagation.propagator_afqmc(n_walkers=n_walkers, n_chunks=5)

wave_data = {}
wave_data["mo_coeff"] = jnp.eye(norb)[:, : nelec[0]]
wave_data["rdm1"] = jnp.array([wave_data["mo_coeff"] @ wave_data["mo_coeff"].T] * 2)

ham_data = {}
ham_data["h0"] = h0
ham_data["h1"] = h1.copy()
ham_data["chol"] = chol.copy()
ham_data["ene0"] = 0.0
ham_data = ham_handler.build_measurement_intermediates(ham_data, trial, wave_data)
ham_data = ham_handler.build_propagation_intermediates(
    ham_data, prop_handler, trial, wave_data
)

prop_data = prop_handler.init_prop_data(trial, wave_data, ham_data, seed)
# prop_data["key"] = random.PRNGKey(seed)
prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)


# -----------------------------------------------------------------------------
# UHF propagator.
nelec_sp = (5, 4)
trial_u = wavefunctions.uhf(norb, nelec_sp)
prop_handler_u = propagation.propagator_afqmc(
    n_walkers=n_walkers, n_chunks=5, walker_type="unrestricted"
)

wave_data_u = {}
wave_data_u["mo_coeff"] = [
    jnp.array(np.random.rand(norb, nelec_sp[0])),
    jnp.array(np.random.rand(norb, nelec_sp[1])),
]
wave_data_u["rdm1"] = jnp.array(
    [
        jnp.array(wave_data_u["mo_coeff"][0] @ wave_data_u["mo_coeff"][0].T),
        jnp.array(wave_data_u["mo_coeff"][1] @ wave_data_u["mo_coeff"][1].T),
    ]
)

ham_data_u = {}
ham_data_u["h0"] = h0
ham_data_u["h1"] = h1.copy()
ham_data_u["chol"] = chol.copy()
ham_data_u["ene0"] = 0.0
ham_data_u = ham_handler.build_measurement_intermediates(
    ham_data_u, trial_u, wave_data_u
)
ham_data_u = ham_handler.build_propagation_intermediates(
    ham_data_u, prop_handler_u, trial_u, wave_data_u
)

prop_data_u = prop_handler_u.init_prop_data(trial_u, wave_data_u, ham_data_u, seed)
# prop_data_u["key"] = random.PRNGKey(seed)
prop_data_u["overlaps"] = trial_u.calc_overlap(prop_data_u["walkers"], wave_data_u)

# -----------------------------------------------------------------------------
# GHF propagator from UHF.
nocc = sum(nelec_sp)
trial_g = wavefunctions.ghf(norb, nelec_sp)
prop_handler_g = propagation.propagator_afqmc(
    n_walkers=n_walkers, n_chunks=5, walker_type="generalized"
)

wave_data_g = {}
wave_data_g["mo_coeff"] = jsp.linalg.block_diag(*wave_data_u["mo_coeff"])
#wave_data_g["rdm1"] = jnp.array([wave_data_g["mo_coeff"] @ wave_data_g["mo_coeff"].T])
wave_data_g["rdm1"] = wave_data_g["mo_coeff"] @ wave_data_g["mo_coeff"].T

ham_data_g = {}
ham_data_g["h0"] = h0
ham_data_g["h1"] = jsp.linalg.block_diag(*h1)
ham_data_g["chol"] = chol_g.copy()
ham_data_g["ene0"] = 0.0
ham_data_g = ham_handler.build_measurement_intermediates(
    ham_data_g, trial_g, wave_data_g
)
ham_data_g = ham_handler.build_propagation_intermediates(
    ham_data_g, prop_handler_g, trial_g, wave_data_g
)

init_walkers = np.zeros((n_walkers, 2*norb, nocc), dtype=np.complex128)
for iw in range(n_walkers):
    init_walkers[iw] = jsp.linalg.block_diag(
        prop_data_u["walkers"].data[0][iw, :, :nelec_sp[0]],
        prop_data_u["walkers"].data[1][iw, :, :nelec_sp[1]],
    )
init_walkers = GHFWalkers(jnp.array(init_walkers))

prop_data_g = prop_handler_g.init_prop_data(
    trial_g, wave_data_g, ham_data_g, seed, init_walkers)
#prop_data_g["key"] = prop_data_u["key"]
prop_data_g["overlaps"] = trial_g.calc_overlap(prop_data_g["walkers"], wave_data_g)

# -----------------------------------------------------------------------------
# GHF-complex propagator from UHF.
nocc = sum(nelec_sp)
trial_gc = wavefunctions.ghf_complex(norb, nelec_sp)
prop_handler_gc = propagation.propagator_afqmc(
    n_walkers=n_walkers, n_chunks=5, walker_type="generalized"
)

wave_data_gc = {}
wave_data_gc["mo_coeff"] = jsp.linalg.block_diag(*wave_data_u["mo_coeff"])
#wave_data_g["rdm1"] = jnp.array([wave_data_g["mo_coeff"] @ wave_data_g["mo_coeff"].T])
wave_data_gc["rdm1"] = wave_data_gc["mo_coeff"] @ wave_data_gc["mo_coeff"].T

ham_data_gc = {}
ham_data_gc["h0"] = h0
ham_data_gc["h1"] = jsp.linalg.block_diag(*h1)
ham_data_gc["chol"] = chol_g.copy()
ham_data_gc["ene0"] = 0.0
ham_data_gc = ham_handler.build_measurement_intermediates(
    ham_data_gc, trial_gc, wave_data_gc
)
ham_data_gc = ham_handler.build_propagation_intermediates(
    ham_data_gc, prop_handler_gc, trial_gc, wave_data_gc
)

init_walkers = np.zeros((n_walkers, 2*norb, nocc), dtype=np.complex128)
for iw in range(n_walkers):
    init_walkers[iw] = jsp.linalg.block_diag(
        prop_data_u["walkers"].data[0][iw, :, :nelec_sp[0]],
        prop_data_u["walkers"].data[1][iw, :, :nelec_sp[1]],
    )
init_walkers = GHFWalkers(jnp.array(init_walkers))

prop_data_gc = prop_handler_gc.init_prop_data(
    trial_gc, wave_data_gc, ham_data_gc, seed, init_walkers)
#prop_data_g["key"] = prop_data_u["key"]
prop_data_gc["overlaps"] = trial_gc.calc_overlap(prop_data_gc["walkers"], wave_data_gc)

# -----------------------------------------------------------------------------
# UHF-CPMC propagator.
trial_cpmc_u = wavefunctions.uhf_cpmc(norb, nelec_sp)
prop_handler_cpmc_u = propagation.propagator_cpmc(
    n_walkers=n_walkers, n_chunks=5, walker_type="unrestricted"
)

ham_data_cpmc_u = {}
ham_data_cpmc_u["h0"] = h0
ham_data_cpmc_u["h1"] = h1.copy()
ham_data_cpmc_u["chol"] = chol.copy()
ham_data_cpmc_u["u"] = 4.0
ham_data_cpmc_u["ene0"] = 0.0
ham_data_cpmc_u = ham_handler.build_measurement_intermediates(
    ham_data_cpmc_u, trial_cpmc_u, wave_data_u
)
ham_data_cpmc_u = ham_handler.build_propagation_intermediates(
    ham_data_cpmc_u, prop_handler_cpmc_u, trial_cpmc_u, wave_data_u
)

prop_data_cpmc_u = prop_handler_cpmc_u.init_prop_data(
    trial_cpmc_u, wave_data_u, ham_data_cpmc_u, seed
)
#prop_data_cpmc_u["key"] = random.PRNGKey(seed)
prop_data_cpmc_u["overlaps"] = trial_cpmc_u.calc_overlap(
    prop_data_cpmc_u["walkers"], wave_data_u
)

# -----------------------------------------------------------------------------
# GHF-CPMC propagator.
trial_cpmc_g = wavefunctions.ghf_cpmc(norb, nelec_sp)
prop_handler_cpmc_g = propagation.propagator_cpmc(
    n_walkers=n_walkers, n_chunks=5, walker_type="generalized"
)

ham_data_cpmc_g = {}
ham_data_cpmc_g["h0"] = h0
ham_data_cpmc_g["h1"] = jsp.linalg.block_diag(*h1)
ham_data_cpmc_g["chol"] = chol_g.copy()
ham_data_cpmc_g["u"] = 4.0
ham_data_cpmc_g["ene0"] = 0.0
ham_data_cpmc_g = ham_handler.build_measurement_intermediates(
    ham_data_cpmc_g, trial_cpmc_g, wave_data_g
)
ham_data_cpmc_g = ham_handler.build_propagation_intermediates(
    ham_data_cpmc_g, prop_handler_cpmc_g, trial_cpmc_g, wave_data_g
)

prop_data_cpmc_g = prop_handler_cpmc_g.init_prop_data(
    trial_cpmc_g, wave_data_g, ham_data_cpmc_g, seed, init_walkers
)
#prop_data_cpmc_g["key"] = prop_data_cpmc_u["key"]
prop_data_cpmc_g["overlaps"] = trial_cpmc_g.calc_overlap(
    prop_data_cpmc_g["walkers"], wave_data_g
)

# -----------------------------------------------------------------------------
# UHF-CPMC-nn propagator.
neighbors = tuple((i, (i + 1) % norb) for i in range(norb))
prop_handler_cpmc_nn_u = propagation.propagator_cpmc_nn(n_walkers=n_walkers, neighbors=neighbors)

ham_data_cpmc_nn_u = {}
ham_data_cpmc_nn_u["h0"] = h0
ham_data_cpmc_nn_u["h1"] = h1.copy()
ham_data_cpmc_nn_u["chol"] = chol.copy()
ham_data_cpmc_nn_u["u"] = 4.0
ham_data_cpmc_nn_u["u_1"] = 1.0
ham_data_cpmc_nn_u["ene0"] = 0.0
ham_data_cpmc_nn_u = ham_handler.build_measurement_intermediates(
    ham_data_cpmc_nn_u, trial_cpmc_u, wave_data_u
)
ham_data_cpmc_nn_u = ham_handler.build_propagation_intermediates(
    ham_data_cpmc_nn_u, prop_handler_cpmc_nn_u, trial_cpmc_u, wave_data_u
)

prop_data_cpmc_nn_u = prop_handler_cpmc_nn_u.init_prop_data(
    trial_cpmc_u, wave_data_u, ham_data_cpmc_nn_u, seed
)
#prop_data_cpmc_nn_u["key"] = random.PRNGKey(seed)
prop_data_cpmc_nn_u["overlaps"] = trial_cpmc_u.calc_overlap(
    prop_data_cpmc_nn_u["walkers"], wave_data_u
)

#prop_handler_cpmc = propagation.propagator_cpmc(n_walkers=n_walkers)
#prop_handler_cpmc_slow = propagation.propagator_cpmc_slow(n_walkers=n_walkers)
#
#neighbors = tuple((i, (i + 1) % norb) for i in range(norb))
#prop_handler_cpmc_nn = propagation.propagator_cpmc_nn(n_walkers=10, neighbors=neighbors)
#prop_handler_cpmc_nn_slow = propagation.propagator_cpmc_nn_slow(
#    n_walkers=10, neighbors=neighbors
#)


# -----------------------------------------------------------------------------
# RHF tests.
def test_stochastic_reconfiguration_local():
    prop_data["key"], subkey = random.split(prop_data["key"])
    zeta = random.uniform(subkey)
    new_walkers, new_weights = prop_data["walkers"].stochastic_reconfiguration_local(
        prop_data["weights"], zeta
    )
    assert new_walkers.data.shape == prop_data["walkers"].data.shape
    assert new_weights.shape == prop_data["weights"].shape


def test_propagate():
    prop_data_new = prop_handler.propagate_constrained(
        trial, ham_data, prop_data, fields, wave_data
    )
    assert prop_data_new["walkers"].data.shape == prop_data["walkers"].data.shape
    assert prop_data_new["weights"].shape == prop_data["weights"].shape
    assert prop_data_new["overlaps"].shape == prop_data["overlaps"].shape


# -----------------------------------------------------------------------------
# UHF tests.
def test_stochastic_reconfiguration_local_u():
    prop_data_u["key"], subkey = random.split(prop_data_u["key"])
    zeta = random.uniform(subkey)
    new_walkers, new_weights = prop_data_u["walkers"].stochastic_reconfiguration_local(
        prop_data_u["weights"], zeta
    )
    assert new_walkers.data[0].shape == prop_data_u["walkers"].data[0].shape
    assert new_walkers.data[1].shape == prop_data_u["walkers"].data[1].shape
    assert new_weights.shape == prop_data_u["weights"].shape


def test_propagate_u():
    prop_data_new = prop_handler_u.propagate_constrained(
        trial_u, ham_data_u, prop_data_u, fields, wave_data_u
    )
    assert (
        prop_data_new["walkers"].data[0].shape == prop_data_u["walkers"].data[0].shape
    )
    assert (
        prop_data_new["walkers"].data[1].shape == prop_data_u["walkers"].data[1].shape
    )
    assert prop_data_new["weights"].shape == prop_data_u["weights"].shape
    assert prop_data_new["overlaps"].shape == prop_data_u["overlaps"].shape


def test_propagate_free_u():
    prop_data_new = prop_handler_u.propagate_free(
        trial_u, ham_data_u, prop_data_u, fields, wave_data_u
    )
    assert (
        prop_data_new["walkers"].data[0].shape == prop_data_u["walkers"].data[0].shape
    )
    assert (
        prop_data_new["walkers"].data[1].shape == prop_data_u["walkers"].data[1].shape
    )
    assert prop_data_new["weights"].shape == prop_data_u["weights"].shape
    assert prop_data_new["overlaps"].shape == prop_data_u["overlaps"].shape


# -----------------------------------------------------------------------------
# GHF tests.
def test_apply_trotprop_g():
    walkers_u = prop_data_u["walkers"]
    walkers_g = np.zeros((n_walkers, 2*norb, nocc), dtype=np.complex128)
    walkers_g[:, :norb, :nelec_sp[0]] = walkers_u.data[0]
    walkers_g[:, norb:, nelec_sp[0]:] = walkers_u.data[1]
    walkers_g = GHFWalkers(jnp.array(walkers_g))
    
    walkers_new = prop_handler_g._apply_trotprop(
        walkers_g, fields, ham_data_g
    )
    walkers_new_ref = prop_handler_u._apply_trotprop(
        walkers_u, fields, ham_data_u
    )

    for iw in range(n_walkers):
        np.testing.assert_allclose(
            walkers_new.data[iw],
            jsp.linalg.block_diag(
                walkers_new_ref.data[0][iw], walkers_new_ref.data[1][iw]
            )
        )


def test_propagate_g():
    prop_data_new = prop_handler_g.propagate_constrained(
        trial_g, ham_data_g, prop_data_g, fields, wave_data_g
    )
    prop_data_new_ref = prop_handler_u.propagate_constrained(
        trial_u, ham_data_u, prop_data_u, fields, wave_data_u
    )

    np.testing.assert_allclose(
        prop_data_new["overlaps"], 
        prop_data_new_ref["overlaps"]
    )
    np.testing.assert_allclose(
        prop_data_new["weights"], 
        prop_data_new_ref["weights"]
    )
    np.testing.assert_allclose(
        prop_data_new["walkers"].data[:, :norb, :nelec_sp[0]], 
        prop_data_new_ref["walkers"].data[0]
    )
    np.testing.assert_allclose(
        prop_data_new["walkers"].data[:, norb:, nelec_sp[0]:], 
        prop_data_new_ref["walkers"].data[1]
    )
    np.testing.assert_allclose(
        prop_data_new["pop_control_ene_shift"], 
        prop_data_new_ref["pop_control_ene_shift"]
    )


def test_propagate_gc():
    prop_data_new = prop_handler_gc.propagate_constrained(
        trial_gc, ham_data_gc, prop_data_gc, fields, wave_data_gc
    )
    prop_data_new_ref = prop_handler_u.propagate_constrained(
        trial_u, ham_data_u, prop_data_u, fields, wave_data_u
    )

    np.testing.assert_allclose(
        prop_data_new["overlaps"], 
        prop_data_new_ref["overlaps"]
    )
    np.testing.assert_allclose(
        prop_data_new["weights"], 
        prop_data_new_ref["weights"]
    )
    np.testing.assert_allclose(
        prop_data_new["walkers"].data[:, :norb, :nelec_sp[0]], 
        prop_data_new_ref["walkers"].data[0]
    )
    np.testing.assert_allclose(
        prop_data_new["walkers"].data[:, norb:, nelec_sp[0]:], 
        prop_data_new_ref["walkers"].data[1]
    )
    np.testing.assert_allclose(
        prop_data_new["pop_control_ene_shift"], 
        prop_data_new_ref["pop_control_ene_shift"]
    )

# -----------------------------------------------------------------------------
# UHF-CPMC tests.
def test_propagate_cpmc_u():
    prop_handler_cpmc_slow = propagation.propagator_cpmc_slow(n_walkers=n_walkers)
    prop_data_new = prop_handler_cpmc_u.propagate_constrained(
        trial_cpmc_u, ham_data_cpmc_u, prop_data_cpmc_u, fields, wave_data_u
    )
    prop_data_new_slow = prop_handler_cpmc_slow.propagate_constrained(
        trial_cpmc_u, ham_data_cpmc_u, prop_data_cpmc_u, fields, wave_data_u
    )
    greens_new_slow = trial_cpmc_u.calc_green_full(
        prop_data_new_slow["walkers"].data, wave_data_u
    )

    np.testing.assert_allclose(
        prop_data_new["overlaps"],
        prop_data_new_slow["overlaps"]
    )
    np.testing.assert_allclose(
        prop_data_new["weights"],
        prop_data_new_slow["weights"]
    )
    np.testing.assert_allclose(
        prop_data_new["walkers"].data[0],
        prop_data_new_slow["walkers"].data[0]
    )
    np.testing.assert_allclose(
        prop_data_new["walkers"].data[1],
        prop_data_new_slow["walkers"].data[1]
    )
    np.testing.assert_allclose(
        prop_data_new["greens"],
        greens_new_slow
    )
    np.testing.assert_allclose(
        prop_data_new["pop_control_ene_shift"],
        prop_data_new_slow["pop_control_ene_shift"]
    )


# -----------------------------------------------------------------------------
# GHF-CPMC tests.
def test_propagate_cpmc_one_body_g():
    prop_data_new = prop_handler_cpmc_g.propagate_one_body(
        trial_cpmc_g, ham_data_cpmc_g, prop_data_cpmc_g, wave_data_g
    )
    prop_data_new_ref = prop_handler_cpmc_u.propagate_one_body(
        trial_cpmc_u, ham_data_cpmc_u, prop_data_cpmc_u, wave_data_u
    )

    np.testing.assert_allclose(
        prop_data_new["overlaps"], 
        prop_data_new_ref["overlaps"]
    )
    np.testing.assert_allclose(
        prop_data_new["weights"], 
        prop_data_new_ref["weights"]
    )
    np.testing.assert_allclose(
        prop_data_new["walkers"].data[:, :norb, :nelec_sp[0]], 
        prop_data_new_ref["walkers"].data[0]
    )
    np.testing.assert_allclose(
        prop_data_new["walkers"].data[:, norb:, nelec_sp[0]:], 
        prop_data_new_ref["walkers"].data[1]
    )
    np.testing.assert_allclose(
        prop_data_new["pop_control_ene_shift"], 
        prop_data_new_ref["pop_control_ene_shift"]
    )

    for iw in range(n_walkers):
        np.testing.assert_allclose(
            prop_data_new["greens"][iw], 
            sp.linalg.block_diag(*prop_data_new_ref["greens"][iw])
        )


def test_propagate_cpmc_g():
    prop_data_new = prop_handler_cpmc_g.propagate_constrained(
        trial_cpmc_g, ham_data_cpmc_g, prop_data_cpmc_g, fields, wave_data_g
    )
    prop_data_new_ref = prop_handler_cpmc_u.propagate_constrained(
        trial_cpmc_u, ham_data_cpmc_u, prop_data_cpmc_u, fields, wave_data_u
    )

    np.testing.assert_allclose(
        prop_data_new["overlaps"],
        prop_data_new_ref["overlaps"]
    )
    np.testing.assert_allclose(
        prop_data_new["weights"],
        prop_data_new_ref["weights"]
    )
    np.testing.assert_allclose(
        prop_data_new["walkers"].data[:, :norb, :nelec_sp[0]],
        prop_data_new_ref["walkers"].data[0]
    )
    np.testing.assert_allclose(
        prop_data_new["walkers"].data[:, norb:, nelec_sp[0]:],
        prop_data_new_ref["walkers"].data[1]
    )
    np.testing.assert_allclose(
        prop_data_new["pop_control_ene_shift"],
        prop_data_new_ref["pop_control_ene_shift"]
    )

    for iw in range(n_walkers):
        np.testing.assert_allclose(
            prop_data_new["greens"][iw], 
            sp.linalg.block_diag(*prop_data_new_ref["greens"][iw])
        )

"""
def test_propagate_cpmc_nn_u():
    trial_cpmc_u = wavefunctions.uhf_cpmc(norb, nelec_sp)
    ham_data_u["u"] = 4.0
    ham_data_u["u_1"] = 1.0

    prop_handler_cpmc_nn_slow = propagation.propagator_cpmc_nn_slow(
        n_walkers=n_walkers, neighbors=neighbors
    )
    prop_data_new = prop_handler_cpmc_nn_u.propagate_constrained(
        trial_cpmc_u, ham_data_cpmc_nn_u, prop_data_cpmc_nn_u, fields, wave_data_u
    )
    assert prop_data_new["weights"].shape == prop_data_u["weights"].shape
    assert prop_data_new["overlaps"].shape == prop_data_u["overlaps"].shape

    prop_data_cpmc_nn_u["key"] = random.PRNGKey(seed)
    prop_data_new_slow = prop_handler_cpmc_nn_slow.propagate_constrained(
        trial_cpmc_u, ham_data_cpmc_nn_u, prop_data_cpmc_nn_u, fields, wave_data_u
    )
    
    np.testing.assert_allclose(
        prop_data_new["overlaps"],
        prop_data_new_slow["overlaps"]
    )
    np.testing.assert_allclose(
        prop_data_new["weights"],
        prop_data_new_slow["weights"]
    )
    np.testing.assert_allclose(
        prop_data_new["walkers"].data[0],
        prop_data_new_slow["walkers"].data[0]
    )
    np.testing.assert_allclose(
        prop_data_new["walkers"].data[1],
        prop_data_new_slow["walkers"].data[1]
    )
"""

if __name__ == "__main__":
    test_stochastic_reconfiguration_local()
    test_propagate()

    test_stochastic_reconfiguration_local_u()
    test_propagate_u()
    test_propagate_free_u()
    
    test_apply_trotprop_g()
    test_propagate_g()
    test_propagate_gc()

    test_propagate_cpmc_u()
    test_propagate_cpmc_one_body_g()
    test_propagate_cpmc_g()

    #test_propagate_cpmc_nn_u()
