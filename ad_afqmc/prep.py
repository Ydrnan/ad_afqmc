import os
from enum import Enum, auto
import numpy as np
import pickle
from ad_afqmc import utils

import pyscf
from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD
from pyscf.cc.gccsd import GCCSD
from pyscf.scf.hf import RHF
from pyscf.scf.rohf import ROHF
from pyscf.scf.uhf import UHF
from pyscf.scf.ghf import GHF

class PrepAfqmc:
    __slots__ = (
        "mol", "ao_basis", "mo_basis", "options",
        "ci", "cc", "trial", "path", "tmp", "io", "pyscf"
    )
    def __init__(self):
        self.mol = Mol()
        self.ao_basis = AoBasis()
        self.mo_basis = MoBasis()
        self.options = None
        self.ci = None # Not used
        self.cc = None # Not used
        self.trial = None # Not used
        self.path = Path()
        self.tmp = Tmp() # This is bad and need to be changed
        self.io = IO()
        self.pyscf = Pyscf()

    # Returns data needed for the afqmc calculation
    def get_setup_data(self):
        # TODO add an assert to make sure setup_afqmc has been run before
        return (
            self.tmp.ham_data,
            self.tmp.ham,
            self.tmp.prop,
            self.tmp.trial,
            self.tmp.wave_data,
            self.tmp.trial_ket,
            self.tmp.wave_data_ket,
            self.tmp.sampler,
            self.tmp.observable,
            self.options,
        )

    # Prepares the data for the afqmc calculation
    def setup_afqmc(self):
        # TODO add an assert to make sure prep has been run before
        self.set_ham()
        self.apply_symmetry_mask()
        self.read_observable()
        self.set_trial()
        self.set_prop()
        self.set_sampler()
        self.setup_print()

    # Reads/Computes/Writes what is needed for the calculation
    def prep(self):
        self.set_options()
        self.set_integrals()
        self.set_trial_coeff()
        self.set_amplitudes()

    def prep_afqmc(
        mf_or_cc,
        mf_or_cc_ket = None,
        basis_coeff = None,
        norb_frozen = 0,
        chol_cut = 1e-5,
        integrals = None,
        tmpdir = None,
    ):
        prep = PrepAfqmc()
        prep.set_mol(mf_or_cc.mol)
        prep.set_pyscf_mf_cc(mf_or_cc, mf_or_cc_ket)
        prep.set_basis_coeff(basis_coeff)
        prep.set_frozen_core(norb_frozen)
        prep.set_chol_cut(chol_cut)
        if integrals is not None: prep.set_custom_integrals(integrals)
        prep.set_tmpdir(tmpdir)
        prep.options = {}

        if tmpdir is None:
            prep.io.set_noio()
        else:
            prep.io.set_write()

        prep.prep()

        return prep

    ################
    ### Wrappers ###
    ################
    def setup_print(self):
        print(f"# norb: {self.mo_basis.norb}")
        print(f"# nelec: {(self.mo_basis.nelec_sp)}")
        print("#")
        for op in self.options:
            if self.options[op] is not None:
                print(f"# {op}: {self.options[op]}")
        print("#")

    def read_observable(self):
        self.tmp.observable = utils.read_observable(
            self.mo_basis.norb,
            self.options,
            self.path.tmpdir,
        )

    def set_ham(self):
        mo = self.mo_basis
        mol = self.mol
        ham, ham_data = utils.set_ham(mo.norb, mo.h0, mo.h1, mo.chol, mol.ene0)
        self.tmp.ham = ham
        self.tmp.ham_data = ham_data

    def apply_symmetry_mask(self):
        self.tmp.ham_data = utils.apply_symmetry_mask(self.tmp.ham_data, self.options)

    def set_trial(self):
        self.tmp.trial, self.tmp.wave_data = utils.set_trial(
            self
        )
        if "trial_ket" in self.options:
            self.tmp.trial_ket, self.tmp.wave_data_ket = utils.set_trial(
                self
            )

    def set_prop(self):
        self.tmp.prop = utils.set_prop(self.options)

    def set_sampler(self):
        self.tmp.sampler = utils.set_sampler(self.options)

    def set_mol(self, mol):
        if not isinstance(mol, pyscf.gto.Mole):
            raise TypeError(f"Expected an argument of type '{type(pyscf.gto.Mole)}' but received '{type(mol)}'.")
        self.pyscf.mol = mol
        self.mol.spin = mol.spin
        self.mol.n_a, self.mol.n_b = mol.nelec
        self.ao_basis.n_ao = mol.nao
        self.ao_basis.overlap = mol.intor('int1e_ovlp')

    def set_pyscf_mf_cc(self, mf_or_cc, mf_or_cc_ket):
        self.pyscf.set_mf_cc(mf_or_cc, mf_or_cc_ket)

    def set_basis_coeff(self, basis_coeff):
        if basis_coeff is None:
            basis_coeff = self.pyscf.get_basis_coeff()

        # TODO assert basis coeff shape
        self.mo_basis.basis_coeff = basis_coeff

    def set_frozen_core(self, norb_frozen):
        # Super dirty
        if self.pyscf.cc is not None and hasattr(self.pyscf.cc, "frozen"):
            if self.pyscf.cc.frozen is not None:
                norb_frozen = self.pyscf.cc.frozen
            else:
                norb_frozen = 0

        if type(norb_frozen) != int:
            raise TypeError(f"Number of frozen orbitals must be an integer, but is '{type(norb_frozen)}'.")
        if norb_frozen < 0:
            raise ValueError(f"Number of frozen orbitals must be >= 0, but is '{norb_frozen}'.")
        if norb_frozen >= self.pyscf.mf.mo_coeff.shape[-1] :
            raise ValueError(f"Number of frozen orbitals '{norb_frozen}' must be smaller than the number of MOs '{self.pyscf.mf.mo_coeff.shape[-1]}'.")

        self.mo_basis.norb_frozen = norb_frozen

    def set_chol_cut(self, chol_cut):
        if type(chol_cut) != float:
            raise TypeError(f"chol_cut expected to be a float but has type '{type(chol_cut)}'.")
        if chol_cut < 0.0:
            raise ValueError(f"chol_cut expected to be >= 0.")

        self.ao_basis.chol_cut = chol_cut

    def set_custom_integrals(self, integrals):
        if type(integrals) != dict:
            raise TypeError(f"integrals expected to be a dict but has type '{type(integrals)}'.")

        self.ao_basis.custom_integrals = integrals

    def set_tmpdir(self, tmpdir):
        self.path.set(tmpdir)

    ###############
    ### Options ###
    ###############
    def set_options(self):
        io = self.io.options

        # Check
        IOMode.check(io)

        if io.is_read():
            self.path.check(self.path.options + "/options.bin")

        # Read
        if io.is_read():
            self.read_options()
        # Compute
        elif io.is_write() or io.is_noio():
            self.options = utils.get_options(self.options)
            self.mol.ene0 = self.options["ene0"]
        else:
            raise ValueError(f"io should be IOMode.Read/Write/NoIO, not {io}.")

        # Write
        if io.is_write():
            self.write_options()
            self.io.set_read_options()

    def read_options(self):
        self.options = utils.read_options(self.options, self.path.options)
        self.mol.ene0 = self.options["ene0"]

    def write_options(self):
        assert self.path.options is not None
        with open(self.path.options+"/options.bin", "wb") as f:
            pickle.dump(self.options, f)

    def replace_options(self, options):
        if self.options == {}:
            raise ValueError("PrepAfqmc.options must not be empty dict.")
        if type(self.options) != type({}):
            raise TypeError(f"PrepAfqmc.options must be a dict but is '{type(options)}'.")
        if type(options) != type({}):
            raise TypeError(f"Expected a dict but received a '{type(options)}'.")

        for key, value in options.items():
            if key not in self.options:
                raise KeyError(f"Key '{key}' not found.")
            self.options[key] = value

    #################
    ### Integrals ###
    #################
    # TODO add asserts
    def set_integrals(self):
        io = self.io.fcidump

        # Check
        IOMode.check(io)

        if io.is_read():
            self.path.check(self.path.fcidump + "/FCIDUMP_chol")

        # Read
        if io.is_read():
            self.read_fcidump()
        # Compute
        elif io.is_write() or io.is_noio():
            #if self.mo_basis.chol is None:
            self.compute_integrals()
        else:
            raise ValueError(f"io should be IOMode.Read/Write/NoIO, not {io}.")

        # Write
        if io.is_write():
            self.write_fcidump()
            self.io.set_read_fcidump()

    def compute_integrals(self):
        # TODO the function compute_cholesky_integrals should be split
        h1e, chol, nelec, enuc, _, _ = utils.compute_cholesky_integrals(
            self.pyscf.mol,
            self.pyscf.mf,
            self.mo_basis.basis_coeff,
            self.ao_basis.custom_integrals,
            self.mo_basis.norb_frozen,
            self.ao_basis.chol_cut,
        )
        print("# Finished calculating Cholesky integrals\n#")
        
        norb = h1e.shape[-1]
        n_chol = chol.shape[0]
        print("# Size of the correlation space:")
        print(f"# Number of electrons: {nelec}")
        print(f"# Number of orbitals: {norb}")
        print(f"# Number of Cholesky vectors: {n_chol}\n#")
        chol = chol.reshape((-1, norb, norb))
        v0 = 0.5 * np.einsum("nik,njk->ij", chol, chol, optimize="optimal")
        h1e_mod = h1e - v0
        chol = chol.reshape((n_chol, norb * norb))
        self.mo_basis.norb = norb
        self.mo_basis.nelec_sp = nelec
        self.mo_basis.h0 = enuc
        self.mo_basis.h1 = h1e
        self.mo_basis.h1_mod = h1e_mod
        self.mo_basis.chol = chol
        self.ao_basis.n_chol = n_chol

    def read_fcidump(self):
        h0, h1, h1_mod, chol, nbasis, nelec_sp = utils.read_fcidump(self.path.fcidump)
        if self.options["walker_type"] == "generalized":
            norb = nbasis // 2
        else: 
            norb = nbasis
        self.mo_basis.norb = norb
        self.mo_basis.h0 = h0
        self.mo_basis.h1 = h1
        self.mo_basis.h1_mod = h1_mod
        self.mo_basis.chol = chol
        self.mo_basis.nelec_sp = nelec_sp

    def write_fcidump(self):
        utils.write_dqmc(
            self.mo_basis.h1,
            self.mo_basis.h1_mod,
            self.mo_basis.chol,
            sum(self.mo_basis.nelec_sp),
            self.mo_basis.norb,
            self.mo_basis.h0,
            ms=self.mol.spin,
            filename=self.path.fcidump + "/FCIDUMP_chol",
            mo_coeffs=self.mo_basis.trial_coeff,
        )

    ###################
    ### Trial coeff ###
    ###################
    # TODO add asserts
    def set_trial_coeff(self):
        io = self.io.trial_coeff

        # Check
        IOMode.check(io)

        if io.is_read():
            self.path.check(self.path.trial_coeff + "/mo_coeff.npz")

        # Read
        if io.is_read():
            self.read_trial_coeff()

        # Compute
        elif io.is_write() or io.is_noio():
            #if self.mo_basis.trial_coeff is None:
            self.mo_basis.trial_coeff = utils.get_trial_coeffs(
                self.pyscf.mol,
                self.pyscf.mf,
                self.mo_basis.basis_coeff,
                self.mo_basis.norb,
                self.mo_basis.norb_frozen,
            )
        else:
            raise ValueError(f"io should be IOMode.Read/Write/NoIO, not {io}.")

        # Write
        if io.is_write():
            self.write_trial_coeff()
            self.io.set_read_trial_coeff()

    def read_trial_coeff(self):
        self.mo_basis.trial_coeff = utils.load_mo_coefficients(self.path.trial_coeff)

    # TODO write should only write them
    def write_trial_coeff(self):
        utils.write_trial_coeffs(
            self.pyscf.mol,
            self.pyscf.mf,
            self.mo_basis.basis_coeff,
            self.mo_basis.norb,
            self.mo_basis.norb_frozen,
            self.path.trial_coeff,
        )

    ##################
    ### Amplitudes ###
    ##################
    def set_amplitudes(self):
        bra = self.options["trial"]
        ket = self.options["trial_ket"]

        trial = ["cisd", "CISD", "ucisd", "UCISD", "gcisd", "gcisd_complex", "ccsd", "uccsd"]
        if not bra in trial and not ket in trial and self.pyscf.cc is None:
            return

        io = self.io.amplitudes

        # Check
        IOMode.check(io)

        if io.is_read():
            self.path.check(self.path.amplitudes + "/amplitudes.npz")

        # Read
        if io.is_read():
            self.tmp.amplitudes = np.load(self.path.amplitudes + "/amplitudes.npz")
        # Compute
        elif io.is_write() or io.is_noio():
            self.set_ci_from_cc()
        else:
            raise ValueError(f"io should be IOMode.Read/Write/NoIO, not {io}.")

        # Write
        if io.is_write():
            self.write_amplitudes()
            self.io.set_read_amplitudes()

    def set_ci_from_cc(self):
        if self.pyscf.cc is None:
            raise AttributeError(f"self.pyscf.cc must exist and point to the cc pyscf object in order to compute the amplitudes.")
        self.tmp.amplitudes = utils.get_ci_amplitudes_from_cc(self.pyscf.cc)

    # TODO should only write the amplitudes
    # TODO check the shape of amplitude to avoid issue when using frozen orbitals
    def write_amplitudes(self):
        utils.write_pyscf_ccsd(self.pyscf.cc, self.path.amplitudes)

class Path:
    __slots__ = ("tmpdir", "options", "fcidump", "trial_coeff", "amplitudes")

    def __init__(self):
        self.tmpdir = None
        self.options = None
        self.fcidump = None
        self.trial_coeff = None
        self.amplitudes = None

    # Brute-force
    def set(self, path):
        t = type(path)
        if t != str and t is not type(None):
            raise TypeError(f"Expected a string/NoneType but received '{t}'.")

        for attr in self.__slots__:
            setattr(self, attr, path)

    # Check if the path exists
    def check(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File '{path}' does not exist.")

class Tmp: pass

class Mol:
    __slots__ = ("n_a", "n_b", "spin", "ene0")

    def __init__(self):
        self.n_a = None
        self.n_b = None
        self.spin = None
        self.ene0 = None

class AoBasis:
    __slots__ = ("n_ao", "n_chol", "chol_cut", "overlap", "custom_integrals")

    def __init__(self):
        self.n_ao = None
        self.n_chol = None
        self.chol_cut = None
        self.overlap = None
        self.custom_integrals = None

class MoBasis:
    __slots__ = (
        "mo_type", "norb", "nelec_sp", "norb_frozen",
        "mo_coeff", "basis_coeff", "trial_coeff","chol",
        "h0", "h1", "h1_mod",
    )
    def __init__(self):
        self.mo_type = None
        self.norb = None # Number of active MOs
        self.nelec_sp = None # Number of active electron
        self.norb_frozen = None
        self.mo_coeff = None
        self.basis_coeff = None
        self.trial_coeff = None
        self.chol = None
        self.h0 = None
        self.h1 = None
        self.h1_mod = None

    class MoType:
        class Restricted: pass
        class Unrestricted: pass
        class Generalized: pass

class IOMode(Enum):
    Read = auto()
    Write = auto()
    NoIO = auto()

    def is_read(self):
        return self is self.Read

    def is_write(self):
        return self is self.Write

    def is_noio(self):
        return self is self.NoIO

    def check(io_mode):
        if not isinstance(io_mode, IOMode):
            raise TypeError(f"Type '{type(io_mode)}' instead of IOMode.Read/Write/NoIO.")

class IO:
    __slots__ = ("options", "fcidump", "trial_coeff", "amplitudes")

    def __init__(self):
        self.options = None
        self.fcidump = None
        self.trial_coeff = None
        self.amplitudes = None

    def set(self, io_mode):
        IOMode.check(io_mode)
        for attr in self.__slots__:
            setattr(self, attr, io_mode)

    def set_noio(self):
        self.set(IOMode.NoIO)

    def set_read(self):
        self.set(IOMode.Read)

    def set_write(self):
        self.set(IOMode.Write)

# Creates the method set_read/write/noio_ for all the IO fields, i.e.,
# set_read_options, set_read_fcidump, ...
def _make_setter(field, io_mode):
    def setter(self):
        setattr(self, field, io_mode)
    return setter

for field in IO.__slots__:
    for io_mode in IOMode:
        fname = f"set_{io_mode.name.lower()}_{field}"
        setattr(IO, fname, _make_setter(field, io_mode))

class Pyscf:
    __slots__ = ("mol", "mf", "cc")

    def __init__(self):
        self.mol = None
        self.mf = None
        self.cc = None

    def set_mf_cc(self, mf_or_cc, mf_or_cc_ket):
        if isinstance(mf_or_cc, (CCSD, UCCSD, GCCSD)):
            self.mf = mf_or_cc._scf
            self.cc = mf_or_cc
        elif isinstance(mf_or_cc_ket, (CCSD, UCCSD, GCCSD)):
            self.mf = mf_or_cc_ket._scf
            self.cc = mf_or_cc_ket
        elif isinstance(mf_or_cc, (RHF, ROHF, UHF, GHF)):
            self.mf = mf_or_cc
        else:
            raise TypeError(f"Unexpected object '{mf_or_cc}'.")

        if self.mf.mo_coeff is None:
            raise ValueError(f"mo_coeff is None in the pyscf scf instance, please check that you ran correctly the scf calculation.")

        if self.cc is not None and (self.cc.t1 is None or self.cc.t2 is None):
            raise ValueError(f"t1 and/or t2 are None in the pyscf ccsd instance, please check that you ran correctly the ccsd calculation.")

    def get_basis_coeff(self):
        if isinstance(self.mf, UHF):
            basis_coeff = self.mf.mo_coeff[0]
        elif isinstance(self.mf, (RHF, ROHF, GHF)):
            basis_coeff = self.mf.mo_coeff
        else:
            raise TypeError(f"Unexpected object '{self.mf}'.")

        return basis_coeff
