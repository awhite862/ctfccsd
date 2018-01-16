import ctfccsd
from pyscf import gto, scf, lib
from mpi4py import MPI
import sys

natm, bas, nproc = sys.argv[1:]

mol = gto.M(atom=[['H', 0, 0, i*1.8] for i in range(int(natm))],
            unit='bohr', basis=bas, verbose=3)
mf = scf.RHF(mol).density_fit()
with ctfccsd.omp(int(nproc)):
    if ctfccsd.rank == 0:
        mf.verbose = 4
        mf.run()
mf.mo_occ = ctfccsd.comm.bcast(mf.mo_occ)
mf.mo_coeff = ctfccsd.comm.bcast(mf.mo_coeff)
mf.mo_energy = ctfccsd.comm.bcast(mf.mo_energy)


mycc = ctfccsd.CCSD(mf)
mycc.verbose = 6
mycc.conv_tol = 1e-4
mycc.conv_tol_normt = 1e-2
mycc.max_cycle = 3
mycc.run()

MPI.Finalize()
