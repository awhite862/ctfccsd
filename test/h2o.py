from pyscf import gto
from pyscf import scf
from ctfccsd import CCSD
from ctf_helper import comm, rank, omp

mol = gto.Mole()
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]
mol.basis = 'ccpvdz'
mol.build()
mol.verbose = 4
mol.max_memory = 1
mf = scf.RHF(mol)
with omp(4):
    if rank == 0:
        mf.verbose = 4
        mf.kernel()
mf.mo_occ = comm.bcast(mf.mo_occ)
mf.mo_coeff = comm.bcast(mf.mo_coeff)
mf.mo_energy = comm.bcast(mf.mo_energy)
CCSD(mf).run()
