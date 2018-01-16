#!/usr/bin/env python

from kccsd_rhf import RCCSD
#from pyscf.pbc.cc.kccsd_rhf import RCCSD
from pyscf.pbc import gto, scf

from ctf_helper import comm, rank, size, omp

cell = gto.Cell()
cell.atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = 'gth-dzvp'
cell.pseudo = 'gth-pade'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.mesh = [15,15,15]
#cell.verbose = 5
cell.build()
#cell.max_memory = 50000

kpts = cell.make_kpts([3,3,3])
mf = scf.KRHF(cell, kpts)
mf.__dict__.update(scf.chkfile.load('k27_c2_dzp.chk', 'scf'))
#mf.kernel()
#with omp(16):
#    if rank == 0:
#        mf.verbose = 4
#        mf.chkfile = 'k27_c2_dzp.chk'
#        mf.kernel()
#mf.mo_occ = comm.bcast(mf.mo_occ)
#mf.mo_coeff = comm.bcast(mf.mo_coeff)
#mf.mo_energy = comm.bcast(mf.mo_energy)

mycc = RCCSD(mf)
mycc.verbose = 6
mycc.max_cycle = 3
mycc.kernel()
