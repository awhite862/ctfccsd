#!/usr/bin/env python
#
# Authors: James D. McClain <jmcclain@princeton.edu>
#          Timothy Berkelbach <tim.berkelbach@gmail.com>
#

import time
import tempfile
import numpy
import numpy as np

import pyscf.pbc.tools.pbc as tools
from pyscf import lib
import pyscf.ao2mo
from pyscf.lib import logger
import pyscf.cc
import pyscf.cc.ccsd
from pyscf.pbc import scf
from pyscf.pbc import df
from pyscf.pbc.cc.kccsd import get_frozen_mask
from pyscf.pbc.cc.kccsd_rhf import get_nocc, get_nmo
from pyscf.pbc.cc import kintermediates_rhf as imdk
from pyscf.lib import linalg_helper
from pyscf.pbc.cc import kpoint_helper

import ctf
from ctf_helper import comm, rank, size, Logger, omp, static_partition
#comm = ctf.comm()
#rank = comm.rank()
#size = comm.np()
#ctf.tensor = np.ndarray
#ctf.astensor = np.asarray
#ctf.to_nparray = np.asarray

einsum = ctf.einsum
#einsum = np.einsum
#einsum = lib.einsum

# This is restricted (R)CCSD
# Ref: Hirata, et al., J. Chem. Phys. 120, 2581 (2004)

def kernel(cc, eris, t1=None, t2=None, max_cycle=50, tol=1e-8, tolnormt=1e-6,
           max_memory=2000, verbose=logger.INFO):
    """Exactly the same as pyscf.cc.ccsd.kernel, which calls a
    *local* energy() function."""
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = Logger(cc.stdout, verbose)

    if t1 is None and t2 is None:
        t1, t2 = cc.init_amps(eris)[1:]
    elif t1 is None:
        nocc = cc.nocc
        nvir = cc.nmo - nocc
        t1 = numpy.zeros((nocc,nvir), eris.dtype)
    elif t2 is None:
        t2 = cc.init_amps(eris)[2]

    cput1 = cput0 = (time.clock(), time.time())
    nkpts, nocc, nvir = t1.shape
    eold = 0.0
    eccsd = 0.0
    if cc.diis:
        adiis = lib.diis.DIIS(cc, cc.diis_file)
        adiis.space = cc.diis_space
    else:
        adiis = lambda t1,t2,*args: (t1,t2)

    conv = False
    for istep in range(max_cycle):
        t1new, t2new = cc.update_amps(t1, t2, eris)
        #normt = numpy.linalg.norm(t1new-t1) + numpy.linalg.norm(t2new-t2)
        dt = t1new - t1
        normt = einsum('xia,xia->', dt.conj(), dt)
        dt = t2new - t2
        normt+= einsum('xyzijab,xyzijab->', dt.conj(), dt)
        normt = numpy.real(normt.sum()) ** .5

        t1, t2 = t1new, t2new
        t1new = t2new = dt = None
        if cc.diis:
            t1, t2 = cc.diis(t1, t2, istep, normt, eccsd-eold, adiis)
        eold, eccsd = eccsd, energy(cc, t1, t2, eris)
        log.info('istep = %d  E(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                 istep, eccsd, eccsd - eold, normt)
        cput1 = log.timer('CCSD iter', *cput1)
        if abs(eccsd-eold) < tol and normt < tolnormt:
            conv = True
            break
    log.timer('CCSD', *cput0)
    return conv, eccsd, t1, t2


def update_amps(cc, t1, t2, eris):
    time0 = time.clock(), time.time()
    log = Logger(cc.stdout, cc.verbose)
    nkpts, nocc, nvir = t1.shape

    kconserv = cc.kconserv
# Ps is Permutation transformation matrix
# The physical meaning of Ps matrix is the conservation of moment.
# Given the four indices in Ps, the element shows whether moment conservation
# holds (1) or not (0)
    Ps = kconserve_pmatrix(nkpts, kconserv)

    tau = t2.copy()
    #idx = numpy.arange(nkpts)
    #tau[idx,:,idx] += einsum('xic,ykd->xyikcd', t1, t1)
    tau.i('xyxijab') << einsum('xic,ykd->xyikcd', t1, t1).i('xyijab')

    print 'memory before Wvvvv', rank, lib.current_memory()
#    #Wvvvv = imdk.cc_Wvvvv(t1,t2,eris,kconserv)
#    Wvvvv = einsum('xyzakcd,ykb->xyzabcd', eris.vovv, -1*t1)
#    Wvvvv += einsum('xyzabcd,xyzw->yxwbadc', Wvvvv, Ps)
#    Wvvvv += eris.vvvv
#
#    t2new = einsum('zwuabcd,xyzw,xyuijcd->xyzijab', Wvvvv, Ps, tau)

    tau2 = einsum('xyuijcd,xyuv->xuvijcd', tau, Ps)
    vovv = einsum('zwuakcd,zwuv->zuvakcd', eris.vovv, Ps)
    print 'memory Wvvvv 1', rank, lib.current_memory()
    #tmp  = einsum('zuvakcd,xuvijcd,xyuv->xyzijak', vovv, tau2, Ps)
    tmp  = einsum('zuvakcd,xuvijcd->xzuvijak', vovv, tau2)
    tmp = einsum('xzuvijak,xyuv->xyzijak', tmp, Ps)
    print 'memory Wvvvv 2', rank, lib.current_memory()
    t2new = einsum('xyzw,wkb,xyzijak->xyzijab', Ps, -1*t1, tmp)
    t2new += einsum('xyzijab,xyzw->yxwjiba', t2new, Ps)
    print 'memory Wvvvv 3', rank, lib.current_memory()
    #t2new += einsum('zuvabcd,xuvijcd,xyuv->xyzijab', eris.vvvv1, tau2, Ps)
    tmp = einsum('zuvabcd,xuvijcd->xzuvijab', eris.vvvv1, tau2)
    t2new += einsum('xzuvijab,xyuv->xyzijab', tmp, Ps)
    Wvvvv = tau2 = vovv = None
    print 'memory after Wvvvv', rank, lib.current_memory()

    #Foo = imdk.cc_Foo(t1,t2,eris,kconserv)
    Soovv = 2 * eris.oovv
    Soovv -= einsum('xyzw,xyzijab->xywijba', Ps, eris.oovv)
    Foo = einsum('xyzklcd,xyzilcd->xki', Soovv, tau)
    Foo+= eris.foo

    #Fvv = imdk.cc_Fvv(t1,t2,eris,kconserv)
    Fvv = eris.fvv.copy()
    Fvv-= einsum('xyzklcd,xyzklad->zac', Soovv, tau)

    #Fov = imdk.cc_Fov(t1,t2,eris,kconserv)
    Fov = einsum('xyxklcd,yld->xkc', Soovv, t1)
    Fov += eris.fov

    #Loo = imdk.Loo(t1,t2,eris,kconserv)
    Loo  = einsum('xyxklic,ylc->xki', eris.ooov, t1) * 2
    Loo -= einsum('yxxlkic,ylc->xki', eris.ooov, t1)
    Loo += einsum('xkc,xic->xki', eris.fov, t1)
    Loo += Foo

    #Lvv = imdk.Lvv(t1,t2,eris,kconserv)
    Lvv  = einsum('xyxakcd,ykd->xac',eris.vovv,t1) * 2
    Lvv -= einsum('xyyakdc,ykd->xac',eris.vovv,t1)
    Lvv -= einsum('xkc,xka->xac', eris.fov, t1)
    Lvv += Fvv

    print 'memory before Woooo', rank, lib.current_memory()
    #:Woooo = imdk.cc_Woooo(t1,t2,eris,kconserv)
    #Woooo = einsum('xyzw,wjc,xyzklic->xyzklij', Ps, t1, eris.ooov)
    tmp = einsum('xyzw,wjc->xyzjc', Ps, t1)
    Woooo = einsum('xyzjc,xyzklic->xyzklij', tmp, eris.ooov)
    Woooo += einsum('xyzklij,xyzw->yxwlkji', Woooo, Ps)
    #Woooo += einsum('zwuijcd,xyzw,xyuklcd->xyzklij', tau, Ps, eris.oovv)
    tmp = einsum('zwuijcd,xyuklcd->xyzwklij', tau, eris.oovv)
    Woooo += einsum('xyzwklij,xyzw->xyzklij', tmp, Ps)
    Woooo += eris.oooo
    tmp = None

    print 'memory before Wvoov', rank, lib.current_memory()
    #Wvoov = imdk.cc_Wvoov(t1,t2,eris,kconserv)
    Wvoov  = einsum('xyzakdc,zid->xyzakic', eris.vovv, t1)
    Wvoov -= einsum('xyzlkic,xla->xyzakic', eris.ooov, t1)
    Wvoov += eris.voov
    #Wvoov += 0.5*einsum('uyvlkdc,uzxv,zuxilad->xyzakic', Soovv, Ps, t2)
    tmp = einsum('uyvlkdc,uzxv->xyzulkdc', Soovv, Ps)
    Wvoov += 0.5*einsum('xyzulkdc,zuxilad->xyzakic', tmp, t2)
    tmp = None
    tau2 = tau.copy()
    #tau2[idx,:,idx] += einsum('xic,ykd->xyikcd', t1, t1)
    tau2.i('xyxijab') << einsum('xic,ykd->xyikcd', t1, t1).i('xyijab')
    #Wvoov -= 0.5*einsum('uyvlkdc,uzxv,uzxliad->xyzakic', eris.oovv, Ps, tau2)
    tmp = einsum('uyvlkdc,uzxv->xyzulkdc', eris.oovv, Ps)
    Wvoov -= 0.5*einsum('xyzulkdc,uzxliad->xyzakic', tmp, tau2)
    Soovv = tmp = None

    print 'memory before Wvoov1', rank, lib.current_memory()
    #Wvovo = imdk.cc_Wvovo(t1,t2,eris,kconserv)
    #Wvoov1 = einsum('xyzbkci,xyzw->xywbkic', Wvovo, Ps)

    #Wvoov1  = einsum('xywakcd,xyzw,zid->xyzakic', eris.vovv, Ps, t1)
    tmp = einsum('xyzw,zid->xyzwid', Ps, t1)
    Wvoov1  = einsum('xywakcd,xyzwid->xyzakic', eris.vovv, tmp)
    Wvoov1 -= einsum('yxzklic,xla->xyzakic', eris.ooov, t1)
    Wvoov1 += eris.ovov.transpose(1,0,2,4,3,5,6)
    #Wvoov1 -= 0.5*einsum('uyvlkcd,xyzv,uzxliad->xyzakic', eris.oovv, Ps, tau2)
    tmp = einsum('uyvlkcd,xyzv->xyzulkcd', eris.oovv, Ps)
    Wvoov1 -= 0.5*einsum('xyzulkcd,uzxliad->xyzakic', tmp, tau2)
    tmp = None

    # T1 equation
    t1new = -2.*einsum('xkc,xka,xic->xia', eris.fov, t1, t1)
    t1new += einsum('xac,xic->xia', Fvv, t1)
    t1new -= einsum('xki,xka->xia', Foo, t1)

    tau2 = 2 * einsum('yxyijab->xyijab', t2)
    tau2 -= einsum('xyyijab->xyjiab', t2)
    #tau2[idx,idx] += einsum('xic,xka->xkica', t1, t1)
    tau2.i('xxijab') << einsum('xic,xka->xkica', t1, t1).i('xijab')

    t1new += einsum('ykc,xykica->xia', Fov, tau2)
    tau2 = None
    t1new += einsum('xyxakic,ykc->xia',eris.voov, t1) * 2
    t1new -= einsum('yxxkaic,ykc->xia',eris.ovov, t1)

    Svovv = 2 * eris.vovv
    Svovv -= einsum('xyzaibc,xyzw->xywaicb', eris.vovv, Ps)
    t1new += einsum('xyzakcd,xyzikcd->xia', Svovv, tau)
    Svovv = None

    ooov = 2*eris.ooov
    ooov -= eris.ooov.transpose(1,0,2,4,3,5,6)
    t1new -= einsum('xyzklic,xyzklac->zia', ooov, tau)
    ooov = None

    print 'memory before T2', rank, lib.current_memory()
    # T2 equation
    #t2new += einsum('uwxklij,uwxy,uwzklab->xyzijab', Woooo, Ps, tau)
    tmp = einsum('uwxklij,uwxy->uwxyklij', Woooo, Ps)
    t2new += einsum('uwxyklij,uwzklab->xyzijab', tmp, tau)
    tmp = None
    t2new += eris.oovv.conj()
    t2new *= .5

    t2new += einsum('zac,xyzijcb->xyzijab', Lvv, t2)
    t2new -= einsum('xki,xyzkjab->xyzijab', Loo, t2)

    vovv_C = eris.vovv.conj().copy()
    #vovv_C -= einsum('xyzw,zka,zwxkbic->yxwciba', Ps, t1, eris.ovov)
    tmp = einsum('xyzw,zka->xyzwka', Ps, t1)
    vovv_C -= einsum('xyzwka,zwxkbic->yxwciba', tmp, eris.ovov)
    t2new += einsum('xyzcjab,xic->xyzijab', vovv_C, t1)
    vovv_C = tmp = None

    ooov_C = eris.ooov.conj().copy()
    #ooov_C += einsum('zwxakic,xyzw,yjc->yxwjika',eris.voov, Ps, t1)
    tmp = einsum('xyzw,yjc->xyzwjc', Ps, t1)
    ooov_C += einsum('zwxakic,xyzwjc->yxwjika', eris.voov, tmp)
    t2new -= einsum('xyzw,wkb,yxwjika->xyzijab', Ps, t1, ooov_C)
    ooov_C = tmp = None

    tmp2 = 2 * Wvoov
    tmp2 -= Wvoov1
    #t2new += einsum('xyzw,yuwjkbc,zuxakic->xyzijab', Ps, t2, tmp2)
    tmp = einsum('xyzw,yuwjkbc->xyzujkbc', Ps, t2)
    t2new += einsum('xyzujkbc,zuxakic->xyzijab', tmp, tmp2)
    tmp2 = tmp = None
    #t2new -= einsum('xyzw,uywkjbc,zuxakic->xyzijab', Ps, t2, Wvoov)
    tmp = einsum('xyzw,uywkjbc->xyzukjbc', Ps, t2)
    t2new -= einsum('xyzukjbc,zuxakic->xyzijab', tmp, Wvoov)
    tmp = None
    #t2new -= einsum('wuxbkic,xyzw,uyzkjac->xyzijab', Wvoov1, Ps, t2)
    tmp = einsum('wuxbkic,xyzw->xyzubkic', Wvoov1, Ps)
    t2new -= einsum('xyzubkic,uyzkjac->xyzijab', tmp, t2)
    tmp = None
    t2new += einsum('xyzijab,xyzw->yxwjiba', t2new, Ps)
    Wvoov = Wvoov1 = None

    mo_e = eris.mo_energy
    t1new /= (mo_e[:,:nocc].reshape(nkpts,nocc,1) -
              mo_e[:,nocc:].reshape(nkpts,1,nvir))

    eia = (mo_e[:,:nocc].reshape(nkpts,1,nocc,1) -
           mo_e[:,nocc:].reshape(1,nkpts,1,nvir))
    ejb = einsum('ywjb,xyzw->xyzjb', eia, Ps)
    eijab = (eia.reshape(nkpts,1,nkpts,nocc,1,nvir,1) +
             ejb.reshape(nkpts,nkpts,nkpts,1,nocc,1,nvir))
    t2new /= eijab

    time0 = log.timer_debug1('update t1 t2', *time0)
    return t1new, t2new


def energy(cc, t1, t2, eris):
    nkpts, nocc, nvir = t1.shape
    e  = einsum('xia,xia', eris.fov, t1) * 2
    e += einsum('xyzijab,xyzijab', eris.oovv, t2) * 2
    e -= einsum('yxzjiab,xyzijab', eris.oovv, t2)
    e += einsum('xyxijab,xia,yjb', eris.oovv, t1, t1) * 2
    e -= einsum('yxxjiab,xia,yjb', eris.oovv, t1, t1)
    e /= nkpts
    return numpy.real(e.sum())

# Ps is Permutation transformation matrix
# The physical meaning of Ps matrix is the conservation of moment.
# Given the four indices in Ps, the element shows whether moment conservation
# holds (1) or not (0)
def kconserve_pmatrix(nkpts, kconserv):
    Ps = numpy.zeros((nkpts,nkpts,nkpts,nkpts))
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                # Chemist's notation for momentum conserving t2(ki,kj,ka,kb)
                kb = kconserv[ki,ka,kj]
                Ps[ki,kj,ka,kb] = 1
    return ctf.astensor(Ps)


class RCCSD(pyscf.cc.ccsd.CCSD):

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        assert(isinstance(mf, scf.khf.KSCF))
        pyscf.cc.ccsd.CCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.max_space = 20
        self._keys = self._keys.union(['max_space'])
        self.kpts = mf.kpts
        self.mo_energy = mf.mo_energy
        self.nkpts = len(self.kpts)
        self.kconserv = tools.get_kconserv(mf.cell, mf.kpts)
        self.khelper = kpoint_helper.unique_pqr_list(mf.cell, mf.kpts)
        self.made_ee_imds = False
        self.made_ip_imds = False
        self.made_ea_imds = False
        self.diis = None

    nocc = property(get_nocc)
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    nmo = property(get_nmo)
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    get_nocc = get_nocc
    get_nmo = get_nmo

    def dump_flags(self):
        if rank == 0:
            pyscf.cc.ccsd.CCSD.dump_flags(self)

    def init_amps(self, eris):
        time0 = time.clock(), time.time()
        log = Logger(self.stdout, self.verbose)
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts

        mo_e = eris.mo_energy
        t1 = eris.fov.conj() / (mo_e[:,:nocc].reshape(nkpts,nocc,1) -
                                mo_e[:,nocc:].reshape(nkpts,1,nvir))

        eia = (mo_e[:,:nocc].reshape(nkpts,1,nocc,1) -
               mo_e[:,nocc:].reshape(1,nkpts,1,nvir))
        Ps = kconserve_pmatrix(nkpts, self.kconserv)
        ejb = einsum('ywjb,xyzw->xyzjb', eia, Ps)
        eijab = (eia.reshape(nkpts,1,nkpts,nocc,1,nvir,1) +
                 ejb.reshape(nkpts,nkpts,nkpts,1,nocc,1,nvir))
        t2 = eris.oovv.conj() / eijab

        emp2 = einsum('xyzijab,xyzijab', eris.oovv, t2) * 2
        emp2-= einsum('yxzjiab,xyzijab', eris.oovv, t2)
        self.emp2 = numpy.real(emp2.sum()) / nkpts

        log.info('Init t2, MP2 energy = %.15g', self.emp2)
        log.timer('init mp2', *time0)
        return self.emp2, t1, t2

    def kernel(self, t1=None, t2=None, eris=None, mbpt2=False):
        return self.ccsd(t1, t2, eris, mbpt2=mbpt2)
    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False):
        '''Ground-state CCSD.

        Kwargs:
            mbpt2 : bool
                Use one-shot MBPT2 approximation to CCSD.
        '''
        self.dump_flags()
        if eris is None:
            #eris = self.ao2mo()
            eris = self.ao2mo(self.mo_coeff)
        self.eris = eris
        log = Logger(self.stdout, self.verbose)
        if mbpt2:
            cctyp = 'MBPT2'
            self.e_corr, self.t1, self.t2 = self.init_amps(eris)
        else:
            cctyp = 'CCSD'
            self.converged, self.e_corr, self.t1, self.t2 = \
                    kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                           tol=self.conv_tol,
                           tolnormt=self.conv_tol_normt,
                           max_memory=self.max_memory, verbose=self.verbose)
            if self.converged:
                log.info('CCSD converged')
            else:
                log.info('CCSD not converged')
        if self._scf.e_tot == 0:
            log.info('E_corr = %.16g', self.e_corr)
        else:
            log.info('E(%s) = %.16g  E_corr = %.16g',
                     cctyp, self.e_tot, self.e_corr)
        return self.e_corr, self.t1, self.t2

    def ao2mo(self, mo_coeff=None):
        return _ERIS(self, mo_coeff)

    def update_amps(self, t1, t2, eris):
        return update_amps(self, t1, t2, eris)


class _ERIS:
    def __init__(self, cc, mo_coeff=None, method='incore',
                 ao2mofn=pyscf.ao2mo.outcore.general_iofree):
        cput0 = (time.clock(), time.time())
        log = Logger(cc.stdout, cc.verbose)
        moidx = get_frozen_mask(cc)
        nkpts = cc.nkpts
        nocc = cc.nocc
        nmo = cc.nmo
        nvir = nmo - nocc
        if mo_coeff is None:
            mo_coeff = cc.mo_coeff

        nao = cc.mo_coeff[0].shape[0]
        dtype = numpy.result_type(*cc.mo_coeff).char
        self.foo = ctf.tensor((nkpts,nocc,nocc), dtype=dtype)
        self.fov = ctf.tensor((nkpts,nocc,nvir), dtype=dtype)
        self.fvv = ctf.tensor((nkpts,nvir,nvir), dtype=dtype)
        self.mo_energy = ctf.astensor(cc.mo_energy)

        self.oooo = ctf.tensor((nkpts,nkpts,nkpts,nocc,nocc,nocc,nocc), dtype=dtype)
        self.ooov = ctf.tensor((nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir), dtype=dtype)
        self.oovv = ctf.tensor((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=dtype)
        self.ovov = ctf.tensor((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=dtype)
        self.voov = ctf.tensor((nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir), dtype=dtype)
        self.vovv = ctf.tensor((nkpts,nkpts,nkpts,nvir,nocc,nvir,nvir), dtype=dtype)
        self.vvvv1 = ctf.tensor((nkpts,nkpts,nkpts,nvir,nvir,nvir,nvir), dtype=dtype)

        with_df = cc._scf.with_df
        fao2mo = cc._scf.with_df.ao2mo

        kconserv = cc.kconserv

        oooo = [[]]
        ooov = [[]]
        oovv = [[]]
        ovov = [[]]
        voov = [[]]
        vovv = [[]]
        vvvv = [[]]
        idx_oooo = np.arange(nocc*nocc*nocc*nocc)
        idx_ooov = np.arange(nocc*nocc*nocc*nvir)
        idx_oovv = np.arange(nocc*nocc*nvir*nvir)
        idx_ovov = np.arange(nocc*nvir*nocc*nvir)
        idx_voov = np.arange(nvir*nocc*nocc*nvir)
        idx_vovv = np.arange(nvir*nocc*nvir*nvir)
        idx_vvvv = np.arange(nvir*nvir*nvir*nvir)

# TODO:
# For GDF method, the 7D eri tensor can be computed by 5D DF-tensor contraction

# Note, for FFTDF and AFTDF method, the 7D integral tensor cannot be initialized
# by the contraction of two 5D Fourier tensor (like rho_pq(G+k_{pq})).  This is
# because the k-point wrap-around in the (pq|rs).  See also the get_eri function
# in aft_ao2mo.py.

        k_prq = []
        tasks = list(static_partition(range(nkpts**3)))
        ntasks = max(comm.allgather(len(tasks)))
        print 'ao ints', rank, lib.current_memory(), 'ntasks', len(tasks)
        for itask in range(ntasks):
            if itask >= len(tasks):
                self.oooo.write([], [])
                self.ooov.write([], [])
                self.oovv.write([], [])
                self.ovov.write([], [])
                self.voov.write([], [])
                self.vovv.write([], [])
                self.vvvv1.write([], [])
                continue

            pqr = tasks[itask]
            kp = pqr // nkpts**2
            kq, kr = (pqr % nkpts**2).__divmod__(nkpts)
            ks = kconserv[kp,kq,kr]
            eri_kpt = fao2mo((mo_coeff[kp],mo_coeff[kq],mo_coeff[kr],mo_coeff[ks]),
                             (cc.kpts[kp],cc.kpts[kq],cc.kpts[kr],cc.kpts[ks]), compact=False)
            # <ij|kl> = (ik|jl)
            eri_kpt = eri_kpt.reshape(nmo,nmo,nmo,nmo).transpose(0,2,1,3)
            k_prq.append((kp,kr,kq))
            oooo = eri_kpt[:nocc,:nocc,:nocc,:nocc].ravel() / nkpts
            ooov = eri_kpt[:nocc,:nocc,:nocc,nocc:].ravel() / nkpts
            oovv = eri_kpt[:nocc,:nocc,nocc:,nocc:].ravel() / nkpts
            ovov = eri_kpt[:nocc,nocc:,:nocc,nocc:].ravel() / nkpts
            voov = eri_kpt[nocc:,:nocc,:nocc,nocc:].ravel() / nkpts
            vovv = eri_kpt[nocc:,:nocc,nocc:,nocc:].ravel() / nkpts
            vvvv = eri_kpt[nocc:,nocc:,nocc:,nocc:].ravel() / nkpts

            off = kp * nkpts**2 + kr * nkpts + kq
            self.oooo.write(off*idx_oooo.size+idx_oooo, oooo)
            self.ooov.write(off*idx_ooov.size+idx_ooov, ooov)
            self.oovv.write(off*idx_oovv.size+idx_oovv, oovv)
            self.ovov.write(off*idx_ovov.size+idx_ovov, ovov)
            self.voov.write(off*idx_voov.size+idx_voov, voov)
            self.vovv.write(off*idx_vovv.size+idx_vovv, vovv)

            # Move from the contraction
            # Wvvvv = einsum('zwuabcd,zwuv->zuvabcd', eris.vvvv, Ps)
            ks = kconserv[kp,kq,kr]
            off = kp * nkpts**2 + kq * nkpts + ks
            self.vvvv1.write(off*idx_vvvv.size+idx_vvvv, vvvv)
            log.debug1('_ERIS pqr %d', pqr)


if __name__ == '__main__':
    from pyscf.pbc import gto, scf
    cell = gto.Cell()
    cell.atom='''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.mesh = [15,15,15]
    #cell.verbose = 5
    cell.build()

    kpts = cell.make_kpts([1,1,3])
    mf = scf.KRHF(cell, kpts)
    #mf.chkfile = 'k_c2.chk'
    #mf.kernel()
    mf.__dict__.update(scf.chkfile.load('k_c2.chk', 'scf'))

    mycc = RCCSD(mf)
    mycc.verbose = 4

#    numpy.save('eris_fock.npy', eris.fock)
#    numpy.save('eris_oooo.npy', eris.oooo)
#    numpy.save('eris_ooov.npy', eris.ooov)
#    numpy.save('eris_oovv.npy', eris.oovv)
#    numpy.save('eris_ovov.npy', eris.ovov)
#    numpy.save('eris_voov.npy', eris.voov)
#    numpy.save('eris_vovv.npy', eris.vovv)
#    numpy.save('eris_vvvv.npy', eris.vvvv)
#    eris = lambda:None
#    eris.mo_energy = numpy.array(mf.mo_energy)
#    eris.fock = numpy.load('eris_fock.npy')
#    eris.oooo = numpy.load('eris_oooo.npy')
#    eris.ooov = numpy.load('eris_ooov.npy')
#    eris.oovv = numpy.load('eris_oovv.npy')
#    eris.ovov = numpy.load('eris_ovov.npy')
#    eris.voov = numpy.load('eris_voov.npy')
#    eris.vovv = numpy.load('eris_vovv.npy')
#    eris.vvvv = numpy.load('eris_vvvv.npy')
#
#    nocc = mycc.nocc
#    eris.foo = eris.fock[:,:nocc,:nocc] * 0
#    eris.fvv = eris.fock[:,nocc:,nocc:] * 0
#    eris.fov = eris.fock[:,:nocc,nocc:]

    eris = mycc.ao2mo()
#    print(abs(ctf.to_nparray(eris1.oooo) - eris.oooo).max())
#    print(abs(ctf.to_nparray(eris1.ooov) - eris.ooov).max())
#    print(abs(ctf.to_nparray(eris1.oovv) - eris.oovv).max())
#    print(abs(ctf.to_nparray(eris1.ovov) - eris.ovov).max())
#    print(abs(ctf.to_nparray(eris1.voov) - eris.voov).max())
#    print(abs(ctf.to_nparray(eris1.vovv) - eris.vovv).max())
#    print(abs(ctf.to_nparray(eris1.vvvv) - eris.vvvv).max())
    emp2,t1,t2 = mycc.init_amps(eris)
    print(emp2 - -0.131341714277)
    print(lib.finger(ctf.to_nparray(t2)) - (-0.0576606747244+0.0336759095279j))

    nkpts = kpts.shape[0]
    numpy.random.seed(1)
    no = mycc.nocc
    nv = mycc.nmo - no
    t2 = eris.oovv
    t1 =(numpy.random.random((nkpts,no,nv)) - .5
        +numpy.random.random((nkpts,no,nv))*.1j)
    t1 = ctf.astensor(t1)
    t2 = ctf.astensor(t2)

    t1b, t2b = mycc.update_amps(t1, t2, eris)
    print(lib.finger(ctf.to_nparray(t1b)) - (0.17663716146287139-0.16562727702376817j))
    print(lib.finger(ctf.to_nparray(t2b)) - (-0.51694318446604348+0.44507111765134028j))
    print(energy(mycc, t1b, t2b, eris) - -0.139737987402)
    mycc.kernel(eris=eris)

#    from pyscf.pbc.cc import kccsd_rhf
#    mycc1 = kccsd_rhf.KRCCSD(mf)
#    mycc1.diis = None
#    mycc1.verbose = 4
#    t1a, t2a = mycc1.update_amps(t1, t2, eris)
#    print(lib.finger(t1a) - (0.17663716146287139-0.16562727702376817j))
#    print(lib.finger(t2a) - (-0.51694318446604348+0.44507111765134028j))
#    emp2,t1,t2 = mycc1.init_amps(eris)
#    print(emp2 - -0.131341714277)
#    print(lib.finger(t2) - (-0.0576606747244+0.0336759095279j))
#    #mycc1.kernel(eris=eris)
#
#    t1b, t2b = mycc.update_amps(t1b, t2b, eris)
#    t1a, t2a = mycc1.update_amps(t1a, t2a, eris)
#    print(abs(t1a-t1b).max(), abs(t2a-t2b).max())
#
