import numpy as np
import itertools
from pyscf import lib
from pyscf.pbc import gto, scf, cc
from pyscf.pbc.lib import kpts_helper
from pyscf.lib import logger
import pyscf.pbc.tools.pbc as tools
import time
import pyscf
from pyscf.cc import ccsd
einsum = lib.einsum

def trans_21(eri):
    nk, d1, d2 = eri.shape
    dtype = eri.dtype
    tmp = np.zeros([nk,nk,d1,d1],dtype=dtype)
    tmp[np.diag_indices(nk)] = eri
    tmp = tmp.transpose(0,2,1,3).reshape(nk*d1,nk*d2)
    return tmp

def trans_41_amp(amp, kconserv):
    nk, _, _, d1, d2, d3, d4 = amp.shape
    dtype = amp.dtype
    tmp = np.zeros([nk,d1,nk,d2,nk,d3,nk,d4], dtype=dtype)
    for ki,kj,ka in itertools.product(range(nk), repeat=3):
        kb = kconserv[ki,ka,kj]
        tmp[ki,:,kj,:,ka,:,kb,:] = amp[ki,kj,ka]
    tmp = tmp.reshape(nk*d1,nk*d2,nk*d3,nk*d4)
    return tmp

def trans_41_eri(eri, kconserv):
    nk, _, _, d1, d2, d3, d4 = eri.shape
    dtype = eri.dtype
    tmp = np.zeros([nk,d1,nk,d3,nk,d2,nk,d4], dtype=dtype)
    for kp,kq,kr in itertools.product(range(nk), repeat=3):
        ks = kconserv[kp,kq,kr]
        tmp[kp,:,kq,:,kr,:,ks,:] = eri[kp,kr,kq].transpose(0,2,1,3)
    tmp = tmp.reshape(nk*d1,nk*d3,nk*d2,nk*d4)
    return tmp

def energy(cc, t1, t2, eris):
    nkpts = eris.nkpts
    e = 0.0 + 1j * 0.0
    e += 2*einsum('ia,ia', eris.fov, t1)
    tau = t2 + einsum('ia,jb->ijab', t1, t1)
    eris_ovov = 2*eris.ovov - eris.ovov.transpose(0,3,2,1)
    e += einsum('iajb, ijab', eris_ovov, tau)
    e /= nkpts
    if abs(e.imag) > 1e-4:
        logger.warn(cc, 'Non-zero imaginary part found in KRCCSD energy %s', e)
    return e.real

def update_amps(cc, t1, t2, eris):
    time0 = time1 = time.clock(), time.time()
    log = logger.Logger(cc.stdout, cc.verbose)
    nocc, nvir = t1.shape
    kconserv = cc.khelper.kconserv
    time1 = log.timer_debug1('intermediates', *time1)

    fov_temp = eris.fov
    foo_temp = eris.foo
    fvv_temp = eris.fvv

    tau = t2 + einsum('ic,ld->ilcd', t1, t1)

    eri_ovov = 2*eris.ovov - eris.ovov.transpose(0,3,2,1)
    Foo_temp = einsum('kcld,ilcd->ki',eri_ovov, tau)
    Foo_temp+= foo_temp
    Foo_temp[np.diag_indices(nocc)] -= eris.mo_e_o

    Fvv_temp =-einsum('kcld,klad->ac', eri_ovov, tau)
    Fvv_temp+= fvv_temp
    Fvv_temp[np.diag_indices(nvir)] -= eris.mo_e_v

    Fov_temp = einsum('kcld,ld->kc', eri_ovov, t1)
    Fov_temp+= fov_temp

    Loo_temp = Foo_temp + einsum('kc,ic->ki', fov_temp, t1)
    eri_ooov = 2*eris.ooov - eris.ooov.transpose(2,1,0,3)
    Loo_temp += einsum('kilc,lc->ki', eri_ooov, t1)

    Lvv_temp = Fvv_temp - einsum('kc,ka->ac', fov_temp, t1)
    eri_vvov = 2*eris.vvov - eris.vvov.transpose(0,3,2,1)
    Lvv_temp+= einsum('ackd,kd->ac', eri_vvov, t1)

    t1new = np.array(fov_temp).astype(t1.dtype).conj()

    t1new +=-2*einsum('kc,ka,ic->ia', fov_temp, t1, t1)

    t1new +=   einsum('ac,ic->ia', Fvv_temp, t1)
    t1new +=  -einsum('ki,ka->ia', Foo_temp, t1)

    t1new += 2*einsum('kc,kica->ia', Fov_temp, t2)
    t1new +=  -einsum('kc,ikca->ia', Fov_temp, t2)
    t1new +=   einsum('kc,ka,ic->ia', Fov_temp, t1, t1)

    t1new += 2*einsum('aikc,kc->ia', eris.voov, t1)
    t1new +=  -einsum('kiac,kc->ia', eris.oovv, t1)

    t1new += einsum('ackd,ikcd->ia', eri_vvov, tau)
    t1new +=-einsum('kilc,klac->ia', eri_ooov, tau)



    time1 = log.timer_debug1('t1', *time1)

    t2new = eris.ovov.transpose(0,2,1,3).conj()

    Woooo  = einsum('kilc,jc->klij', eris.ooov, t1)
    Woooo += einsum('ljkc,ic->klij', eris.ooov, t1)
    Woooo += einsum('kcld,ijcd->klij', eris.ovov, tau)
    Woooo += eris.oooo.transpose(0,2,1,3)
    t2new += einsum('klij,klab->ijab', Woooo, tau)
    time1 = log.timer_debug1('t2 oooo', *time1)

    Wvvvv = einsum('ackd,kb->abcd', eris.vvov, -t1)
    Wvvvv += Wvvvv.transpose(1,0,3,2)
    Wvvvv+= eris.vvvv.transpose(0,2,1,3)
    t2new += einsum('abcd,ijcd->ijab', Wvvvv, tau)
    time1 = log.timer_debug1('t2 vvvv', *time1)

    tmp  = einsum('ac,ijcb->ijab', Lvv_temp, t2)
    tmp -= einsum('ki,kjab->ijab', Loo_temp, t2)
    t2new += (tmp + tmp.transpose(1,0,3,2))

    tmp2  = einsum('kibc,ka->abic',eris.oovv, -t1)
    tmp2 += eris.vvov.transpose(3,1,2,0).conj()
    tmp = einsum('abic,jc->ijab',tmp2, t1)
    t2new += tmp + tmp.transpose(1,0,3,2)

    tmp2  = einsum('ckia,jc->akij', eris.voov.conj(),t1)
    tmp2 += eris.ooov.transpose(3,1,2,0).conj()
    tmp   = einsum('akij,kb->ijab', tmp2, t1)
    t2new -= tmp + tmp.transpose(1,0,3,2)


    Wvoov_temp  = einsum('adkc,id->akic', eris.vvov, t1)
    Wvoov_temp -= einsum('likc,la->akic', eris.ooov, t1)
    Wvoov_temp += eris.voov.conj().transpose(3,1,2,0)
    Wvoov_temp -= 0.5*einsum('ldkc,ilda->akic', eris.ovov, t2)
    Wvoov_temp -= einsum('ldkc,id,la->akic', eris.ovov, t1, t1)
    Wvoov_temp += .5*einsum('ldkc,ilad->akic', eri_ovov, t2)

    Wvovo_temp  = einsum('ackd,id->akci', eris.vvov, t1)
    Wvovo_temp -= einsum('kilc,la->akci', eris.ooov, t1)
    Wvovo_temp += eris.oovv.transpose(2,0,3,1)
    Wvovo_temp -= 0.5*einsum('lckd,ilda->akci', eris.ovov, t2)
    Wvovo_temp -= einsum('lckd,id,la->akci', eris.ovov, t1, t1)

    tmp  = 2*einsum('akic,kjcb->ijab', Wvoov_temp, t2)
    tmp -=   einsum('akci,kjcb->ijab', Wvovo_temp, t2)
    tmp -=   einsum('akic,kjbc->ijab', Wvoov_temp, t2)
    tmp -=   einsum('bkci,kjac->ijab', Wvovo_temp, t2)
    t2new += tmp + tmp.transpose(1,0,3,2)

    t1new *= eris.eia
    t2new *= eris.eijab

    time0 = log.timer_debug1('update t1 t2', *time0)

    return t1new, t2new

class _ERIS:
    def __init__(self, cc, mo_coeff=None, method='incore'):
        if method != 'incore': raise NotImplementedError
        from pyscf.pbc import df
        from pyscf.pbc import tools
        from pyscf.pbc.cc.ccsd import _adjust_occ

        log = logger.Logger(cc.stdout, cc.verbose)
        cput0 = (time.clock(), time.time())

        cell = cc._scf.cell
        kpts = cc.kpts
        nkpts = cc.nkpts
        nocc = cc.nocc
        nmo = cc.nmo
        nvir = nmo - nocc

        if mo_coeff is None:
            mo_coeff = np.asarray(cc.mo_coeff)
        dtype = mo_coeff[0].dtype

        self.mo_coeff = mo_coeff
        dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
        with lib.temporary_env(cc._scf, exxdiv=None):
            fockao = cc._scf.get_hcore() + cc._scf.get_veff(cell, dm)
        self.fock = np.asarray([reduce(np.dot, (mo.T.conj(), fockao[k], mo))
                                for k, mo in enumerate(mo_coeff)])

        self.mo_energy = [self.fock[k].diagonal().real for k in range(nkpts)]
        madelung = tools.madelung(cell, kpts)
        self.mo_energy = np.asarray([_adjust_occ(mo_e, nocc, -madelung)
                          for k, mo_e in enumerate(self.mo_energy)])
        fao2mo = cc._scf.with_df.ao2mo
        kconserv = cc.khelper.kconserv
        khelper = cc.khelper
        orbv = np.asarray(mo_coeff[:,:,nocc:], order='C')
        log.info('using incore ERI storage')
        oooo = np.empty((nkpts,nkpts,nkpts,nocc,nocc,nocc,nocc), dtype=dtype)
        ooov = np.empty((nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir), dtype=dtype)
        oovv = np.empty((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=dtype)
        ovov = np.empty((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=dtype)
        voov = np.empty((nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir), dtype=dtype)
        vovv = np.empty((nkpts,nkpts,nkpts,nvir,nocc,nvir,nvir), dtype=dtype)
        vvvv = cc._scf.with_df.ao2mo_7d(orbv, factor=1./nkpts).transpose(0,2,1,3,5,4,6)

        for (ikp,ikq,ikr) in khelper.symm_map.keys():
            iks = kconserv[ikp,ikq,ikr]
            eri_kpt = fao2mo((mo_coeff[ikp],mo_coeff[ikq],mo_coeff[ikr],mo_coeff[iks]),
                             (kpts[ikp],kpts[ikq],kpts[ikr],kpts[iks]), compact=False)
            if dtype == np.float: eri_kpt = eri_kpt.real
            eri_kpt = eri_kpt.reshape(nmo, nmo, nmo, nmo)
            for (kp, kq, kr) in khelper.symm_map[(ikp, ikq, ikr)]:
                eri_kpt_symm = khelper.transform_symm(eri_kpt, kp, kq, kr).transpose(0, 2, 1, 3)
                oooo[kp, kr, kq] = eri_kpt_symm[:nocc, :nocc, :nocc, :nocc] / nkpts
                ooov[kp, kr, kq] = eri_kpt_symm[:nocc, :nocc, :nocc, nocc:] / nkpts
                oovv[kp, kr, kq] = eri_kpt_symm[:nocc, :nocc, nocc:, nocc:] / nkpts
                ovov[kp, kr, kq] = eri_kpt_symm[:nocc, nocc:, :nocc, nocc:] / nkpts
                voov[kp, kr, kq] = eri_kpt_symm[nocc:, :nocc, :nocc, nocc:] / nkpts
                vovv[kp, kr, kq] = eri_kpt_symm[nocc:, :nocc, nocc:, nocc:] / nkpts

        self.dtype = dtype
        log.timer('CCSD integral transformation', *cput0)

        foo = self.fock[:,:nocc, :nocc]
        fov = self.fock[:,:nocc, nocc:]
        fvv = self.fock[:,nocc:, nocc:]
        dijab = np.zeros([nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir])
        eia = self.mo_energy[:,:nocc].reshape(nkpts, 1, nocc, 1) - self.mo_energy[:,nocc:].reshape(1, nkpts, 1, nvir)
        dia = 1.0/eia[np.diag_indices(nkpts)]
        for ki,kj,ka in itertools.product(range(nkpts), repeat=3):
            kb = kconserv[ki,ka,kj]
            dijab[ki,kj,ka] = 1.0/(eia[ki,ka].reshape(nocc,1,nvir,1) + eia[kj,kb].reshape(1,nocc,1,nvir))

        self.kconserv = kconserv
        self.nkpts, self.nocc, self.nvir = nkpts, nocc, nvir
        self.fock = trans_21(self.fock)
        self.foo = trans_21(foo)
        self.fov = trans_21(fov)
        self.fvv = trans_21(fvv)
        self.mo_e_o = self.mo_energy[:,:nocc].reshape(-1)
        self.mo_e_v = self.mo_energy[:,nocc:].reshape(-1)
        self.oooo = trans_41_eri(oooo, kconserv)
        self.ooov = trans_41_eri(ooov, kconserv)
        self.ovov = trans_41_eri(oovv, kconserv)
        self.oovv = trans_41_eri(ovov, kconserv)
        self.voov = trans_41_eri(voov, kconserv)
        self.vvov = trans_41_eri(vovv, kconserv)
        self.vvvv = trans_41_eri(vvvv, kconserv)
        self.eia = trans_21(dia)
        self.eijab = trans_41_amp(dijab, kconserv)

def get_nocc(cc):
    return np.count_nonzero(cc.mo_occ[0])

def get_nmo(cc):
    return len(cc.mo_coeff[0])

class CCSD(pyscf.cc.ccsd.CCSD):

    def __init__(self, kmf, frozen=0, mo_coeff=None, mo_occ=None):
        if frozen!=0:
            raise NotImplementedError
        self._scf = kmf
        self.kpts = kmf.kpts
        self.nkpts = len(kmf.kpts)
        self.kconserv = tools.get_kconserv(kmf.cell, kmf.kpts)
        self.diis = None
        self.khelper = kpts_helper.KptsHelper(kmf.cell, kmf.kpts)
        self.verbose = kmf.verbose
        if mo_coeff is None: mo_coeff = kmf.mo_coeff
        if mo_occ is None: mo_occ = kmf.mo_occ
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.mo_energy = kmf.mo_energy
        self.max_cycle = kmf.max_cycle
        self.conv_tol = kmf.conv_tol

    nocc = property(get_nocc)
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    nmo = property(get_nmo)
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    @property
    def e_tot(self):
        return self.e_corr + self._scf.e_tot

    energy = energy
    update_amps = update_amps

    def ao2mo(self, mo_coeff=None):
        return _ERIS(self)

    def init_amps(self, eris=None):
        time0 = time.clock(), time.time()
        if eris is None: eris = self.ao2mo()
        nkpts, nocc, nmo = self.nkpts, self.nocc, self.nmo
        nvir = nmo - nocc

        t1 = np.zeros([nkpts*nocc, nkpts*nvir], dtype=eris.dtype)
        t2 = eris.ovov.transpose(0,2,1,3).conj() * eris.eijab
        eris_ovov = 2 * eris.ovov - eris.ovov.transpose(0,3,2,1)
        emp2 = einsum('ijab,iajb', t2, eris_ovov) / nkpts
        self.emp2 = emp2.real
        logger.info(self, 'Init t2, MP2 energy (with fock eigenvalue shift) =%.15g', self.emp2)
        logger.timer(self, 'init mp2', *time0)
        return self.emp2, t1, t2

    def kernel(self):
        eris = self.ao2mo(self.mo_coeff)
        log = logger.new_logger(self, self.verbose)
        cput1 = cput0 = (time.clock(), time.time())

        max_cycle = self.max_cycle
        tol = self.conv_tol
        tolnormt = self.conv_tol_normt

        t1, t2 = self.init_amps(eris)[1:]
        nocc, nvir = t1.shape
        eold = 0
        eccsd = 0

        conv = False
        for istep in range(self.max_cycle):
            t1new, t2new = self.update_amps(t1, t2, eris)
            normt = np.linalg.norm(t1new-t1) + np.linalg.norm(t2new-t2)
            t1, t2 = t1new, t2new
            t1new = t2new = None
            eold, eccsd = eccsd, self.energy(t1, t2, eris)

            log.info('istep = %d  E(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                    istep, eccsd, eccsd - eold, normt)
            cput1 = log.timer('CCSD iter', *cput1)
            if abs(eccsd-eold) < tol and normt < tolnormt:
                conv = True
                break
        log.timer('CCSD', *cput0)
        self.converged = conv
        self.e_corr = eccsd
        self.t1 = t1
        self.t2 = t2
        if self.converged:
            logger.note(self, 'RCCSD converged')
        else:
            logger.note(self, 'RCCSD not converged')
        logger.note(self, 'E_tot = %.16g E_corr = %.16g', self.e_tot, self.e_corr)
        return eccsd, t1, t2

if __name__=='__main__':
    cell = gto.Cell()
    cell.atom = '''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = {'C': [[0, (0.8, 1.0)],
                        [1, (1.0, 1.0)]]}
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 4
    cell.build()

    kmf = scf.KRHF(cell, kpts=cell.make_kpts([1, 1, 3]), exxdiv=None).density_fit()
    #kmf.with_df._cderi_to_save = 'dmd.int'
    kmf.with_df._cderi = 'dmd.int'
    #kmf.chkfile = 'dmd.chk'
    kmf.__dict__.update(scf.chkfile.load('dmd.chk','scf'))
    #kmf.kernel()
    mycc = CCSD(kmf)
    ecc, t1, t2 = mycc.kernel()

    print(ecc - -0.2901165510449186)
    print(lib.finger(t1)- (-0.03821719933614061-0.015788931255275286j))
    print(lib.finger(t2)- (-0.3467595606706193-0.041921860837675406j))
