import ctf
import time
import sys

def update_amps(t1, t2, eris):
    time0 = time.clock(), time.time()
    nocc, nvir = t1.shape
    nov = nocc*nvir

    t1new = ctf.tensor(t1.shape)
    #tau = t2 + ctf.einsum('ia,jb->ijab', t1, t1)
    #t2new = ctf.einsum('acbd,ijab->ijcd', eris.vvvv, tau)
    t2new = ctf.einsum('acbd,ijab->ijcd', eris.vvvv, t2)
    t2new += ctf.einsum('acbd,ia,jb->ijcd', eris.vvvv, t1, t1)
    t2new *= .5

    t1new += eris.fov
    foo = eris.foo + .5 * ctf.einsum('ia,ja->ij', eris.fov, t1)
    fvv = eris.fvv - .5 * ctf.einsum('ia,ib->ab', t1, eris.fov)

    foo += ctf.einsum('kc,jikc->ij',  2*t1, eris.ooov)
    foo += ctf.einsum('kc,jkic->ij', -1*t1, eris.ooov)
    woooo = ctf.einsum('ijka,la->ijkl', eris.ooov, t1)
    woooo = woooo + woooo.transpose(2,3,0,1)
    woooo += eris.oooo

    #for p0, p1 in lib.prange(0, nvir, vblk):
    fvv += ctf.einsum('kc,kcba->ab',  2*t1, eris.ovvv)
    fvv += ctf.einsum('kc,kbca->ab', -1*t1, eris.ovvv)

    woVoV = ctf.einsum('ijka,kb->ijab', eris.ooov, t1)
    woVoV -= ctf.einsum('jc,icab->ijab', t1, eris.ovvv)

    #:eris_ovvv = ctf.einsum('ial,bcl->iabc', ovL, vvL)
    #:tmp = ctf.einsum('ijcd,kcdb->kijb', tau, eris_ovvv)
    #:t2new += ctf.einsum('ka,kijb->jiba', -t1, tmp)
    #tmp = ctf.einsum('ijcd,kcdb->kijb', tau, eris.ovvv)
    tmp = ctf.einsum('ijcd,kcdb->kijb', t2, eris.ovvv)
    tmp += ctf.einsum('ic,jd,kcdb->kijb', t1, t1, eris.ovvv)
    t2new -= ctf.einsum('ka,kijb->jiba', t1, tmp)

    wOVov  = ctf.einsum('ikjb,ka->ijba', eris.ooov, -1*t1)
    wOVov += ctf.einsum('jc,iabc->jiab', t1, eris.ovvv)
    t2new += wOVov.transpose(0,1,3,2)

    theta = t2.transpose(0,1,3,2) * 2 - t2
    t1new += ctf.einsum('ijcb,jcba->ia', theta, eris.ovvv)

    t2new += eris.ovov.transpose(0,2,1,3) * .5

    fov = eris.fov.copy()
    fov += ctf.einsum('kc,iakc->ia', t1, eris.ovov) * 2
    fov -= ctf.einsum('kc,icka->ia', t1, eris.ovov)

    t1new += ctf.einsum('jb,jiab->ia', fov, theta)
    t1new -= ctf.einsum('kijb,kjba->ia', eris.ooov, theta)
    theta = None

    wOVov += eris.ovov.transpose(0,2,3,1)
    wOVov -= .5 * ctf.einsum('icka,jkbc->jiab', eris.ovov, t2)
    tau = t2.transpose(0,2,1,3) * 2 - t2.transpose(0,3,1,2)
    #tau -= ctf.einsum('ia,jb->ibja', t1*2, t1)
    #wOVov += .5 * ctf.einsum('iakc,jbkc->jiab', eris.ovov, tau)

    wOVov += .5 * ctf.einsum('iakc,jc,kb->jiab', eris.ovov, t1*2, t1)
    wOVov += .5 * ctf.einsum('iakc,jbkc->jiab', eris.ovov, tau)
    theta = t2 * 2 - t2.transpose(0,1,3,2)
    t2new += ctf.einsum('ikac,jkcb->jiba', theta, wOVov)
    tau = theta = wOVov = None

    tau = ctf.einsum('ia,jb->ijab', t1*.5, t1) + t2
    theta = tau.transpose(0,1,3,2)*2 - tau
    fvv -= ctf.einsum('ijca,ibjc->ab', theta, eris.ovov)
    foo += ctf.einsum('iakb,jkba->ij', eris.ovov, theta)
    tau = theta = None

    tmp = ctf.einsum('ic,jkbc->jibk', t1, eris.oovv)
    t2new -= ctf.einsum('ka,jibk->jiab', t1, tmp)
    tmp = ctf.einsum('ic,jbkc->jibk', t1, eris.ovov)
    t2new -= ctf.einsum('ka,jibk->jiba', t1, tmp)
    tmp = None

    t1new += ctf.einsum('jb,iajb->ia',  2*t1, eris.ovov)
    t1new += ctf.einsum('jb,ijba->ia', -1*t1, eris.oovv)

    woVoV -= eris.oovv

    #tau = t2 + ctf.einsum('ia,jb->ijab', t1, t1)
    #woooo += ctf.einsum('iajb,klab->ikjl', eris.ovov, tau)
    #t2new += .5 * ctf.einsum('kilj,klab->ijab', woooo, tau)
    #tau -= t2 * .5
    woooo += ctf.einsum('iajb,klab->ikjl', eris.ovov, t2)
    woooo += ctf.einsum('iajb,ka,lb->ikjl', eris.ovov, t1, t1)
    t2new += .5 * ctf.einsum('kilj,klab->ijab', woooo, t2)
    t2new += .5 * ctf.einsum('kilj,ka,lb->ijab', woooo, t1, t1)
    #tau -= t2 * .5
    woVoV += ctf.einsum('jkca,ickb->ijba', .5*t2, eris.ovov)
    woVoV += ctf.einsum('jc,ka,ickb->ijba', t1, t1, eris.ovov)
    t2new += ctf.einsum('kicb,kjac->ijab', woVoV, t2)
    t2new += ctf.einsum('kica,kjcb->ijab', woVoV, t2)
    woooo = tau = woVoV = None

    ft_ij = foo + ctf.einsum('ja,ia->ij', .5*t1, fov)
    ft_ab = fvv - ctf.einsum('ia,ib->ab', .5*t1, fov)
    t2new += ctf.einsum('ijac,bc->ijab', t2, ft_ab)
    t2new -= ctf.einsum('ki,kjab->ijab', ft_ij, t2)

    mo_e = ctf.tensor([t1.shape[0]+t1.shape[1]])
    ctf.random.seed(42)
    mo_e.fill_random(0.,1.)
    eia = mo_e[:nocc].reshape(nocc,1) - mo_e[nocc:].reshape(1,nvir)
    t1new += ctf.einsum('ib,ab->ia', t1, fvv)
    t1new -= ctf.einsum('ja,ji->ia', t1, foo)
    t1new /= eia

    t2new = t2new + t2new.transpose(1,0,3,2)
    dijab = eia.reshape(nocc,1,nvir,1) + eia.reshape(1,nocc,1,nvir)
    t2new /= dijab

    return t1new, t2new

class integrals:
    def __init__(self):
        self.ovvv = None
        self.oovv = None
        self.oooo = None
        self.ooov = None
        self.vvvv = None
        self.ovov = None
        self.foo = None
        self.fvv = None
        self.fov = None

if __name__ == '__main__':
    assert(len(sys.argv)<=4)
    assert(len(sys.argv)>=3)
    nocc = int(sys.argv[1])
    nvir = int(sys.argv[2])
    cutoff = None
    if (len(sys.argv)>3):
        cutoff = float(sys.argv[3])

    eris = integrals()
    NS = ctf.SYM.NS
    SY = ctf.SYM.SY
    eris.oovv = ctf.tensor([nocc,nocc,nvir,nvir], sym=[NS,NS,SY,NS])
    eris.ovvv = ctf.tensor([nocc,nvir,nvir,nvir], sym=[NS,NS,SY,NS])
    eris.vvvv = ctf.tensor([nvir,nvir,nvir,nvir], sym=[SY,NS,SY,NS])
    eris.ovov = ctf.tensor([nocc,nvir,nocc,nvir], sym=[NS,NS,NS,NS])
    eris.ooov = ctf.tensor([nocc,nocc,nocc,nvir], sym=[NS,NS,NS,NS])
    eris.oooo = ctf.tensor([nocc,nocc,nocc,nocc], sym=[NS,NS,NS,NS])
    eris.fvv = ctf.tensor([nvir,nvir]);
    eris.fov = ctf.tensor([nocc,nvir]);
    eris.foo = ctf.tensor([nocc,nocc]);

    for e in [eris.ovvv, eris.oovv, eris.oooo, eris.ooov, eris.vvvv, eris.ovov, eris.fov, eris.fvv, eris.foo]:
        e.fill_random(0.,1.)

    if cutoff != None:
        print("Using cutoff",cutoff)
        eris.ovvv = eris.ovvv.sparsify(cutoff)
        eris.oovv = eris.oovv.sparsify(cutoff)
        eris.oooo = eris.oooo.sparsify(cutoff)
        eris.ooov = eris.ooov.sparsify(cutoff)
        eris.vvvv = eris.vvvv.sparsify(cutoff)
        eris.ovov = eris.ovov.sparsify(cutoff)
        if (ctf.comm().rank() == 0):
            for e in [eris.ovvv, eris.oovv, eris.oooo, eris.ooov, eris.vvvv, eris.ovov]:
                print "For integral tensor with shape", e.shape,"symmetry",e.sym,"number of nonzeros with cutoff", cutoff, "is ", (int(10000000*e.nnz_tot/e.size))/100000., "%"
          
    t1 = ctf.zeros([nocc,nvir])
    t2 = ctf.zeros([nocc,nocc,nvir,nvir])
    t2.fill_random(0.,1.)
    if (cutoff != None):
        t2 = t2.sparsify(1-(1-cutoff)*10)
        if (ctf.comm().rank() == 0):
            print "For amplitude tensor with shape", t2.shape,"symmetry",t2.sym,"number of nonzeros with cutoff", 1-(1-cutoff)*10, "is ", (int(10000000*t2.nnz_tot/t2.size))/100000., "%"

    start = time.time()
    [t1new, t2new] = update_amps(t1,t2,eris)
    end = time.time()
    t1norm = ctf.vecnorm(t1new)
    t2norm = ctf.vecnorm(t2new)
    if (ctf.comm().rank() == 0):
        print "t1 norm is",t1norm,"t2 norm is",t2norm
        print "CCSD iteration time was", end - start, "sec"
