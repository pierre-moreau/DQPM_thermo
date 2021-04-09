import numpy as np
import vegas
import scipy
# parametrization of lattice data & susceptibilities
from EoS_HRG.fit_lattice import param, list_chi, BQS
# import from __init__.py
from . import *

########################################################################
def DQPM_s(T,muB,muQ,muS,**kwargs):
    """
    calculate the entropy density of the DQPM
    """
    # if input is a single temperature value T
    if(isinstance(T,float)):
        # integration limits
        ommin = -15.*T
        ommax = 15.*T
        pmin = 0.
        pmax = 15.*T

        # on-shell contribution
        @vegas.batchintegrand
        def int_s_on_all(x):
            return dict(zip(list_parton,[parton.int_sq0(x[:,0],T,muB,muQ,muS,**kwargs) for parton in list_parton]))

        integ = vegas.Integrator([[pmin, pmax]])
        result_s = integ(int_s_on_all, nitn=10, neval=2500)
        s = sum([result_s[parton].mean for parton in list_parton])/T**3.

        if(offshell):
            # off-shell contribution
            @vegas.batchintegrand
            def int_s_all(x):
                return dict(zip(list_parton,[parton.int_sqint(x[:,0],x[:,1],T,muB,muQ,muS,**kwargs) for parton in list_parton]))

            integ = vegas.Integrator([[ommin, ommax],[pmin, pmax]])
            result_s = integ(int_s_all, nitn=10, neval=15000)
            s += sum([result_s[parton].mean for parton in list_parton])/T**3.

    # if the input is a list of temperature values
    elif(isinstance(T,np.ndarray) or isinstance(T,list)):
        s = np.zeros_like(T)
        for i,xT in enumerate(T):
            # see if arrays are also given for chemical potentials
            try:
                xmuB = muB[i]
            except:
                xmuB = muB
            try:
                xmuQ = muQ[i]
            except:
                xmuQ = muQ
            try:
                xmuS = muS[i]
            except:
                xmuS = muS
            s[i] = DQPM_s(xT,xmuB,xmuQ,xmuS,**kwargs)

    else:
        raise Exception('Problem with input')
                
    return s

########################################################################
def DQPM_n(T,muB,muQ,muS,**kwargs):
    """
    calculate the charge densities of the DQPM
    """
    if(len(list_quarks_all)==0):
        return {'n_B':0., 'n_Q':0., 'n_S':0.}
    # if input is a single temperature value T
    if(isinstance(T,float)):
        # integration limits
        ommin = -15.*T
        ommax = 15.*T
        pmin = 0.
        pmax = 15.*T

        # on-shell contribution
        @vegas.batchintegrand
        def int_n_on_all(x):
            return dict(zip(list_parton,[parton.int_nq0(x[:,0],T,muB,muQ,muS,**kwargs) for parton in list_quarks_all]))

        integ = vegas.Integrator([[pmin, pmax]])
        result_n = integ(int_n_on_all, nitn=10, neval=1500)
        nB = sum([parton.Bcharge*result_n[parton].mean for parton in list_quarks_all])/T**3.
        nQ = sum([parton.Qcharge*result_n[parton].mean for parton in list_quarks_all])/T**3.
        nS = sum([parton.Scharge*result_n[parton].mean for parton in list_quarks_all])/T**3.

        if(offshell):
            # off-shell contribution
            @vegas.batchintegrand
            def int_n_all(x):
                return dict(zip(list_parton,[parton.int_nqint(x[:,0],x[:,1],T,muB,muQ,muS,**kwargs) for parton in list_quarks_all]))

            integ = vegas.Integrator([[ommin, ommax],[pmin, pmax]])
            result_n = integ(int_n_all, nitn=10, neval=7000)
            nB += sum([parton.Bcharge*result_n[parton].mean for parton in list_quarks_all])/T**3.
            nQ += sum([parton.Qcharge*result_n[parton].mean for parton in list_quarks_all])/T**3.
            nS += sum([parton.Scharge*result_n[parton].mean for parton in list_quarks_all])/T**3.

    # if the input is a list of temperature values
    elif(isinstance(T,np.ndarray) or isinstance(T,list)):
        nB = np.zeros_like(T)
        nQ = np.zeros_like(T)
        nS = np.zeros_like(T)

        for i,xT in enumerate(T):
            # see if arrays are also given for chemical potentials
            try:
                xmuB = muB[i]
            except:
                xmuB = muB
            try:
                xmuQ = muQ[i]
            except:
                xmuQ = muQ
            try:
                xmuS = muS[i]
            except:
                xmuS = muS
            result = DQPM_n(xT,xmuB,xmuQ,xmuS,**kwargs)
            nB[i] = result['n_B']
            nQ[i] = result['n_Q']
            nS[i] = result['n_S']

    else:
        raise Exception('Problem with input')
                
    return {'n_B':nB, 'n_Q':nQ, 'n_S':nS}

########################################################################
# precalculate entropy density at mu=0 to speed the calculation of the pressure
time0 = time.time()
muBmax = 0.6
T0 = Tc(muBmax)
xtemp = np.arange(T0,1.,0.01)
data_DQPM_s0 = DQPM_s(xtemp,0.,0.,0.)
# Interpolation of the entropy density s/T^3 at mu=0
DQPM_s0 = create_spline(xtemp, data_DQPM_s0)
print(f's0(T) calculated in {time.time()-time0}s')

########################################################################
def DQPM_P(T,muB,muQ,muS,**kwargs):
    """
    calculate the pressure of the DQPM
    """

    # if input is a single temperature value T
    if(isinstance(T,float)):
        # integration limits
        ommin = -15.*T
        ommax = 15.*T
        pmin = 0.
        pmax = 15.*T

        # integration constant at P0(Tc0)
        if(T>Tc(muBmax) and T>Tc(muB)):
            T0 = Tc(muB)
        else:
            T0 = T

        # integration constant for P at T0
        if(Nf==0):
            p_T0 = fit_SU3(T0)['P']*(T0)**4.
        else:
            p_T0 = param(T0,0.,0.,0.)['P']*(T0)**4.

        # mu-independant part of the pressure
        p = p_T0/T**4.
        if(T>Tc(muBmax) and T>Tc(muB)):
            p += scipy.integrate.quad(lambda xT: DQPM_s0(xT)*xT**3.,T0,T,epsrel=0.01)[0]/T**4.

        # mu-dependent part of the pressure
        if(muB!=0. or muQ!=0. or muS!=0.):

            # precalculate for the integral over mu
            xgam = np.linspace(0.,1.,11)
            nq_gamma = np.zeros_like(xgam)

            # evaluation of the integrand for each value of gamma
            for igam,gam in enumerate(xgam):
                if(gam==0):
                    # nq is automatically 0 when gamma=0
                    continue

                if(not offshell):
                    # on-shell densities
                    @vegas.batchintegrand
                    def int_DP_on_all(x):
                        return dict(zip(list_parton,[parton.int_nq0(x[:,0],T,gam*muB,gam*muQ,gam*muS) for parton in list_parton]))

                    integ = vegas.Integrator([[pmin, pmax]])
                    result_DP = integ(int_DP_on_all, nitn=10, neval=2000)

                elif(offshell):
                    # off-shell densities
                    @vegas.batchintegrand
                    def int_DP_all(x):
                        return dict(zip(list_parton,[parton.int_nq(x[:,0],x[:,1],T,gam*muB,gam*muQ,gam*muS) for parton in list_parton]))

                    integ = vegas.Integrator([[ommin, ommax],[pmin, pmax]])
                    result_DP = integ(int_DP_all, nitn=10, neval=5000)
                    
                nq_gamma[igam] = sum([(parton.Bcharge*muB+parton.Qcharge*muQ+parton.Scharge*muS)*result_DP[parton].mean for parton in list_parton])
            
            # create spline
            #spline_nq_gamma = scipy.interpolate.splrep(xgam, nq_gamma)
            # mu-dependant part of the pressure
            # integration over gamma from [0,1]
            p += scipy.integrate.quad(create_spline(xgam, nq_gamma),0,1,epsrel=0.01)[0]/T**4.
            #p += scipy.integrate.quad(lambda gam: scipy.interpolate.splev(gam, spline_nq_gamma, der=0),0,1,epsrel=0.001)[0]/T**4.

    # if the input is a list of temperature values
    elif(isinstance(T,np.ndarray) or isinstance(T,list)):
        p = np.zeros_like(T)
        for i,xT in enumerate(T):
            # see if arrays are also given for chemical potentials
            try:
                xmuB = muB[i]
            except:
                xmuB = muB
            try:
                xmuQ = muQ[i]
            except:
                xmuQ = muQ
            try:
                xmuS = muS[i]
            except:
                xmuS = muS
            p[i] = DQPM_P(xT,xmuB,xmuQ,xmuS)
    else:
        raise Exception('Problem with input')
                
    return p

########################################################################
def DQPM_thermo(T,muB,muQ,muS,**kwargs):
    """
    calculate the thermodynamics of the DQPM
    """

    p = DQPM_P(T,muB,muQ,muS,**kwargs)
    s = DQPM_s(T,muB,muQ,muS,**kwargs)
    result_n = DQPM_n(T,muB,muQ,muS,**kwargs)
    nB = result_n['n_B']
    nQ = result_n['n_Q']
    nS = result_n['n_S']
    e = s-p+(muB/T)*nB+(muQ/T)*nQ+(muS/T)*nS
    cs2 = speed_sound(T,muB,muQ,muS,**kwargs)
                
    return {'n_B':nB, 'n_Q':nQ, 'n_S':nS, 's':s, 'P':p, 'e':e, 'I':e-3*p, 'cs^2':cs2}

########################################################################
def DQPM_chi(T):
    """
    calculate the susceptibilities of the DQPM
    """

    # if input is a single temperature value T
    if(isinstance(T,float)):

        print(f'     T = {T} GeV')
        ########################################################################
        # evaluate EoS at fixed T before evaluating all the derivatives
        ########################################################################
        order = 5 # number of points in the derivative
        NmuB = order
        NmuQ = order
        NmuS = order

        dx = 0.1*T
        muBmax = (order-1)/2*dx
        muQmax = (order-1)/2*dx
        muSmax = (order-1)/2*dx
        xmuB = np.linspace(-muBmax,muBmax,NmuB,endpoint=True)
        xmuQ = np.linspace(-muQmax,muQmax,NmuQ,endpoint=True)
        xmuS = np.linspace(-muSmax,muSmax,NmuS,endpoint=True)

        time0 = time.time()
        # store data for P
        data_EoS = np.zeros((NmuB,NmuQ,NmuS))
        for imuB in range(NmuB):
            for imuQ in range(NmuQ):
                for imuS in range(NmuS):
                    data_EoS[imuB,imuQ,imuS] = DQPM_P(T,xmuB[imuB],xmuQ[imuQ],xmuS[imuS])

        print(f'     data grid calculated in {(time.time()-time0)/60.}min')

        chi = np.zeros((len(list_chi)))
        xdata = (xmuB,xmuQ,xmuS)
        # construct the function of linear interpolation
        def interpP(muB,muQ,muS):
            f_interp = scipy.interpolate.RegularGridInterpolator(xdata,data_EoS[:,:,:])
            return f_interp([[muB,muQ,muS]])[0]
        
        chi = np.zeros(len(list_chi))
        for ichi,xchi in enumerate(list_chi):
            ii = BQS[xchi]['B']
            jj = BQS[xchi]['Q']
            kk = BQS[xchi]['S']
            chi[ichi] = der_mu(interpP,[ii,jj,kk],dx=dx,npoints=order)*T**(ii+jj+kk)

    # if the input is a list of temperature values
    elif(isinstance(T,np.ndarray) or isinstance(T,list)):
        chi = np.zeros((len(list_chi),len(T)))
        for iT,xT in enumerate(T):
            chi[:,iT] = DQPM_chi(xT)['chi']
    else:
        raise Exception('Problem with input')            
    
    return {'chi': chi}

########################################################################
def DQPM_Tmunu(T,muB,muQ,muS):
    """
    calculate the energy momentum tensor of the DQPM
    """
    # if input is a single temperature value T
    if(isinstance(T,float)):

        # integration limits
        ommin = 0.
        ommax = 15.*T
        pmin = 0.
        pmax = 15.*T
        integ = vegas.Integrator([[ommin, ommax],[pmin, pmax]])

        @vegas.batchintegrand
        def int_Tmunuep_all(x):
            return dict(zip(list_parton,[parton.int_Tmunu_e(x[:,0],x[:,1],T,muB,muQ,muS)*np.heaviside(((x[:,0])**2.-(x[:,1])**2.),0.5) for parton in list_parton]))

        @vegas.batchintegrand
        def int_Tmunuem_all(x):
            return dict(zip(list_parton,[parton.int_Tmunu_e(x[:,0],x[:,1],T,muB,muQ,muS)*np.heaviside(-((x[:,0])**2.-(x[:,1])**2.),0.5) for parton in list_parton]))

        @vegas.batchintegrand
        def int_TmunuPp_all(x):
            return dict(zip(list_parton,[parton.int_Tmunu_P(x[:,0],x[:,1],T,muB,muQ,muS)*np.heaviside(((x[:,0])**2.-(x[:,1])**2.),0.5) for parton in list_parton]))

        @vegas.batchintegrand
        def int_TmunuPm_all(x):
            return dict(zip(list_parton,[parton.int_Tmunu_P(x[:,0],x[:,1],T,muB,muQ,muS)*np.heaviside(-((x[:,0])**2.-(x[:,1])**2.),0.5) for parton in list_parton]))

        result_ep = integ(int_Tmunuep_all, nitn=10, neval=10000)
        result_em = integ(int_Tmunuem_all, nitn=10, neval=10000)
        result_Pp = integ(int_TmunuPp_all, nitn=10, neval=10000)
        result_Pm = integ(int_TmunuPm_all, nitn=10, neval=10000)

        pp = sum([result_Pp[parton].mean for parton in list_parton])/T**4.
        pm = sum([result_Pm[parton].mean for parton in list_parton])/T**4.
        p = pp + pm
        ep = sum([result_ep[parton].mean for parton in list_parton])/T**4.
        em = sum([result_em[parton].mean for parton in list_parton])/T**4.
        e = ep + em

    # if the input is a list of temperature values
    elif(isinstance(T,np.ndarray) or isinstance(T,list)):
        pp = np.zeros_like(T)
        pm = np.zeros_like(T)
        p = np.zeros_like(T)
        ep = np.zeros_like(T)
        em = np.zeros_like(T)
        e = np.zeros_like(T)
        for i,xT in enumerate(T):
            result = DQPM_Tmunu(xT,muB,muQ,muS)
            p[i] = result['P']
            pp[i] = result['Pp']
            pm[i] = result['Pm']
            e[i] = result['e']
            ep[i] = result['ep']
            em[i] = result['em']
    else:
        raise Exception('Problem with input')
                
    return {'P':p, 'Pp':pp, 'Pm':pm, 'e':e, 'ep':ep, 'em':em}

########################################################################
def speed_sound(T,muB,muQ,muS,**kwargs):
    """
    Calculation of the speed of sound squared
    """
    # if input is a single temperature value T
    if(isinstance(T,float)):
        xs = DQPM_s(T,muB,muQ,muS,**kwargs)*T**3 # entropy density

        xmu = [muB,muQ,muS] # array of chemical potential

        # derivative of entropy and densities wrt T
        dsdT = scipy.misc.derivative(lambda xT: DQPM_s(xT,muB,muQ,muS,**kwargs)*xT**3, T, dx=0.01, n=1, order=3)
        if(muB!=0. and muQ!=0. and muS!=0.):
            dndT = scipy.misc.derivative(lambda xT: np.array([n*xT**3 for n in DQPM_n(xT,muB,muQ,muS,**kwargs).values()]), T, dx=0.01, n=1, order=3)
        else:
            dndT = np.zeros(len(xmu))
            
        cs2 = xs/(T*dsdT+sum([xmu[imu]*dndT[imu] for imu in range(len(xmu))]))
        
    # if the input is a list of temperature values
    elif(isinstance(T,np.ndarray) or isinstance(T,list)):
        cs2 = np.zeros_like(T)
        for i,xT in enumerate(T):
            # see if arrays are also given for chemical potentials
            try:
                xmuB = muB[i]
            except:
                xmuB = muB
            try:
                xmuQ = muQ[i]
            except:
                xmuQ = muQ
            try:
                xmuS = muS[i]
            except:
                xmuS = muS
            cs2[i] = speed_sound(xT,xmuB,xmuQ,xmuS,**kwargs)

    else:
        raise Exception('Problem with input')

    return cs2

########################################################################
def DQPM_visc(T,muB,muQ,muS,**kwargs):
    """
    calculate the viscosities
    """
    # if input is a single temperature value T
    if(isinstance(T,float)):
        # integration limits
        pmin = 0.
        pmax = 15.*T

        # speed of sound squared
        cs2 = speed_sound(T,muB,muQ,muS,**kwargs)

        # on-shell
        @vegas.batchintegrand
        def int_eta_on_all(x):
            return dict(zip(list_parton,[parton.int_eta0(x[:,0],T,muB,muQ,muS,**kwargs) for parton in list_parton]))

        @vegas.batchintegrand
        def int_zeta_on_all(x):
            return dict(zip(list_parton,[parton.int_zeta0(x[:,0],T,muB,muQ,muS,cs2,**kwargs) for parton in list_parton]))

        integ = vegas.Integrator([[pmin, pmax]])
        result_eta = integ(int_eta_on_all, nitn=10, neval=1000)

        integ = vegas.Integrator([[pmin, pmax]])
        result_zeta = integ(int_zeta_on_all, nitn=10, neval=1000)

        s = DQPM_s(T,muB,muQ,muS,**kwargs)
        eta = sum([result_eta[parton].mean for parton in list_parton])/(s*T**3)
        zeta = sum([result_zeta[parton].mean for parton in list_parton])/(s*T**3)

    # if the input is a list of temperature values
    elif(isinstance(T,np.ndarray) or isinstance(T,list)):
        eta = np.zeros_like(T)
        zeta = np.zeros_like(T)
        for i,xT in enumerate(T):
            eta[i],zeta[i] = DQPM_visc(xT,muB,muQ,muS,**kwargs)

    else:
        raise Exception('Problem with input')
                
    return eta,zeta