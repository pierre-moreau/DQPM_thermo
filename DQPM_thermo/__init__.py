"""
To calculate and plot the thermodynamics of the Dynamical QuasiParticle Model (DQPM)
"""

__version__ = '1.0.0'

import os
from math import pi
import time
import numpy as np
import scipy
import pandas as pd
# parametrization of lattice data & susceptibilities
from EoS_HRG.fit_lattice import param, Tc_lattice, Tc_lattice_muBoT, SB_lim

# directory where the file is located
dir_path = os.path.dirname(os.path.realpath(__file__))

########################################################################
# parameters
Nc = 3 # number of colors
Nf = 3 # number of flavors
CF = (Nc**2-1)/(2*Nc) # Casimir factor associated with a gluon emission from a quark
CA = Nc # color factor associated with gluon emission from a gluon
dgq = 2.*Nc # degeneracy factor for quarks
dgg = 2.*(Nc**2.-1.) # degeneracy factor for gluons

########################################################################
# version of the DQPM partons
offshell = True
massive = False

# for outputs
if(offshell):
    version_out = '_offshell'
else:
    version_out = '_onshell'
if(massive):
    version_out += '_m'
if(Nf!=3):
    version_out += f'_Nf{Nf}'

########################################################################
def create_spline(x,y):
    return scipy.interpolate.Akima1DInterpolator(x,y)

########################################################################
def Tc(muB=0.,muBoT=False):
    """
    Critical temperature as a function of muB [GeV]
    """
    if(Nf==0):
        return 0.270
    else:
        if(not muBoT):
            return Tc_lattice(muB)
        elif(muBoT):
            return Tc_lattice_muBoT(muB)

########################################################################
# lQCD data for SU(3)
# doi:10.1007/JHEP07(2012)056
SU3_lQCD = pd.read_csv(f'{dir_path}/test/SU3_lQCD.csv').to_dict(orient='list')
for item in SU3_lQCD:
    SU3_lQCD[item] = np.array(SU3_lQCD[item])

# pressure
SU3_spline_P0 = create_spline(SU3_lQCD['T']*Tc(), SU3_lQCD['P'])
# interaction mesure
SU3_spline_I0 = create_spline(SU3_lQCD['T']*Tc(), SU3_lQCD['I'])

def fit_SU3(T):
    """
    fit to SU(3) thermodynamics
    """
    P0 = SU3_spline_P0(T)
    I0 = SU3_spline_I0(T)
    e0 = I0+3*P0
    s0 = e0+P0
    cs2 = (s0*T**3)/(T*scipy.misc.derivative(lambda xT: (SU3_spline_I0(xT)+4*SU3_spline_P0(xT))*xT**3, T, dx=0.005, n=1, order=3))
    return {'P':P0,'e':e0,'s':s0,'I':I0,'cs^2':cs2}

########################################################################
# pre calculate s(T) at mu=0
if(Nf>0):
    xtemp = np.arange(0.05,100.,0.005)
    spline_s0 = create_spline(xtemp, param(xtemp,0.,0.,0.)['s'])

def s0(T):
    """
    Interpolation of the entropy density s/T^3 at mu=0
    """
    if(Nf==0):
        return fit_SU3(T)['s']
    else:
        return spline_s0(T)

########################################################################
def g2(T,muB,muBoT=False):
    """
    DQPM coupling constant g^2 as a function of T and muB
    From a parametrization of s/T^3 at muB=0 from lQCD results
    Scaling hypothesis to extend at finite muB
    """

    # if input is a single temperature value T
    if(isinstance(T,float)):
        # my fit using standard DQPM masses
        if(Nf==0 and offshell):
            d = 247.694820
            e = -0.122011039
            f = 1.18562140
        elif(Nf==0 and not offshell):
            d = 122.15432283
            e = -0.17795281
            f = 1.13645454
        if(Nf==3 and offshell):
            d = 169.30064363
            e = -0.17286324
            f = 1.14180968
        elif(Nf==3 and not offshell):
            d = 108.28
            e = -0.23558394
            f = 1.12724123
        
        muq = muB/3.
        
        Tstar = np.sqrt(T**2.+(muq/pi)**2.)
        Tscal = Tstar*Tc(0.)/Tc(muB)
        
        sSB = SB_lim(1000000.,0,0,0,Nf=Nf)['s'] # s_sB/T^3
        s = s0(Tscal)

        g2s = d*((s/sSB)**e-1.)**f
    
    # if the input is a list of temperature values
    elif(isinstance(T,np.ndarray) or isinstance(T,list)):

        g2s = np.zeros_like(T)
        for i,xT in enumerate(T):
            # see if arrays are also given for chemical potentials
            try:
                xmuB = muB[i]
            except:
                xmuB = muB
            g2s[i] = g2(xT,xmuB)

    else:
        raise Exception('Problem with input')

    return g2s

########################################################################
def runcoup(T,Nf=3,LambdaMS=0.176):
    """
    Running coupling as a function of scale Lambda [GeV]
    """
    b0 = (11*CA-2*Nf)/(12*pi)
    Lambda = 2*pi*T
    t = np.log(Lambda**2/LambdaMS**2)

    return 1/(b0*t)

########################################################################
def der_mu(func,order,**kwargs):
    """
    Derivatives of function f(muB,muQ,muS) wrt muB,muQ,muS
    """
    # spacing for the derivative
    try:
        dx = kwargs['dx']
    except:
        dx = 0.001 # default

    # number of points to consider for evaluating the derivative
    try:
        npoints = kwargs['npoints']
        # same number of points for each chemical potential
        if(isinstance(npoints,int)):
            order_muB = npoints
            order_muQ = npoints
            order_muS = npoints
        # or different number of points if dx npoints is a list
        else:
            order_muB = npoints[0]
            order_muQ = npoints[1]
            order_muS = npoints[2]
    except:
        # default
        order_muB = 2*order[0]+1
        order_muQ = 2*order[1]+1
        order_muS = 2*order[2]+1

    dfdmuB = lambda xmuQ,xmuS: scipy.misc.derivative(lambda xmuB: func(xmuB,xmuQ,xmuS), 0., dx=dx, n=order[0], order=order_muB)
    dfdmuQ = lambda xmuS: scipy.misc.derivative(lambda xmuQ: dfdmuB(xmuQ,xmuS), 0., dx=dx, n=order[1], order=order_muQ)
    return scipy.misc.derivative(lambda xmuS: dfdmuQ(xmuS), 0., dx=dx, n=order[2], order=order_muS)

########################################################################
class parton:
    """
    Define properties of all DQPM partons
    """
    def __init__(self,name,ID,Bcharge,Qcharge,Scharge,M0,xM,w,dg,BEFD):
        self.name = name # name of the parton
        self.ID = ID # ID of the parton
        self.Bcharge = Bcharge # baryon charge
        self.Qcharge = Qcharge # electric charge
        self.Scharge = Scharge # strangeness
        self.M0 = M0 # bare mass
        self.xM = xM # function for thermal mass
        self.w = w # function for width
        self.dg = dg # degeneracy factor
        self.BEFD = BEFD # BE enhancement or FD blocking factor

    def muq(self,muB,muQ,muS):
        """
        parton chemical potential [GeV]
        """
        return self.Bcharge*muB + self.Qcharge*muQ + self.Scharge*muS

    def M(self,T,muB,muQ,muS,**kwargs):
        """
        Parton mass in [GeV]
        """
        if(self.name=='g'):
            result = self.xM(T,muB,muQ,muS,**kwargs)
        else:
            muq = self.muq(muB,muQ,muS) # quark chemical potential
            result = self.xM(T,muB,muQ,muS,muq,**kwargs)
        return result
    
    def fq(self,om,T,muB,muQ,muS):
        """
        parton distribution function (Bose Einstein for gluons, Fermi Dirac for quarks)
        """
        xmuq = self.muq(muB,muQ,muS)
        return 1./(np.exp((om-xmuq)/T)+self.BEFD)
    
    def fqdT(self,om,T,muB,muQ,muS):
        """
        Derivative of the distribution function wrt temperature T [GeV^-1]
        """
        xmuq = self.muq(muB,muQ,muS)
        return (om-xmuq)/T**2.*np.exp((om-xmuq)/T)/(np.exp((om-xmuq)/T)+self.BEFD)**2.
    
    def fqdmuq(self,om,T,muB,muQ,muS):
        """
        Derivative of the distribution function wrt quark chemical potential muq [GeV^-1]
        """
        if(abs(self.ID) == 10):
            return 0.
        else:
            xmuq = self.muq(muB,muQ,muS) # quark chemical potential
            return 1./T*np.exp((om-xmuq)/T)/(np.exp((om-xmuq)/T)+self.BEFD)**2.
    
    def SE(self,om,p,T,muB,muQ,muS,**kwargs):
        """
        self energy [GeV^2]
        """
        xM = self.M(T,muB,muQ,muS,**kwargs) # thermal mass
        xM0 = self.M0 # bare mass
        mass2 = xM0**2. + np.sqrt(2.)*xM0*xM + xM**2. # real part of self-energy
        return mass2 - 2.*complex(0.,1.)*om*self.w(T,muB,muQ,muS,**kwargs)
    
    def Prop(self,om,p,T,muB,muQ,muS,**kwargs):
        """
        Parton propagator [GeV^-2]
        """
        return 1./(om**2.-p**2.-self.SE(om,p,T,muB,muQ,muS,**kwargs))

    def rho(self,om,p,T,muB,muQ,muS,**kwargs):
        """
        Parton spectral function [GeV^-2]
        """
        return 4.*om*self.w(T,muB,muQ,muS**kwargs)/((om**2.-p**2.-(self.M(T,muB,muQ,muS,**kwargs))**2.)**2. + 4.*(om**2.)*(self.w(T,muB,muQ,muS,**kwargs))**2.)

    def int_sq(self,om,p,T,muB,muQ,muS,**kwargs):
        """
        Integrand for the DQPM entropy density [GeV^3]
        """
        xprop = self.Prop(om,p,T,muB,muQ,muS,**kwargs) # propagator
        return -1./(2.*pi)*(4.*pi*p**2.)/((2.*pi)**3)*self.dg*self.fqdT(om,T,muB,muQ,muS)\
            *(np.imag(np.log(-1./xprop))+np.imag(self.SE(om,p,T,muB,muQ,muS,**kwargs))*np.real(xprop))
    
    def int_nq(self,om,p,T,muB,muQ,muS,**kwargs):
        """
        Integrand for the DQPM quark density [GeV^3]
        """
        xprop = self.Prop(om,p,T,muB,muQ,muS,**kwargs) # propagator
        return -1./(2.*pi)*(4.*pi*p**2.)/((2.*pi)**3)*self.dg*self.fqdmuq(om,T,muB,muQ,muS)\
            *(np.imag(np.log(-1./xprop))+np.imag(self.SE(om,p,T,muB,muQ,muS,**kwargs))*np.real(xprop))

    def int_sq0(self,p,T,muB,muQ,muS,**kwargs):
        """
        Integrand for the ideal part of the entropy density [GeV^3]
        """
        Ep = np.sqrt(p**2.+self.M(T,muB,muQ,muS,**kwargs)**2.) # pole energy
        return 1./(2.*pi**2.*T)*(p**2.)*self.dg*(Ep-self.muq(muB,muQ,muS)+(p**2.)/(3.*Ep))*self.fq(Ep,T,muB,muQ,muS)
    
    def int_nq0(self,p,T,muB,muQ,muS,**kwargs):
        """
        Integrand for the ideal part of the quark density [GeV^3]
        """
        Ep = np.sqrt(p**2.+self.M(T,muB,muQ,muS,**kwargs)**2.) # pole energy
        return 1./(2.*pi**2.)*(p**2.)*self.dg*self.fq(Ep,T,muB,muQ,muS)

    def int_sqint(self,om,p,T,muB,muQ,muS,**kwargs):
        """
        Integrand for the interacting part of the entropy density [GeV^3]
        """
        xprop = self.Prop(om,p,T,muB,muQ,muS,**kwargs) # propagator
        Lambda = np.imag(xprop)/np.real(xprop)
        return 1./(2.*pi)*(4.*pi*p**2.)/((2.*pi)**3)*self.dg*self.fqdT(om,T,muB,muQ,muS)\
            *(np.arctan(Lambda)-Lambda/(1.+Lambda**2.))
    
    def int_nqint(self,om,p,T,muB,muQ,muS,**kwargs):
        """
        Integrand for the interacting part of the quark density [GeV^3]
        """
        xprop = self.Prop(om,p,T,muB,muQ,muS,**kwargs) # propagator
        Lambda = np.imag(xprop)/np.real(xprop)
        return 1./(2.*pi)*(4.*pi*p**2.)/((2.*pi)**3)*self.dg*self.fqdmuq(om,T,muB,muQ,muS)\
            *(np.arctan(Lambda)-Lambda/(1.+Lambda**2.))

    def int_eta0(self,p,T,muB,muQ,muS,**kwargs):
        """
        Integrand for the quasiparticle shear viscosity [GeV^3]
        """
        Ep = np.sqrt(p**2.+self.M(T,muB,muQ,muS,**kwargs)**2.) # pole energy
        ffq = self.fq(Ep,T,muB,muQ,muS) # distribution function
        return 1./(15.*T)*self.dg*(4.*pi*p**2.)/((2.*pi)**3)*(p**4.)/(Ep**2.)*ffq*(1-self.BEFD*ffq)/(2*self.w(T,muB,muQ,muS,**kwargs))

    def int_zeta0(self,p,T,muB,muQ,muS,cs2=1./3.,**kwargs):
        """
        Integrand for quasiparticle bulk viscosity [GeV^3]
        """
        Ep = np.sqrt(p**2.+self.M(T,muB,muQ,muS,**kwargs)**2.) # pole energy
        ffq = self.fq(Ep,T,muB,muQ,muS) # distribution function
        dM2d2T = scipy.misc.derivative(lambda xT: self.M(xT,muB,muQ,muS,**kwargs)**2., T, dx=0.001, n=2, order=5)
        return 1./(9.*T)*self.dg*(4.*pi*p**2.)/((2.*pi)**3)/(Ep**2.)*(p**2.-3*cs2*(Ep**2.-T**2.*dM2d2T))*ffq*(1-self.BEFD*ffq)/(2*self.w(T,muB,muQ,muS,**kwargs))

    def int_Tmunu_e(self,om,p,T,muB,muQ,muS,**kwargs):
        """
        Energy density from energy momentum tensor [GeV^4]
        """
        return self.dg*(4.*pi*p**2.)/((2.*pi)**4)*2.*om*self.rho(om,p,T,muB,muQ,muS,**kwargs)*np.heaviside(om,0.5)*self.fq(om,T,muB,muQ,muS)*om

    def int_Tmunu_P(self,om,p,T,muB,muQ,muS,**kwargs):
        """
        Pressure from energy momentum tensor [GeV^4]
        """
        return self.dg*(4.*pi*p**2.)/((2.*pi)**4)*2.*om*self.rho(om,p,T,muB,muQ,muS,**kwargs)*np.heaviside(om,0.5)*self.fq(om,T,muB,muQ,muS)*(p**2.)/(3.*om)

########################################################################
def Mg(T,muB,muQ,muS,**kwargs):
    """
    DQPM gluon mass [GeV]
    """
    # if input is a single temperature value T
    if(isinstance(T,float)):
        # coupling constant
        try:
            gs2 = kwargs['g2']
        except:
            gs2 = g2(T,muB) # default

        # sum of quark chemical potential squared
        xsum = sum([(parton.muq(muB,muQ,muS))**2. for parton in list_quarks])

        Mg2 = gs2/6.*((Nc+Nf/2.)*T**2. + Nc/2.*xsum/pi**2.)
        result = np.sqrt(Mg2)
        
    # if the input is a list of temperature values
    elif(isinstance(T,np.ndarray) or isinstance(T,list)):

        result = np.zeros_like(T)
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
            result[i] = Mg(xT,xmuB,xmuQ,xmuS,**kwargs)

    else:
        raise Exception('Problem with input')

    return result

########################################################################
def gammag(T,muB,muQ,muS,**kwargs):
    """
    DQPM gluon width [GeV]
    """
    # if input is a single temperature value T
    if(isinstance(T,float)):

        # coupling constant
        try:
            gs2 = kwargs['g2']
        except:
            gs2 = g2(T,muB) # default

        c = 14.4
        result = 1./3.*Nc*(gs2*T)/(8.*pi)*np.log(2.*c/gs2+1.)

    # if the input is a list of temperature values
    elif(isinstance(T,np.ndarray) or isinstance(T,list)):

        result = np.zeros_like(T)
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
            result[i] = gammag(xT,xmuB,xmuQ,xmuS,**kwargs)

    else:
        raise Exception('Problem with input')

    return result

########################################################################
def Mq(T,muB,muQ,muS,muq,**kwargs):
    """
    DQPM light quark mass [GeV]
    """
    # if input is a single temperature value T
    if(isinstance(T,float)):

        # coupling constant
        try:
            gs2 = kwargs['g2']
        except:
            gs2 = g2(T,muB) # default

        Mq2 = ((Nc**2.-1.)/(8.*Nc))*gs2*(T**2.+muq**2./pi**2.)
        result = np.sqrt(Mq2)
        
    # if the input is a list of temperature values
    elif(isinstance(T,np.ndarray) or isinstance(T,list)):

        result = np.zeros_like(T)
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
            try:
                xmuq = muq[i]
            except:
                xmuq = muq
            result[i] = Mq(xT,xmuB,xmuQ,xmuS,xmuq,**kwargs)

    else:
        raise Exception('Problem with input')

    return result

########################################################################
def gammaq(T,muB,muQ,muS,**kwargs):
    """
    DQPM light quark width [GeV]
    """

    # if input is a single temperature value T
    if(isinstance(T,float)):

        # coupling constant
        try:
            gs2 = kwargs['g2']
        except:
            gs2 = g2(T,muB) # default

        c = 14.4
        result = 1./3.*((Nc**2.-1.)/(2.*Nc))*(gs2*T)/(8.*pi)*np.log(2.*c/gs2+1.)

    # if the input is a list of temperature values
    elif(isinstance(T,np.ndarray) or isinstance(T,list)):

        result = np.zeros_like(T)
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
            result[i] = gammaq(xT,xmuB,xmuQ,xmuS,**kwargs)

    else:
        raise Exception('Problem with input')

    return result

########################################################################
def Ms(T,muB,muQ,muS,muq,**kwargs):
    """
    DQPM strange quark mass [GeV]
    """
    return Mq(T,muB,muQ,muS,muq,**kwargs) + 0.030

########################################################################
# construct all parton objects
def construct_partons(Nf,massive):
    """
    Construct parton objects
    """
    if(massive):
        # bare quark masses from PDG 2020
        M0ud = 0.00345 # GeV
        M0s = 0.093 # GeV
        s = parton("s",3,1./3.,-1./3.,-1.,M0s,Mq,gammaq,dgq,1.)
        sbar = parton("sbar",-3,-1./3.,1./3.,1.,M0s,Mq,gammaq,dgq,1.)
    else:
        M0ud = 0.
        M0s = 0.
        s = parton("s",3,1./3.,-1./3.,-1.,M0s,Ms,gammaq,dgq,1.)
        sbar = parton("sbar",-3,-1./3.,1./3.,1.,M0s,Ms,gammaq,dgq,1.)

    u = parton("u",1,1./3.,2./3.,0.,M0ud,Mq,gammaq,dgq,1.)
    d = parton("d",2,1./3.,-1./3.,0.,M0ud,Mq,gammaq,dgq,1.)
    ubar = parton("ubar",-1,-1./3.,-2./3.,-0.,M0ud,Mq,gammaq,dgq,1.)
    dbar = parton("dbar",-2,-1./3.,1./3.,-0.,M0ud,Mq,gammaq,dgq,1.)
    g = parton("g",10,0.,0.,0.,0.,Mg,gammag,dgg,-1.)

    return u,d,s,ubar,dbar,sbar,g

def construct_list_partons(Nf,massive):
    """
    Construct list of partons to consider
    """
    u,d,s,ubar,dbar,sbar,g = construct_partons(Nf,massive)

    if(Nf==0):
        list_parton = [g]
        list_quarks_all = []
        list_quarks = []
    elif(Nf==2):
        list_parton = [u,d,ubar,dbar,g]
        list_quarks_all = [u,d,ubar,dbar]
        list_quarks = [u,d]
    elif(Nf==3):
        list_parton = [u,d,s,ubar,dbar,sbar,g]
        list_quarks_all = [u,d,s,ubar,dbar,sbar]
        list_quarks = [u,d,s]

    return list_parton,list_quarks_all,list_quarks

u,d,s,ubar,dbar,sbar,g = construct_partons(Nf,massive)
list_parton,list_quarks_all,list_quarks = construct_list_partons(Nf,massive)