import os
import numpy as np
import matplotlib
import matplotlib.pyplot as pl
import pandas as pd
import argparse

# import from __init__.py
from DQPM_thermo.test import *
from DQPM_thermo import *
from DQPM_thermo.DQPM import *
# to plot lattice data
from EoS_HRG.fit_lattice import chi_lattice2020,chi_lattice2015,chi_lattice2014,chi_lattice2012,chi_lattice2018,chi_lattice2017
from EoS_HRG.fit_lattice import param, param_chi, list_chi, chi_latex, chi_SB, BQS
from EoS_HRG.test.plot_lattice import plot_lattice

# directory where the file is located
dir_path = os.path.dirname(os.path.realpath(__file__))

########################################################################
# read coupling constant at Nf=0
# PhysRevD.70.074505
alphasNf0 = np.array([[1.032523391,1.449584816,0.201660735],\
[1.183073696,1.138790036,0.168446027],\
[1.293005114,1.005931198,0.182680901],\
[1.49321251,0.697508897,0.06168446],\
[3.004477858,0.429418743,0.054567023],\
[5.995203978,0.296559905,0.075919336],\
[8.995510358,0.234875445,0.023724793],\
[11.98558837,0.232502966,0.035587189]])

# lQCD data for SU(3)
# doi:10.1007/JHEP07(2012)056
SU3_lQCD = pd.read_csv(f'{dir_path}/SU3_lQCD.csv').to_dict(orient='list')

###############################################################################
__doc__ = """Produce plots to compare lQCD data with the DQPM results"""
###############################################################################
@timef
def main(tab,muBoT=False):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--Tmax', type=float, default=0.6,
        help='maximum temperature [GeV]'
    )
    parser.add_argument(
        '--muBmax', type=float, default=0.6,
        help='maximum value of muB [GeV]'
    )
    args = parser.parse_args()

    # min/max for the temperature
    Tmax = args.Tmax
    muBmax = args.muBmax

    ########################################################################
    # plot for alpha_S for # muB
    print('plot alpha_S')
    xval = np.logspace(0,1.3,100) # T/Tc
    fig,ax = pl.subplots(figsize=(10,7))

    ax.errorbar(alphasNf0[:,0],alphasNf0[:,1],yerr=alphasNf0[:,2], fmt='o', color='k', linestyle='None', label=r'$lQCD\ N_f=0$')

    ax.plot(xval,runcoup(xval*Tc(0)), '-.', color='b', linewidth='4', label=r'$\alpha_S(\Lambda)\ N_f=3$')
    ax.plot(xval,runcoup(xval*Tc(0),Nf=0), '-.', color='k', linewidth='4', label=r'$\alpha_S(\Lambda)\ N_f=0$')

    for muB,color in tab:
        if(muBoT and muB!=0):
            Tmax = muBmax/muB # only calculate up to muB ~ 0.4
            if(Tmax<Tc(muB,muBoT)):
                continue
        else:
            Tmax = args.Tmax
        if(not muBoT):
            ax.plot(xval,g2(xval*Tc(muB),muB)/(4.*pi), color=color, linewidth='5', label=r'$ \mu_B = $'+str(muB)+' GeV')
        elif(muBoT):
            ax.plot(xval,g2(xval*Tc(muB,muBoT),muB*xval*Tc(muB,muBoT))/(4.*pi), color=color, linewidth='5', label=r'$ \mu_B/T = $'+str(muB))
        ax.legend(bbox_to_anchor=(0.7, 0.95), loc='upper right', borderaxespad=0., frameon=False)

    ax.set(ylabel=r'$\alpha_S$',xlabel=r'$T/T_c$',xscale='log',ylim=(0.,None))
    ax.set_xticks([1,2,3,4,5,10])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #ax.text(7, 2.65, 'DQPM', fontsize=MEDIUM_SIZE)
    if(not muBoT):
        if(Nf==0):
            fig.savefig(f'{dir_path}/DQPM_alphas_T{version_out}.png')
        else:
            fig.savefig(f'{dir_path}/DQPM_alphas_T{version_out}_muB.png')
    elif(muBoT):
        fig.savefig(f'{dir_path}/DQPM_alphas_T{version_out}_muBoT.png')

    fig.clf()
    pl.close(fig)

    ########################################################################
    # plot for g_S for # muB
    print('plot g_S')
    xval = np.logspace(0,1.3,100) # T/Tc

    fig,ax = pl.subplots(figsize=(10,7))

    ax.errorbar(alphasNf0[:,0],np.sqrt(alphasNf0[:,1]*(4*pi)),yerr=4*pi*alphasNf0[:,2]/(2*np.sqrt(alphasNf0[:,1]*(4*pi))), fmt='o', color='k', linestyle='None', label=r'$lQCD\ N_f=0$')

    ax.plot(xval,np.sqrt(runcoup(xval*Tc(0))*(4*pi)), '-.', color='b', linewidth='4', label=r'$g_S(\Lambda)\ N_f=3$')
    ax.plot(xval,np.sqrt(runcoup(xval*Tc(0),Nf=0)*(4*pi)), '-.', color='k', linewidth='4', label=r'$g_S(\Lambda)\ N_f=0$')

    for muB,color in tab:
        if(muBoT and muB!=0):
            Tmax = muBmax/muB # only calculate up to muB ~ 0.4
            if(Tmax<Tc(muB,muBoT)):
                continue
        else:
            Tmax = args.Tmax
        if(not muBoT):
            ax.plot(xval,np.sqrt(g2(xval*Tc(muB),muB)), color=color, linewidth='5', label=r'$ \mu_B = $'+str(muB)+' GeV')
        elif(muBoT):
            ax.plot(xval,np.sqrt(g2(xval*Tc(muB,muBoT),muB*xval*Tc(muB,muBoT))), color=color, linewidth='5', label=r'$ \mu_B/T = $'+str(muB))
        ax.legend(bbox_to_anchor=(0.7, 0.95), loc='upper right', borderaxespad=0., frameon=False)

    ax.set(ylabel=r'$g_S$',xlabel=r'$T/T_c$',xscale='log',ylim=(0.,None))
    ax.set_xticks([1,2,3,4,5,10])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #ax.text(7, 2.65, 'DQPM', fontsize=MEDIUM_SIZE)
    pl.show()
    if(not muBoT):
        if(Nf==0):
            fig.savefig(f'{dir_path}/DQPM_gs_T{version_out}.png')
        else:
            fig.savefig(f'{dir_path}/DQPM_gs_T{version_out}_muB.png')
    elif(muBoT):
        fig.savefig(f'{dir_path}/DQPM_gs_T{version_out}_muBoT.png')

    fig.clf()
    pl.close(fig)

    ########################################################################
    # plot for Mg for # muB
    print('plot Mg')
    fig,ax = pl.subplots(figsize=(10,7))
    for muB,color in tab:
        if(muBoT and muB!=0):
            Tmax = muBmax/muB # only calculate up to muB ~ 0.4
            if(Tmax<Tc(muB,muBoT)):
                continue
        else:
            Tmax = args.Tmax
        xval = np.arange(Tc(muB,muBoT),Tmax,0.01)
        if(not muBoT):
            ax.plot(xval,g.M(xval,muB,0.,0.), color=color, linewidth='5', label=r'$ \mu_B = $'+str(muB)+' GeV')
            ax.plot(xval,g.w(xval,muB,0.,0.), color=color, linewidth='5')
        elif(muBoT):
            ax.plot(xval,g.M(xval,muB*xval,0.,0.), color=color, linewidth='5', label=r'$ \mu_B/T = $'+str(muB))
            ax.plot(xval,g.w(xval,muB*xval,0.,0.), color=color, linewidth='5')
        ax.legend(bbox_to_anchor=(0.95, 0.5), loc='center right', borderaxespad=0., frameon=False)

    ax.set(ylabel=r'$M_g, \gamma_g$ [GeV]',xlabel=r'$T$ [GeV]')
    ax.set_xlim(0.1,)
    ax.set_ylim(0,)
    #ax.text(0.12, 0.95, 'DQPM', fontsize=MEDIUM_SIZE)
    pl.show()
    if(not muBoT):
        if(Nf==0):
            fig.savefig(f'{dir_path}/DQPM_Mg_T{version_out}.png')
        else:
            fig.savefig(f'{dir_path}/DQPM_Mg_T{version_out}_muB.png')
    elif(muBoT):
        fig.savefig(f'{dir_path}/DQPM_Mg_T{version_out}_muBoT.png')

    fig.clf()
    pl.close(fig)

    ########################################################################
    # plot for Mq for # muB
    if(Nf!=0):
        print('plot Mq')
        fig,ax = pl.subplots(figsize=(10,7))
        for muB,color in tab:
            if(muBoT and muB!=0):
                Tmax = muBmax/muB # only calculate up to muB ~ 0.4
                if(Tmax<Tc(muB,muBoT)):
                    continue
            else:
                Tmax = args.Tmax
            xval = np.arange(Tc(muB,muBoT),Tmax,0.01)
            if(not muBoT):
                ax.plot(xval,u.M(xval,muB,0.,0.), color=color, linewidth='5', label=r'$ \mu_B = $'+str(muB)+' GeV')
                ax.plot(xval,u.w(xval,muB,0.,0.), color=color, linewidth='5')
            elif(muBoT):
                ax.plot(xval,u.M(xval,muB*xval,0.,0.), color=color, linewidth='5', label=r'$ \mu_B/T = $'+str(muB))
                ax.plot(xval,u.w(xval,muB*xval,0.,0.), color=color, linewidth='5')
            ax.legend(bbox_to_anchor=(0.95, 0.5), loc='center right', borderaxespad=0., frameon=False)

        ax.set(ylabel=r'$M_q, \gamma_q$ [GeV]',xlabel=r'$T$ [GeV]')
        ax.set_xlim(0.1,)
        ax.set_ylim(0,)
        #ax.text(0.12, 0.62, 'DQPM', fontsize=MEDIUM_SIZE)
        pl.show()
        if(not muBoT):
            fig.savefig(f'{dir_path}/DQPM_Mq_T{version_out}_muB.png')
        elif(muBoT):
            fig.savefig(f'{dir_path}/DQPM_Mq_T{version_out}_muBoT.png')

    fig.clf()
    pl.close(fig)

    ########################################################################
    # plot therodynamics for # muB
    # array of plots P,nB,s,e

    list_quant = ['P','n_B','n_Q','n_S','s','e','I','cs^2']

    if(Nf!=0):
        Tmax = args.Tmax
        if(not muBoT):
            wparam =True
        else:
            # when muBoT, don't plot param
            wparam = False
        dict_plots = plot_lattice('muB',tab,list_quant,wparam=wparam,all_labels=False,muBoT=muBoT,Tmax=Tmax,SB=True)
    else:
        dict_plots = {}

    print('plot thermo')
    for muB,color in tab:
        if(muBoT and muB!=0):
            Tmax = muBmax/muB # only calculate up to muB ~ 0.4
            if(Tmax<Tc(muB,muBoT)):
                continue
        else:
            Tmax = args.Tmax
        xtemp = np.logspace(np.log(Tc(muB,muBoT)),np.log(Tmax),20,base=np.exp(1))  # array for T
        x_int = np.logspace(np.log(Tc(muB,muBoT)),np.log(Tmax),200,base=np.exp(1)) # to interpolate
        if(not muBoT):
            print('   muB = ', muB, ' GeV')
            yval = DQPM_thermo(xtemp,muB,0.,0.)
        elif(muBoT):
            print('   muB/T = ', muB)
            yval = DQPM_thermo(xtemp,muB*xtemp,0.,0.)

        for quant in list_quant:

            if(Nf==0):
                if(quant=='n_B' or quant=='n_Q' or quant=='n_S'):
                    continue
                fig,ax = pl.subplots(figsize=(10,7))
                ax.plot(np.linspace(0.2,Tmax,1000),fit_SU3(np.linspace(0.2,Tmax,1000))[quant], '--', color='k', linewidth='1')
                try:
                    ax.errorbar(np.array(SU3_lQCD['T'])*Tc(0.),SU3_lQCD[quant],yerr=SU3_lQCD[quant+'_err'], fmt='o', ms='4', color='k',label=r'lQCD $N_f=0$')
                except:
                    pass
                dict_plots.update({quant:[fig,ax]})

            if(not muBoT):
                label = r'$ \mu_B = $'+str(muB)+' GeV'
            elif(muBoT):
                label = r'$ \mu_B/T = $'+str(muB)

            #dict_plots[quant][1].plot(xtemp,yval[quant], 'o', color=color, ms='3', linestyle='None')
            dict_plots[quant][1].plot(x_int,create_spline(xtemp,yval[quant])(x_int), color=color, linewidth='2.',label=label)
            dict_plots[quant][1].legend(bbox_to_anchor=(0.95, 0.4),title='DQPM', loc='center right', borderaxespad=0., frameon=False)
    
    for quant in list_quant:
        if(Nf==0 and (quant=='n_B' or quant=='n_Q' or quant=='n_S')):
            continue
        dict_plots[quant][1].set_xlim(0.12,args.Tmax)
        if(muBoT and (quant=='n_B' or quant=='n_Q' or quant=='n_S')):
            dict_plots[quant][1].set_xlim(0.12,0.3)
        if(not muBoT):
            if(Nf==0):
                dict_plots[quant][0].savefig(f'{dir_path}/DQPM_{quant}_T{version_out}.png')
            else:
                dict_plots[quant][0].savefig(f'{dir_path}/DQPM_{quant}_T{version_out}_muB.png')
        elif(muBoT):
            dict_plots[quant][0].savefig(f'{dir_path}/DQPM_{quant}_T{version_out}_muBoT.png')

        dict_plots[quant][0].clf()
        pl.close(dict_plots[quant][0])


########################################################################
@timef
def plot_Tmunu(tab):
    """
    To plot results from the DQPM energy momentum tensor (e,P)
    """
    muB = 0.
    list_quant = ['e','P']
    dict_plots = plot_lattice('muB',tab,list_quant) # get plot of lQCD data

    print('plot Tmunu')

    xtemp = np.linspace(Tc(muB),0.6,10) # array for T
    yval = DQPM_thermo(xtemp,0.,0.,0.) # DQPM thermo
    result_Tmunu = DQPM_Tmunu(xtemp,0.,0.,0.) # DQPM Tmunu

    x_int = np.linspace(Tc(muB),0.6,30) # for spline

    dict_plots['e'][1].plot(x_int,create_spline(xtemp,yval['e'])(x_int), linewidth='2.5', label=r'$e^{DQPM}$')
    dict_plots['e'][1].plot(x_int,create_spline(xtemp,result_Tmunu['e'])(x_int), linewidth='2.5', label=r'$T^{00}$')
    dict_plots['e'][1].plot(x_int,create_spline(xtemp,result_Tmunu['ep'])(x_int), '--', linewidth='2.5', label=r'$T^{00}_{+}$')
    dict_plots['e'][1].plot(x_int,create_spline(xtemp,result_Tmunu['em'])(x_int), '--', linewidth='2.5', label=r'$T^{00}_{-}$')
    dict_plots['e'][1].plot(x_int,create_spline(xtemp,yval['e']-result_Tmunu['ep'])(x_int), '--', linewidth='2.5', label=r'$e^{DQPM}-T^{00}_{+}$')

    dict_plots['e'][1].legend(loc='best', borderaxespad=0., frameon=False)
    dict_plots['e'][1].set_xlim(0.13,0.6)
    dict_plots['e'][0].savefig(f'{dir_path}/DQPM_Tmunu_e_T{version_out}.png')

    dict_plots['P'][1].plot(x_int,create_spline(xtemp,yval['P'])(x_int), linewidth='2.5', label='DQPM')
    dict_plots['P'][1].plot(x_int,create_spline(xtemp,result_Tmunu['P'])(x_int), linewidth='2.5', label=r'$T^{ii}$')
    dict_plots['P'][1].plot(x_int,create_spline(xtemp,result_Tmunu['Pp'])(x_int), '--', linewidth='2.5', label=r'$T^{ii}_{+}$')
    dict_plots['P'][1].plot(x_int,create_spline(xtemp,result_Tmunu['Pm'])(x_int), '--', linewidth='2.5', label=r'$T^{ii}_{-}$')

    dict_plots['P'][1].legend(loc='best', borderaxespad=0., frameon=False)
    dict_plots['P'][1].set_xlim(0.13,0.6)
    dict_plots['P'][0].savefig(f'{dir_path}/DQPM_Tmunu_P_T{version_out}.png')

########################################################################
@timef
def plot_chi():
    """
    To plot the fit of susceptibilities compared to lQCD results
    """
    # susceptibilities from DQPM
    x_int = np.linspace(Tc(0.),0.6,50) # for spline
    xTDQPM = np.array([Tc(0.),0.2,0.25,0.3,0.35,0.4,0.5,0.6])
    data_DQPM_chi = DQPM_chi(xTDQPM)['chi']

    # loop over susceptibilities
    for ichi,chi in enumerate(list_chi):
        ii = BQS[chi]['B']
        jj = BQS[chi]['Q']
        kk = BQS[chi]['S']

        # skip high orders
        if((ii+jj+kk)>=4):
            continue

        f,ax = pl.subplots(figsize=(8,7))

        # loop over lattice data
        for chi_lattice,point in [[chi_lattice2020,'s'],[chi_lattice2015,'*'],[chi_lattice2014,'^'],[chi_lattice2012,'o'],[chi_lattice2018,'p'],[chi_lattice2017,'P']]:
            try:
                data = chi_lattice[chi]
                ax.plot(data[:,0], data[:,1], point, color='b', ms='6', fillstyle='none',label='lQCD')
                ax.errorbar(data[:,0], data[:,1], yerr=data[:,2], fmt='none', color='b')
                break
            except:
                pass

        # param
        ax.plot(x_int,param_chi(x_int,chi),'--',color='k',linewidth=2.5,label='Parametrization')
        # SB limit
        ax.plot([0.5,0.7],chi_SB[chi]*np.ones(2),color='k',linewidth=5,label='SB limit')
        # susceptibilities from DQPM
        #ax.plot(xTDQPM,data_DQPM_chi[ichi],'--',color='k',linewidth=2,label='DQPM')
        ax.plot(x_int,create_spline(xTDQPM,data_DQPM_chi[ichi])(x_int),'--',color='r',linewidth=2,label='DQPM')

        ylimm = 1.1*min(data_DQPM_chi[ichi][0],chi_SB[chi],np.amin(data[:,1]-data[:,2]))
        ylimp = 1.1*max(data_DQPM_chi[ichi][0],chi_SB[chi],np.amax(data[:,1]+data[:,2]))

        if(ylimm>=0.):
            ylimm = -0.05*ylimp
        else:
            ylimp = max(0.05*abs(ylimm),ylimp)
        ax.set(xlabel='T [GeV]',ylabel=chi_latex[chi],xscale='linear',ylim=[ylimm,ylimp],xlim=[None,0.6])
        ax.legend(title=None, title_fontsize='25', loc='best', borderaxespad=0.5, frameon=False)
        f.savefig(f'{dir_path}/{chi}{version_out}.png')
        f.clf()
        pl.close(f)

###############################################################################
if __name__ == "__main__":
    # values of \mu_B where to calculate
    tab = [[0,'r'],[0.2,'tab:orange'],[0.3,'b'],[0.4,'g']]
    if(Nf==0):
        tab = [[0.,'r']]
    main(tab)

    if(Nf>0):
        # values of \mu_B/T where to calculate
        tab = [[0,'r'],[1,'tab:orange'],[2,'b'],[3,'g'],[3.5,'m']]
        main(tab,muBoT=True)

        print('\nCalculation of chi')
        plot_chi()
    
    #tab = [[0.,'r']]
    #plot_Tmunu(tab)