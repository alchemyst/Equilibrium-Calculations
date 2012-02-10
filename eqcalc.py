#!/usr/bin/env python

# Calculate mixture equilibriums using minimisation of the Gibbs energy

# 201110 Originally by Hendrik Venter
# 201111 Significantly reworked by Carl Sandrock

from __future__ import division
import scipy.optimize
import scipy.linalg
import math
import numpy
import atomparser
import sys
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

smallvalue = 1e-10

nspecies = 9                    # number of species precent
T = 298
Tfinal = 473
R = 8.314
Steps = Tfinal - T

Temprange = numpy.linspace(T, Tfinal, Steps)
dT = Temprange[1] - Temprange[0]
length = numpy.size(Temprange)

Cpv = numpy.zeros([length,nspecies])
Cpvalue = numpy.zeros([length,nspecies])
Cpintvalue = numpy.zeros([length,nspecies])
CpintTvalue = numpy.zeros([length,nspecies])
Hvalue = numpy.zeros([length,nspecies])
Svalue = numpy.zeros([length,nspecies])
Gvalue = numpy.zeros([length,nspecies])

position = 0
mustplot = 0    # 1 for Plotting, 0 for not plotting
Minvalue = 0    # Used to calculate Minimum function value

for i in Temprange:
    T = i
    RT = R*T
    class compound:            
        """ Basic container for compound properties """
        def __init__(self, name, DGf):
            self.name = name
            self.DGf = DGf
            self.parsed = atomparser.parseformula(name)
        
    class mixture:
        """ Container for mixture properties """
        def __init__(self, charge):
            """ Initialise mixture - charge contains tuples of (initialcharge, compound) """
            self.N, self.compounds = zip(*charge)
            self.N = numpy.array(self.N)
            self.compoundnames = [c.name for c in self.compounds]
            self.DGf = numpy.array([c.DGf for c in self.compounds])
            self.elements = reduce(set.union, (c.parsed.distinctelements() 
                                                for c in self.compounds))
            self.S = numpy.array([c.parsed.counts(self.elements) 
                                    for c in self.compounds]).T
            Srank = numpy.count_nonzero(numpy.linalg.svd(self.S, compute_uv=False) > smallvalue)
                
            # Calculating the Degrees of freedom
            self.DOF = self.S.shape[1] - Srank                      
                
            # Coefficients of the Gibbs function
            Ncomps = len(self.compounds)
            # self.A = numpy.tile(numpy.repeat([-1, 1], [1, Ncomps-1]), [Ncomps, 1])
            # number of atoms of each element
            self.atoms = numpy.dot(self.S, self.N)
        
        
        def gibbs(self, N=None):
            """ Gibbs energy of mixture with N of each compound """
            if N is None: N = self.N
            # TODO: There is every chance that this function is not correct. It needs to be checked.
            #return sum(N*(self.DGf/(R*T) + numpy.log(numpy.dot(self.A, N))))
            logs = numpy.log(sum(N)) 
            RT = R*T      
            return sum(N*(self.DGf/RT + numpy.log(N) - logs))
        
        def atombalance(self, N):
        
            """ Atom balance with N of each compound """
            #if N is None: N = self.N
            return numpy.dot(self.S, N) - self.atoms
        
            
        def conversion(self, conversionspec):
            """ Calculate the composition given a certain conversion.
            conversionspec is a list of 2-tuples containing a component index or name and a conversion """
            #TODO: A and B should only be calculated once
            #TODO: This does not take into account any existing products in the mixture
            #TODO: This works only for conversion specified in terms of reagents
            if len(conversionspec) < self.DOF:
                raise Exception("Not enough conversions specified.")
            C = numpy.zeros([len(conversionspec), self.S.shape[1]])
            Ic = C.copy()
            for i, (j, c) in enumerate(conversionspec):
                if type(j) is str:
                    j = self.compoundnames.index(j)
                C[i, j] = 1-c
                Ic[i, j] = 1
            A = numpy.vstack([self.S, C])
            B = numpy.vstack([self.S, Ic])
            # A ni = B nf
            nf, _, _, _ = scipy.linalg.lstsq(B, numpy.dot(A, self.N))
            # assert residuals are neglegable
            nf[nf<0] = 0
            return nf
            
        def equilibrium(self, initialconversion):
            """ Return equilibrium composition as minimum of Gibbs Energy """
            # guess initial conditions
            N0 = self.conversion(initialconversion)
            logN0 = numpy.log(N0)
                
            # This decorator modifies a function to be defined in terms of new variables
            def changevars(f):
                def newf(newX):
        #         print 'newX', newX
        #         print 'X', numpy.exp(newX)
                    r = f(numpy.exp(newX))
        #          print 'f(X)', r
                    return r
                return newf
        
            # Find optimal point in terms of a change of variables
            logN = scipy.optimize.fmin_slsqp(changevars(self.gibbs), 
                                            logN0, 
                                            f_eqcons=changevars(self.atombalance),
                                            acc = 1.0E-12)
    
            N = numpy.exp(logN)
            print ''
            print 'Calculated Optimmum Values'
            print N
            print ''
            print 'Calculated Optimum Function value'
            print m.gibbs(N)
                
            return N
       
       # Function for Cp integration as function of temperature
        def Cpint(T, C): 
           return CpConstants[0,C] + CpConstants[1,C]*(math.pow(10, -3))*T + CpConstants[2,C]*(math.pow(10, 5))*(math.pow(T, -2)) + CpConstants[3,C]*(math.pow(10, -6))*(math.pow(T, 2)) 
       
       
    # Here for a particular mixture:            
        # Initial values manually entered by users for each component
        # calculating Compound property as a function of Temperature
            # Constants for calculating Heat Capacity
                            # MgO(0)      H2O(1)      Mg(OH)2(2)    Al(OH)3(3)     Al2O3(4)      Mg(5)        O2(6)     Al(7)      H2(8)
    Aconstant = numpy.array([47.485,    186.884,     100.055,       30.602,        115.977,     -7.493,      29.790,    32.974,   22.496])
    Bconstant = numpy.array([4.648,     -464.247,    18.337,        209.786,       15.654,      256.809,    -6.177,     -20.677,  17.044])
    Cconstant = numpy.array([-10.340,   -19.565,     -25.255,       0,            -44.290,      0.011,      -0.021,     -4.138,   0.365])
    Dconstant = numpy.array([-0.268,    548.631,     -0.017,        0,            -2.358,       -35.618,    15.997,     23.753,   11.122])
    
    CpConstants = numpy.array([Aconstant, Bconstant, Cconstant, Dconstant]) 
    for C in range(0, nspecies):
        Cpv[position,C] = mixture.Cpint(T, C)
    
            # Enthalpy values at ambient conditions for each compound
                      # MgO(0)      H2O(1)      Mg(OH)2(2)    Al(OH)3(3)     Al2O3(4)      Mg(5)          O2(6)     Al(7)    H2(8)
    Ho = numpy.array([-601996,     -285970,      -924991,     -1295164,     -1676429,        0,             0,       0,       0])
    
    DHofMgO   = Ho[0] - Ho[5] - 0.5*Ho[6]
    DHofH2O   = Ho[1] - Ho[8] - 0.5*Ho[6]
    DHofAl2O3 = Ho[4] - 2*Ho[7] - (3/2)*Ho[6]
    DHofMgOH2 = Ho[2] - Ho[5] - Ho[6] - Ho[8]
    DHofAlOH3 = Ho[3] - Ho[7] - (3/2)*Ho[6] - (3/2)*Ho[8]

            # Entropy values at ambient conditions for each compound
                      # MgO(0)   H2O(1)    Mg(OH)2(2)    Al(OH)3(3)  Al2O3(4)   Mg(5)     O2(6)     Al(7)    H2(8)
    So = numpy.array([26.950,   69.950,     63.137,       71.128,     50.626,   32.535,  205.149,  28.275,   130.679])
    
    DSofMgO   = So[0] - So[5] - 0.5*So[6]
    DSofH2O   = So[1] - So[8] - 0.5*So[6]
    DSofAl2O3 = So[4] - 2*So[7] - (3/2)*So[6]
    DSofMgOH2 = So[2] - So[5] - Ho[6] - So[8]
    DSofAlOH3 = So[3] - So[7] - (3/2)*So[6] - (3/2)*So[8]
    
        # final Compound calculations as a function of Temperature
    for s in range(0, nspecies):  
        space = position - 1
        if position == 0:
            space = 0  
            # Heat Capacity as a function of temperature:
        Cpvalue[position,s] = Aconstant[s] + Bconstant[s]*(math.pow(10, -3))*T + Cconstant[s]*(math.pow(10, 5))*(math.pow(T, -2)) + Dconstant[s]*(math.pow(10, -6))*(math.pow(T, 2))
                # Straight Cp inegration for Enthalpy
        Cpintvalue[position, s] = Cpintvalue[space, s] + dT*Cpvalue[position, s]
        Cpintvalue[0,s] = Cpvalue[0,s]
                # Integradtion for Entropy
        CpintTvalue[position,s] = CpintTvalue[space,s] + dT*(Cpvalue[position, s])/T
        CpintTvalue[0,s] = Cpvalue[0,s]
        # Enthalpy of Formation as a function of temperature:
    for h in range(0, nspecies):
        if h==0:
            Hvalue[position,h] = DHofMgO + Cpintvalue[position,h] - Cpintvalue[position,5] - 0.5*Cpintvalue[position,6]
        if h==1:
            Hvalue[position,h] = DHofH2O + Cpintvalue[position,h] - Cpintvalue[position,8] - 0.5*Cpintvalue[position,6]
        if h==2:
            Hvalue[position,h] = DHofMgOH2 + Cpintvalue[position,h] - Cpintvalue[position,5] - Cpintvalue[position,6] - Cpintvalue[position,8]
        if h==3:
            Hvalue[position,h] = DHofAlOH3 + Cpintvalue[position,h] - Cpintvalue[position,7] - (3/2)*Cpintvalue[position,6] - (3/2)*Cpintvalue[position,8]
        if h==4:
            Hvalue[position,h] = DHofAl2O3 + Cpintvalue[position,h] - 2*Cpintvalue[position,7] - (3/2)*Cpintvalue[position,6]       
        # Entropy of Formation as a function of Temperature
    for s in range(0, nspecies):
        if s==0:
            Svalue[position,s] = DSofMgO + CpintTvalue[position,s] - CpintTvalue[position,5] - 0.5*CpintTvalue[position,6]
        if s==1:
            Svalue[position,s] = DSofH2O + CpintTvalue[position,s] - CpintTvalue[position,8] - 0.5*CpintTvalue[position,6]
        if s==2:
            Svalue[position,s] = DSofMgOH2 + CpintTvalue[position,s] - CpintTvalue[position,5] - CpintTvalue[position,6] - CpintTvalue[position,8]
        if s==3:
            Svalue[position,s] = DSofAlOH3 + CpintTvalue[position,s] - CpintTvalue[position,7] - (3/2)*CpintTvalue[position,6] - (3/2)*CpintTvalue[position,8]
        if s==4:
            Svalue[position,s] = DSofAl2O3 + CpintTvalue[position,s] - 2*CpintTvalue[position,7] - (3/2)*CpintTvalue[position,6]
#    print Svalue[:,0] 
        # Gibbs Free energy of Formation as a function of temperature
    for g in range(0, nspecies):
        if g==0:
            DHvalue = Hvalue[position,g] - Hvalue[position,5] - 0.5*Hvalue[position,6]
            DSvalue = Svalue[position,g] - Svalue[position,5] - 0.5*Svalue[position,6]
            Gvalue[position,g] = DHvalue - T*DSvalue
        if g==1:
            DHvalue = Hvalue[position,g] - Hvalue[position,8] - 0.5*Hvalue[position,6]
            DSvalue = Svalue[position,g] - Svalue[position,8] - 0.5*Svalue[position,6]
            Gvalue[position,g] = DHvalue - T*DSvalue
        if g==2:
            DHvalue = Hvalue[position,g] - Hvalue[position,5] - Hvalue[position,6] - Hvalue[position,8]
            DSvalue = Svalue[position,g] - Svalue[position,5] - Svalue[position,6] - Svalue[position,8]
            Gvalue[position,g] = DHvalue - T*DSvalue
        if g==3:
            DHvalue = Hvalue[position,g] - Hvalue[position,7] - (3/2)*Hvalue[position,6] - (3/2)*Hvalue[position,8]
            DSvalue = Svalue[position,g] - Svalue[position,7] - (3/2)*Svalue[position,6] - (3/2)*Svalue[position,8]
            Gvalue[position,g] = DHvalue - T*DSvalue
            
        if g==4:
            DHvalue = Hvalue[position,g] - 2*Hvalue[position,7] - (3/2)*Hvalue[position,6]
            DSvalue = Svalue[position,g] - 2*Svalue[position,7] - (3/2)*Svalue[position,6]
            Gvalue[position,g] = DHvalue - T*DSvalue
    


    m = mixture([[4.0,  compound('MgO',                                 -568343.0)],
                [1.0, compound('Al2O3',                                -1152420.0)],
                [8.0,  compound('H2O',                                  -237141.0)],
                [0,   compound('Mg(OH)2',                               -833644.0)],
                [0,   compound('Al(OH)3',                              -1835750.0)]])
        
        
        # find Gibbs energy surface for many conversion possibilities
    Nsteps = 10
    convrange = numpy.linspace(0.001, 0.999, Nsteps)
    gibbssurface = numpy.zeros((Nsteps, Nsteps))
    Finalvalue = numpy.array([])
    
    for iMgO, xMgO in enumerate(convrange):         # Calculate convrange for Mg
        xNaHCO3 = xMgO                              # Calculate conrange in Na
        for iAl2O3, xAl2O3 in enumerate(convrange): # Calculate convrange for Al
            N = m.conversion([('MgO', xMgO), 
                              ('Al2O3', xAl2O3)])
            gibbssurface[iMgO, iAl2O3] = m.gibbs(N)
            
        # To find the Minimum function value    
            Finalvalue = m.gibbs(N)
            if Finalvalue < Minvalue:
                Minvalue = Finalvalue
                Finalmole = N
                Temperature = T
    position = position + 1


print 

# For displaying data in Console Window    
print ''
print '*************Final Values***************'
print 'The optimum temperature is', Temperature
print 'The Minimum Function vaue is ', Minvalue
print 'The optimum final values are', Finalmole
print ''
for n, init, name in zip(N, m.N, m.compoundnames):
    print name, 'initial =', init, 'final =', n

# Plot results
if mustplot:
    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')
    Xaxis, Yaxis = numpy.meshgrid(convrange, convrange)
    Zaxis = gibbssurface
    surf = ax.plot_surface(Xaxis, Yaxis, Zaxis, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('Conversion of MgO')
    ax.set_ylabel('Conversion of Al2O3')
    ax.set_zlabel('Gibbs free energy (G/RT)')
    pl.show()


