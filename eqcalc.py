#!/usr/bin/env python

# Calculate mixture equilibriums using minimisation of the Gibbs energy

# 201110 Originally by Hendrik Venter
# 201111 Significantly reworked by Carl Sandrock

from __future__ import division
import scipy.optimize
import scipy.linalg
import scipy.integrate
import math
import numpy
import atomparser
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib import pyplot as pl

smallvalue = 1e-10

nspecies = 9                    # number of species precent
Tstart = 298
Tfinal = 473
R = 8.314
Steps = Tfinal - Tstart

Temprange = numpy.linspace(Tstart, Tfinal, Steps)
dT = Temprange[1] - Temprange[0]
length = numpy.size(Temprange)

Cpv = numpy.zeros([length,nspecies])
Cpvalue = Cpv.copy()
Cpintvalue = Cpv.copy()
CpintTvalue = Cpv.copy()
Hvalue = Cpv.copy()
Svalue = Cpv.copy()
Gvalue = Cpv.copy()
DHvalue = Cpv.copy()
DSvalue = Cpv.copy()

position = 0
mustplot = True    # 1 for Plotting, 0 for not plotting
Minvalue = 0    # Used to calculate Minimum function value

class compound:            
    """ Basic container for compound properties """
    def __init__(self, name, DGf, A, B, C, D, Ho, So):
        self.name = name
        self.DGf = DGf
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Ho = Ho
        self.So = So
        self.parsed = atomparser.parseformula(name)
    
    # Function for Cp as function of temperature
    def Cp(self, T): 
        return self.A + self.B*1e-3*T + self.C*1e5*T**-2 + self.D*1e-6*T**2

    def Cpint(self, Ta, Tb):
        integral, _ = scipy.integrate.quad(self.Cp, Ta, Tb)
        return integral
    
    def CpintT(self, Ta, Tb):
        integral, _ = scipy.integrate.quad(lambda T: self.Cp(T)/T, Ta, Tb)
        return integral

    def __repr__(self):
        return ("compound('%s'" + ", %f"*7) % (self.name, self.DGf, self.A, self.B, self.C, self.D, self.Ho, self.So) + ")"
    
def parse(row):
    return [row[0]] + map(float, row[1:])    

class database:
    def __init__(self, filename):
        import csv
        infile = csv.reader(open(filename))
        infile.next()
        self.compounds = [compound(*parse(row)) for row in infile]
        self.names = [c.name for c in self.compounds]
    
    def __getitem__(self, name):
        return self.compounds[self.names.index(name)]
    
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
                r = f(numpy.exp(newX))
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
        print self.gibbs(N)
            
        return N

# Read properties from file
d = database('compounds.csv')

#TODO: Should be able to build this from the compound list
#     0   1   2   3    4     5    6    7   8 
M = numpy.array([[1,  0,  0,  0,   0,   -1, -1/2,  0,  0],
                 [0,  1,  0,  0,   0,    0, -1/2,  0, -1],
                 [0,  0,  1,  0,   0,   -1,   -1,  0, -1],
                 [0,  0,  0,  1,   0,    0, -3/2, -1, -3/2],
                 [0,  0,  0,  0,   1,    0, -3/2, -2,  0],
                 [0,  0,  0,  0,   0,    1,    0,  0,  0],
                 [0,  0,  0,  0,   0,    0,    1,  0,  0],
                 [0,  0,  0,  0,   0,    0,    0,  1,  0],
                 [0,  0,  0,  0,   0,    0,    0,  0,  1]])

Ho = numpy.array([c.Ho for c in d.compounds])
So = numpy.array([c.So for c in d.compounds])
DHof = M.dot(Ho)
DSof = M.dot(So)

m = mixture([[4.0, d['MgO']], 
             [1.0, d['Al2O3']], 
             [8.0, d['H2O']],    
             [0.0, d['Mg(OH)2']],
             [0.0, d['Al(OH)3']]])

for i, T in enumerate(Temprange):
    for j, c in enumerate(d.compounds):
        # Heat Capacity as a function of temperature:
        Cpv[i, j] = c.Cpint(Tstart, T)
        Cpintvalue[i, j] = c.Cpint(Tstart, T)
        CpintTvalue[i, j] = c.CpintT(Tstart, T)
    # final Compound calculations as a function of Temperature
            
        
    # Enthalpy of Formation as a function of temperature:
    Hvalue[i, :] = DHof + M.dot(Cpintvalue[i, :])
    Svalue[i, :] = DSof + M.dot(CpintTvalue[i, :])

    DHvalue[i, :] = M.dot(Hvalue[i, :])
    DSvalue[i, :] = M.dot(Svalue[i, :])

    # Gibbs Free energy of Formation as a function of temperature
    Gvalue[i, :] = DHvalue[i, :] - T*DSvalue[i, :]
        
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
