#!/usr/bin/env python

# Calculate mixture equilibriums using minimisation of the Gibbs energy

# 201111 Significantly reworked by Carl Sandrock
# 201110 Originally by Hendrik Venter

from __future__ import division
import scipy.optimize
import scipy.linalg
import math
import numpy
import atomparser
import sys
from matplotlib import pyplot as pl
from compiler.ast import For

T = 298.15
R = 8.314
RT = R*T
mustplot = True

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
                print 'newX', newX
                print 'X', numpy.exp(newX)
                r = f(numpy.exp(newX))
                print 'f(X)', r
                return r
            return newf
        
        # Find optimal point in terms of a change of variables
        logN = scipy.optimize.fmin_slsqp(changevars(self.gibbs), 
                                         logN0, 
                                         f_eqcons=changevars(self.atombalance))
        N = numpy.exp(logN)
        return N

# Here for a particular mixture:        
#m = mixture([[1.,  compound('MgO',      -568343.0)],
#             [0.5, compound('Al2O3',   -1563850.0)],
#             [4.,  compound('H2O',      -273141.0)],
#             [0,   compound('Mg(OH)2',  -833644.0)],
#             [0,   compound('Al(OH)3', -1138706.0)]])

m = mixture([[1.,  compound('MgO',      -568343.0)],
             [0.5, compound('Al2O3',    -156385.0)],
             [4.,  compound('H2O',      -273141.0)],
             [0,   compound('Mg(OH)2',  -833644.0)],
             [0,   compound('Al(OH)3', -1138706.0)]])

# find Gibbs energy surface for many conversion possibilities
print "Plotting" 
Nsteps = 100 
convrange = numpy.linspace(0.01, 0.99, Nsteps)
gibbssurface = numpy.zeros((Nsteps, Nsteps))
for iMgO, xMgO in enumerate(convrange):
    for iAl2O3, xAl2O3 in enumerate(convrange):
        N = m.conversion([('MgO', xMgO), 
                          ('Al2O3', xAl2O3)])
        gibbssurface[iMgO, iAl2O3] = m.gibbs(N)

# Try to find equilibrium
# Initial conversion guess
print "Optimizing"
X0 = [0.8, 0.9]
conversionspec = zip(['MgO', 'Al2O3'], X0)
# Solve to equilibrium
N = m.equilibrium(conversionspec)
# Calculate new conversion
X = 1. - N/m.N

# Plot results
if mustplot:
    pl.contour(convrange, convrange, gibbssurface)
    pl.plot(X0[0], X0[1], 'bo', X[0], X[1], 'rx')
    pl.xlabel('Conversion of MgO')
    pl.ylabel('Conversion of Al2O3')
    pl.legend(['Starting point', 'After optimization'], loc='best')
    pl.show()

for n, init, name in zip(N, m.N, m.compoundnames):
    print name, init, n
