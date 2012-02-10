from __future__ import division
import scipy.optimize
import math
import numpy

T = 298.15
R = 8.314

compounds = ['MgO', 'Al2O3', 'H2O', 'MgOH2', 'AlOH3']
    
DGf = numpy.array([-568343.0,  # MgO   
                   -1563850.0, # Al2O3 
                   -273141.0,  # H2O   
                   -833644.0,  # MgOH2 
                   -1138706.0])# AlOH3 

NMgOo = 1
NAl2O3o = 0.5
NH2Oo = 4


xMg = 0.9
xAl = 0.8
NMgO = NMgOo - xMg
NAl2O3 = NAl2O3o -0.5*xAl
NH2O = NH2Oo - xMg - xAl*(6/2)*NAl2O3o
NMgOH2 = xMg*NMgOo
NAlOH3 = 2*xAl*NAl2O3o
mole = numpy.array([NMgO, NAl2O3, NH2O, NMgOH2, NAlOH3])
NTotal = sum(mole)
X = numpy.zeros([10], float)
Xo = mole
X = Xo

# # MgO Function
# def A(X): 
#     return math.exp(X[0])*(DGfMgO/(R*T) - math.log(math.exp(X[0]) + math.exp(X[1])) + math.exp(X[2]) + math.exp(X[3]) + math.exp(X[4]))
# # Al2O3 Function
# def B(X): 
#     return math.exp(X[1])*(DGfAl2O3/(R*T) - math.log(math.exp(X[0]) + math.exp(X[1])) + math.exp(X[2]) + math.exp(X[3]) + math.exp(X[4]))
# # H2O Function
# def C(X): 
#     return math.exp(X[2])*(DGfH2O/(R*T) - math.log(math.exp(X[0]) + math.exp(X[1])) + math.exp(X[2]) + math.exp(X[3]) + math.exp(X[4]))
# # Mg(OH)2
# def D(X): 
#     return math.exp(X[3])*(DGfMgOH2/(R*T) - math.log(math.exp(X[0]) + math.exp(X[1])) + math.exp(X[2]) + math.exp(X[3]) + math.exp(X[4]))
# # Al(OH)3
# def E(X): 
#     return math.exp(X[4])*(DGfAlOH3/(R*T) - math.log(math.exp(X[0]) + math.exp(X[1])) + math.exp(X[2]) + math.exp(X[3]) + math.exp(X[4]))
# # Total Objective Function
# def F(X):
#     return A(X) + B(X) + C(X) + D(X) + E(X)

A = numpy.tile(numpy.repeat([-1, 1], [1, Ncomps]), [Ncomps, 1])
def F(X):
    return sum(numpy.exp(X)*DGf/(R*T) - math.log(numpy.dot(A, numpy.exp(X))))

constraints = [math.exp(X[0]) + math.exp(X[3]) - 1,                                                          # Mg
               2*math.exp(X[1]) + math.exp(X[4]) - 0.5,                                                      # Al
               2*math.exp(X[2]) + 2*math.exp(X[3]) + 3*math.exp(X[4]) - 8,                                   # H
               math.exp(X[0]) + 3*math.exp(X[1]) + math.exp(X[2]) + 2*math.exp(X[3]) + 3*math.exp(X[4])]     # O

print (type(constraints))
answer = scipy.optimize.fmin_slsqp(F, Xo, eqcons = [constraints])

print answer
