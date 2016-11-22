#!/usr/bin/env python

import atomparser
import numpy
from functools import reduce

compounds = ['MgO', 'Al2O3', 'H2O', 'Mg(OH)2', 'Al(OH)3']

parsedcompounds = [atomparser.parseformula(f) for f in compounds]
commonelements = reduce(set.union, (set(g.distinctelements()) for g in parsedcompounds))

S = numpy.array([g.counts(commonelements) for g in parsedcompounds]).T

print(compounds)
print(commonelements)
print(S)
