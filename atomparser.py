#!/usr/bin/env python
# Module for parsing chemical formulae
# Author: Carl Sandrock

import re
import collections

tokenre = ['[A-Z][a-z]*',
           '\(',
           '\)',
           '[0-9]+',
           '\.']

class Group:
    def __init__(self, item=None, multiplier=1):
        self.multiplier = multiplier
        self.contents = []
        self.lastitem = None
        if item:
            self.add(item)

    def add(self, item):
        self.contents.append(item)
        self.lastitem = item

    def elements(self):
        contents = []
        for item in self.contents:
            if type(item) is str:
                contents += [item]
            else:
                contents += item.elements()
        return contents * self.multiplier

    def distinctelements(self):
        return set(self.elements())

    def count(self, element):
        return sum(1 for myelement in self.elements if myelement == element)

    def counts(self, elements=None):
        c = collections.Counter(self.elements())
        if elements is None:
            return c
        else:
            return [c[e] for e in elements]

    def canonicalstr(self, level=0):
        s = ''
        for item in self.contents:
            if type(item) is str:
                s += item
            else:
                s += item.canonicalstr(level+1)
        if level == 0:
            if self.multiplier > 1:
                s = str(self.multiplier) + s
        else:
            if len(self.contents)>1:
                s = '(' + s + ')'
            if self.multiplier > 1:
                s += str(self.multiplier)
        return s

    def __repr__(self):
        return 'Group(' + str(self.contents) + ', ' 'multiplier=' + str(self.multiplier) + ')'


def parse(tokens, level=0):
    currentgroup = Group()

    firsttoken = True
    while tokens:
        t, *tokens = tokens
        if t == '(':
            newgroup, tokens = parse(tokens, level+1)
            currentgroup.add(newgroup)
        elif t == ')':
            if level == 0:
                raise Exception('Parse error: Unmatched closing bracket')
            else:
                return currentgroup, tokens
        elif t == '.':
            newgroup, tokens = parse(tokens, level)
            currentgroup.add(newgroup)
        elif t.isdigit():
            if firsttoken:
                currentgroup.multiplier = int(t)
            else:
                currentgroup.lastitem.multiplier *= int(t)
        else:
            currentgroup.add(Group(t))
        firsttoken = False

    if level > 0:
        raise Exception('Parse error: Unmatched opening bracket')

    return currentgroup, tokens

def parseformula(formula):
    splitter = re.compile('(' + '|'.join(tokenre) + ')')
    tokens = [t for t in splitter.split(formula) if len(t)]
    thegroup, _ = parse(tokens)
    return thegroup

def test(formula):
    g = parseformula(formula)
    print(formula)
    print('='*len(formula))
    print('Parsed result:', g)
    print('Canonical:', g.canonicalstr())
    print('Counts:', g.counts())
    print()

if __name__=='__main__':
    testcases = ['MgOH(C(ClH3)2)3CH3',
                 'Mg(OH)3.2H2O',
                 '2Mg(OH)2.2H2O',
                 'Pb10Ag12']

    for t in testcases:
        test(t)
