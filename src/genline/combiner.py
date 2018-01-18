'''
Created on Jan 17, 2018

@author: loitg
'''
from items import Gen, StringGen, SepGen
import numpy as np
from random import shuffle
from genline.items import RegExGen

class ComplexGen(Gen):
    
    def __init__(self, items, probs):
        self.items = items
        self.itemgens = []
        for item in items:
            if isinstance(item, str):
                self.itemgens.append(RegExGen(item))
            else:
                self.itemgens.append(item)
        self.probs = probs
        self.n = len(self.items)
        assert self.n == len(self.probs)
    
    def gen(self):
        rs = ''
        for i in range(self.n):
            p = np.random.rand()
            if p < self.probs[i]:
                rs += self.itemgens[i].gen()
        return rs

class ListGenWithProb(Gen):
    def __init__(self, items, probs):
        self.n = len(items)
        assert self.n == len(probs)
        self.genlist = []
        p = 0.0
        for i in range(self.n):
            if isinstance(items[i], str):
                g = RegExGen(items[i])
            else:
                g = items[i]
            p += probs[i]
            self.genlist.append((p, g))
        print p
        assert p >= 1.0
        
    def gen(self):
        pval = np.random.rand() #[0.0-1.0]
        for pval_ceil, gen in self.genlist:
            if pval_ceil > pval:
                return gen.gen()
    
    
    
class KeyValueCombiner(Gen):
    '''
    KeyValueCombiner
    '''

    def __init__(self, key, value, sepgen):
        '''
        key: First Gen
        value: Second Gen
        '''
        self.keygen = key
        self.valgen = value
        self.sepgen = sepgen

        
    def gen(self):
        t1 = self.keygen.gen()
        t2 = self.valgen.gen()
        s = self.sepgen.gen()
        return t1 + s + t2

class BlankSpaceConbiner(KeyValueCombiner):
    
    def __init__(self, key, value):
        self.sep = StringGen(' ')
        KeyValueCombiner.__init__(self, key, value, self.sep)
    
    def gen(self):
        rs = KeyValueCombiner.gen(self)
        return rs.strip()
        

class PermutationCombiner(Gen):
    '''
    PermutationCombiner
    ''' 
    
    def __init__(self, genlist, sepgen, lowerbound=1, upperbound=None):
        self.genlist = genlist
        self.lowerbound = lowerbound
        self.upperbound = upperbound if upperbound else len(genlist)
        self.sepgen = sepgen
        
    def gen(self):
        shuffle(self.genlist)
        
        numitems = np.random.randint(self.lowerbound-1, self.upperbound)
        s = self.sepgen.gen()
        rs = ''
        for i,g in enumerate(self.genlist):
            rs += g.gen()
            if i < numitems:
                rs += s
            else:
                break
        return rs


   
if __name__ == '__main__':
    g1 = RegExGen(r'\d')
    g2 = RegExGen(r'\D')
    print isinstance(g1, Gen)
    
    c = ListGenWithProb([g1, '=',  g2],[0.2,0.5,1.0])
    for i in range(20):
        print '----'+c.gen()+'------'
        
        
        
        
        
        
        
        
        
        
        
        