'''
Created on Apr 25, 2018

@author: loitg
'''
import random
import pickle
import os
import string

from params import LogParam, RangeParam, GenerativeParam

class Params(object):
    MODE_CHANGING = 'tune'
    MODE_GENERATE = 'gen'
    
    def __init__(self):
        self.changables = {}
        self.generatives = {}
        self.mode = Params.MODE_CHANGING

    def genShortKeys(self):
        allowedKeys = list(string.ascii_lowercase)
        ret = ''
        keys = allowedKeys[:len(self.changables)]
        self.shortkey2name = {}
        for i, paraname in enumerate(self.changables.iterkeys()):
            self.changables[paraname].shortkey = keys[i]
            self.shortkey2name[keys[i]] = paraname
            ret += paraname + '--' + self.changables[paraname].shortkey + '\n'
        return ret
    
    def updateFromUser(self):
        k = raw_input('Select Param: ')
        inc = k.isupper()
        k = k.lower()
        if k not in self.shortkey2name: return
        param = self.changables[self.shortkey2name[k]]
        if inc:
            param.inc()
        else:
            param.dec()
                
    def new(self, name, initval = None, paramrange=None):
        if self.mode == Params.MODE_CHANGING:
            if name not in self.changables:
                assert initval is not None
                if paramrange is None:
                    self.changables[name] = LogParam(initval, dtype=type(initval))
                else:
                    self.changables[name] = RangeParam(initval, paramrange, dtype=type(initval))
            
            return self.changables[name]
        elif self.mode == Params.MODE_GENERATE:
            return self.generatives[name]
    
    def snapShot(self):
        for i, paraname in enumerate(self.changables.iterkeys()):
            if paraname not in self.generatives:
                self.generatives[paraname] = GenerativeParam()
            print('SNAP ' + paraname + '==' + str(self.changables[paraname].x))
            self.generatives[paraname].snap(self.changables[paraname].x)

    def startGenerative(self, mode):
        self.mode = Params.MODE_GENERATE
        for i, key in enumerate(self.changables.iterkeys()):
            self.generatives[key].makeDistributor()
         
    def save(self, path):
        if path:
            with open(path, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
                return True
        return False
    
    def load(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self = pickle.load(f)
                return True
        return False
    