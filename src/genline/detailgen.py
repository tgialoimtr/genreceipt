'''
Created on Jan 17, 2018

@author: loitg
'''
import numpy as np
from items import StringListGen, Gen
from faker import Faker
import re
import random


def replaceZeroWithProb(oridatetime, replaceProb=0.5):
    if np.random.rand() < replaceProb:
        result = re.sub(r'0(\d\D)', r'\1', oridatetime)
        print oridatetime
        print result
        return result
    else:
        return oridatetime

def unicode2ascii(text):
    ret = ''.join(i for i in text if ord(i)<128)
    return ret.encode('utf-8')

def changeCaseWithProb(oriword, upper=0.5, lower=0.25): #Capitalize = 1.0 - upper - lower
    p = np.random.rand()
    if upper > p:
        return oriword.upper()
    elif upper + lower > p:
        return oriword.lower()
    else:
        return oriword.capitalize()

class DateGen(Gen):
    '''
    Date
    '''
    def __init__(self):
        '''
        Constructor
        '''
        self.faker = Faker()
        self.dateformat = StringListGen([
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%d %b %Y",
            "%d %B %Y",
            "%d/%m/%y",
            ])
        
    def gen(self):
        d = self.faker.future_date(end_date="+350d")
        fm = self.dateformat.gen()
        d = d.strftime(fm)
        d = replaceZeroWithProb(d, 0.99)
        d = changeCaseWithProb(d, upper=0.5, lower=0.5)
        return d


class TimeGen(Gen):
    '''
    Time
    '''
    def __init__(self):
        '''
        Constructor
        '''
        self.faker = Faker()
        self.timeformat = StringListGen([
            "%H:%M:%S",
            "%I:%M:%S %p",
            "%d/%m/%Y"
            ])
        
    def gen(self):
        d = self.faker.future_date(end_date="+350d")
        fm = self.dateformat.gen()
        d = replaceZeroWithProb(d.strftime(fm))
        return d


   
class StoreGen(Gen): 
    
    def __init__(self, storelistpath):
        f = open(storelistpath, 'r')
        self.storeset = set()
        for line in f:
            temp = unicode2ascii(line).strip()
            if len(temp) > 4:
                self.storeset.add(temp)

    def gen(self):
        storestr = random.sample(self.storeset, 1)[0]
        storewords = storestr.split(' ')
        if len(storewords) > 2:
            pos = np.random.randint(len(storewords)-1)
            rs = storewords[pos] + ' ' + storewords[pos+1]
        else:
            rs = storestr
        return changeCaseWithProb(rs, upper=0.75, lower=0.0)

 
class ParragraphGen(Gen): 
    
    def __init__(self, dictpath):
        f = open(dictpath, 'r')
        self.wordset = set()
        temp = set()
        for line in f:
            temp.update(set(line.split(' ')))
        for w in temp:
            w = w.strip()
            if len(w) > 0:
                self.wordset.add(w)
    
    def gen(self):
        words = random.sample(self.wordset, 2)
        return ' '.join(words)
    
          
if __name__ == '__main__':
    g = ParragraphGen('/home/loitg/workspace/genreceipt/resource/parragraph.txt')
    for i in range(20):
        print '----'+g.gen()+'------'
            
            
        