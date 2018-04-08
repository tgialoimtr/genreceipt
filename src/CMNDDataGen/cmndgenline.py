'''
Created on Mar 17, 2018

@author: loitg
'''
from genline.items import *
from genline.combiner import *
from genline.detailgen import *
from utils.params import cmndconfig

class CMND9Gen(Gen):
    def __init__(self):
        self.createDOB()
        self.createIDNumber()

    def createDOB(self):
        self.dob = DateGen(fromdate='1940-01-01', todate='1999-12-31', dateformat=cmndconfig.date)
        return self.dob
        
    def createIDNumber(self):
        self.idnumber = RegExGen(r'[0-3]\d{8}')
        return self.idnumber
    
if __name__ == '__main__':
    c9gen = CMND9Gen()
    for i in range(200):
        print '---' + c9gen.dob.gen() + '---'