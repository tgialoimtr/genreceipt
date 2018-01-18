'''
Created on Jan 18, 2018

@author: loitg
'''
from items import *
from combiner import *
import copy

MALL_SUFFIX = [
    'SHOPPING CENTRE',
    'SHOPPING CTR',
    ]

MALLS = [
    'Tampines Mall',
    'TAMPINES CENTRAL',
    'STAR VISTA',
    'The Star Vista',
    'IMM BUILDING',
    'IMM MALL',
    'IMM BRANCH',
    'IMM BLDG',
    'BUKIT PANJANG PLAZA',
    'LOT 1',
    'LOT ONE',
    'JUNCTION 8',
    'BEDOK',
    'BEDOK MALL',
    'WESTGATE',
    'Sembawang',
    'PLAZA SINGAPURA',
    'BUGIS JUNCTION',
    'BUGIS PLUS',
    'RAFFLES CITY',
    'JCUBE'
    ]

STREET = [
    'TAMPINES CENTRAL', #4 
    'VISTA EXCHANGE GREEN', #1
    'JELEBU ROAD', #1
    'BISHAN PLACE', #9
    'NEW UPPER CHANGI', #311
    'GATEWAY DRIVE', #3
    'JURONG EAST CENTRAL', #2
    'SEMBAWANG', #604
    'ORCHARD ROAD', #68
    'VICTORIA', #201
    'VICTORIA', #230
    'NORTH BRIDGE'
    ]

LOCATIONS = [
    'CHOA CHU KANG'
    ]

class CapitalandGen(Gen):
    def __init__(self):
        r_money = r''
        r_pos_money = r''
        r_id = r''

        self.mall = 0
    def createStore(self):
        pass
    
    def createZipcode(self):
        rprefix = r'\(S\)|S|SE|SG|Singapore|SINGAPORE|Sing|S\'PORE|SIN|Sin'
        rzc = r'\d{6}'
        self.zipcode = BlankSpaceConbiner(RegExGen(rprefix), RegExGen(rzc))
        return self.zipcode
    
    def createBlock(self):
        numbergen = ListGenWithProb([r'[\dB]\d', r'[A-Z]\d', r'[A-Z0-9]\d\d'], [0.9, 0.05, 0.05])
        temp0 = KeyValueCombiner(StringGen('/'), numbergen, StringGen(''))
        temp1 = ComplexGen([numbergen,r'\-', numbergen, temp0],
                           [1.0,1.0,1.0,0.2])
        temp2 = KeyValueCombiner(StringGen('&'), temp1, StringGen(''))
        self.block = ComplexGen([r'\#', temp1, temp2],
                                  [1.0,1.0,0.2])
        return self.block
    
    def createMall(self):
        self.rawmall = StringListGen(MALLS)
        mall_suffix = RegExGen(r'(Shopping Center|Shopping Ctr)')
        return None
        
    def createAddr(self):
        self.addr = ComplexGen([r'No\.?[ ]?', self.createNumberGen(0.5,0.2,0.3),SepGen(',', allowFar=False, allowBlank=True), StringListGen(STREET), ' ', r'(St|st|ROAD|Road|Ave( \d)?|0\d|\d\d)'],
                               [0.2, 0.9, 1.0, 1.0, 1.0,0.4])
        return self.addr
             
    def createNumberGen(self, prob1=0.3, prob2=0.3, prob3=0.3):
        return ListGenWithProb([r'[0-9]', r'[1-9][0-9]', r'[1-9][0-9]{2}'],[prob1, prob2, prob3])
    
#     def 
#     def gen(self):
#         pass


if __name__ == '__main__':
    clgen = CapitalandGen()
    g = clgen.createAddr()
    for i in range(200):
        print '---' + g.gen() + '---'





