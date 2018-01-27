'''
Created on Jan 24, 2018

@author: loitg
'''
import os, sys
from numpy import average
# Config PYTHONPATH and template folder relatively according to current file location
project_dir = os.path.dirname(__file__) + '/../'
sys.path.insert(0, project_dir)
# from synthgen import *
import cv2
import numpy as np
import pygame
from pygame import freetype
from text_utils import RenderFont
import rstr
from genline.receiptgenline import CapitalandGen
from genline.combiner import ListGenWithProb


DATA_PATH = '/home/loitg/Downloads/SynthText/data'
FONT_PATH = '/home/loitg/Downloads/fonts/fontss/receipts/'
fontlist = ['general_fairprice/PRINTF Regular.ttf', #chi so x0.85 (chux0.65-0.75)
             #'general_fairprice/Instruction.otf',
             #'general_fairprice/Instruction Bold.otf',
             'general_fairprice/LEFFC___.TTF', #chi chu x1.3-5 (x0.85)
             'general_fairprice/LEFFC2.TTF', #chi chu x1.3-5 (x0.85)
#              'general_fairprice/LiberationMono-Regular.ttf',
             'general_fairprice/Merchant Copy.ttf', #x0.8-9- (x1.15-) (x0.72)
#              'general_fairprice/Merchant Copy Wide.ttf',
#              'general_fairprice/Monaco.ttf',
#              'general_fairprice/typewcond_regular.otf',
#              'general_fairprice/typewcond_demi.otf',
#              'general_fairprice/saxmono.ttf',
#              'bedok/Galeries.ttf', 
#             'dotted/5by7_b.ttf',
#             'dotted/DOTMATRI.TTF',
#             'dotted/5by7.ttf',
            'dotted/fake receipt.ttf', # dung duoc (x0.65) (0.5) (0.8)
            'dotted/jd_lcd_rounded.ttf', # dung duoc x0.95 (x0.65) (0.8)
#             'dotted/epson1.ttf',
#             'westgate/PetMe1282Y.ttf',
#             'westgate/PetMe128.ttf',
#             'westgate/PetMe64.ttf',
            'westgate/PKMN-Mystery-Dungeon.ttf', # chi chu va so (NOT SYMBOL)
#             'westgate/PetMe.ttf',
#             'westgate/PetMe642Y.ttf',
#             'westgate/PetMe2X.ttf',
            'westgate/PetMe2Y.ttf', #dung duoc x1.43 (x0.8) x1.2
            'westgate/karmasut.ttf', # dung duoc (x0.75)
#             'westgate/SYDTWO_2.ttf',
             ]
loitgfonts = [
    (0.08,'general_fairprice/PRINTF Regular.ttf', 0.6,0.9),
    (0.08,'general_fairprice/LEFFC2.TTF', 0.8, 1.5),
    (0.37,'general_fairprice/Merchant Copy.ttf', 0.7, 1.0),
    (0.08,'general_fairprice/Merchant Copy.ttf', 1.0, 1.2),
    (0.08,'dotted/fake receipt.ttf', 0.45, 0.85),
    (0.08,'dotted/jd_lcd_rounded.ttf',0.65, 0.95),
    (0.08,'westgate/PKMN-Mystery-Dungeon.ttf',0.7,1.3),
    (0.08,'westgate/PetMe2Y.ttf',0.8,1.5),
    (0.08,'westgate/karmasut.ttf',0.6,0.8),
    ]
class LoitgFont(object):
    
    def __init__(self, fontpath, p, widthrange, size=100):
        self.prob = p
        self.fontpath = fontpath
        self.font = freetype.Font(fontpath, size=size)
        self.widthrange = widthrange
        self.font.underline = False
        self.font.strong = False
        self.font.oblique = False
        self.font.strength = False
        self.font.antialiased = True
        self.font.origin = True
        
    def getFont(self):
        return self.font
    
    def getRatio(self):
        return (self.widthrange[1] - self.widthrange[0])*np.random.sample() + self.widthrange[0]
    
    
class RenderText(object):
    def __init__(self):
        pygame.init()
        self.renderfont = RenderFont(DATA_PATH)
        self.height = 100
        self.pad_for_cut = 10
        self.rel_line_spacing = 1.0
        self.loitgfonts = []
        self.fonts = []
        probs = []
        for p, fontname, range0, range1 in loitgfonts:
            self.fonts.append(LoitgFont(FONT_PATH+fontname, p, (range0, range1)))
            probs.append(p)
        self.fontgen = ListGenWithProb(self.fonts, probs)

    def init_font(self, fontpath, spacing):
        font = freetype.Font(FONT_PATH + fontpath, size=self.height)
        font.underline = False
        font.strong = False
        font.oblique = False
        font.strength = False
        font.antialiased = True
        font.origin = True
        self.spacing = spacing
        return font    
    
    def genFont(self):
        return self.fontgen.loitgfont()
    
    def toMask(self, loitgfont, txt, otherlines=False):
        if loitgfont.fontpath == '/home/loitg/Downloads/fonts/fontss/receipts/westgate/PKMN-Mystery-Dungeon.ttf' and any(c in txt for c in ['<','>',';','!','$','#','&','/','`','~','@','%','^','*']):
            return None
        if loitgfont.fontpath == '/home/loitg/Downloads/fonts/fontss/receipts/general_fairprice/LEFFC2.TTF' and any(c.isdigit() for c in txt):
            return None
        above = rstr.rstr('ABC0123456789abcdef ', len(txt))
        below = rstr.rstr('ABC0123456789abcdef ', len(txt))
        multilines = above + '\n' + txt + '\n' + below
        if otherlines:
            txt_arr, txt, bbs = self.renderfont.render_multiline(loitgfont.font, multilines , 10, 0.1, 1)
        else:
            txt_arr, txt, bbs = self.renderfont.render_multiline(loitgfont.font, multilines , 10, 0.5, 1)
            
        newwidth = int(txt_arr.shape[1]*loitgfont.getRatio())
        txt_arr = cv2.resize(txt_arr,(newwidth, txt_arr.shape[0]))
        return txt_arr
 
if __name__ == '__main__':
    p, path, r0, r1 = loitgfonts[6]
    render = RenderText()
    lf = LoitgFont(FONT_PATH+path, p, (r0, r1),size=40)
    txt = ';:\'\",.<>/?'
    txt='`~!@#$%^&*()-=_+'
    txt='[]{}\|'
    txt = [x for x in txt]
    txt = '\n'.join(txt)
    print txt
    txt_arr, txt, bbs = render.renderfont.render_multiline(lf.font, txt , 10, 0.1, -1)
    cv2.imshow('m', txt_arr)
    cv2.waitKey(-1)
    
    exit(0)
    
    
    
    
    render = RenderText()
    clgen = CapitalandGen()
    txt = 'Staff: Asther Canoy0103344' #clgen.gen()
#    fontpath = fontslist[int(np.random.randint(0, len(fontslist)))]
    for _ in range(30):
        loitgfont = render.genFont()
        print loitgfont.fontpath
        m = render.toMask(loitgfont, txt=txt, otherlines = True)
        cv2.imshow('m', m)
        cv2.waitKey(-1)
        
    

        
    
