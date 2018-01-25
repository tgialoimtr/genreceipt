'''
Created on Jan 24, 2018

@author: loitg
'''
import os, sys
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


DATA_PATH = '/home/loitg/Downloads/SynthText/data'
FONT_PATH = '/home/loitg/Downloads/fonts/fontss/receipts/'
fontslist = ['general_fairprice/PRINTF Regular.ttf', # Ngieng va CHU IN
             'general_fairprice/Instruction.otf', # Nghieng va CHU IN, so 071 rat NGU
             'general_fairprice/Instruction Bold.otf',
             'general_fairprice/LEFFC___.TTF',
             'general_fairprice/LEFFC2.TTF',
             'general_fairprice/LiberationMono-Regular.ttf',
             'general_fairprice/Merchant Copy.ttf',
             'general_fairprice/Merchant Copy Wide.ttf',
             'general_fairprice/Monaco.ttf',
             'general_fairprice/typewcond_regular.otf',
             'general_fairprice/typewcond_demi.otf',
             'general_fairprice/saxmono.ttf',
             ]


class RenderText(object):
    def __init__(self):
        pygame.init()
        self.renderfont = RenderFont(DATA_PATH)
        self.height = 100
        self.pad_for_cut = 10
        self.rel_line_spacing = 1.0

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
    
    def toMask(self, font, txt, otherlines=False):
        print len(txt), txt        
        above = rstr.rstr('ABC0123456789abcdef ', len(txt))
        below = rstr.rstr('ABC0123456789abcdef ', len(txt))
        multilines = above + '\n' + txt + '\n' + below
        if otherlines:
            txt_arr, txt, _ = self.renderfont.render_multiline(font, multilines , 10, 0.1, 1)
        else:
            txt_arr, txt, _ = self.renderfont.render_multiline(font, multilines , 10, 0.5, 1)        
        cv2.imshow('mm', txt_arr)
        cv2.waitKey(-1)
        return txt_arr
 
if __name__ == '__main__':
    render = RenderText()
    clgen = CapitalandGen()
    txt = '012379 Sing LOT 1 Tampines JUNCTION 8' #clgen.gen()
#    fontpath = fontslist[int(np.random.randint(0, len(fontslist)))]n
    for fontpath in fontslist:
        print fontpath
        m = render.toMask(render.init_font(fontpath=fontpath,spacing=0), txt=txt, otherlines = False)

        
    
