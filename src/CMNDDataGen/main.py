'''
Created on Mar 17, 2018

@author: loitg
'''

import numpy as np
import cv2
from CMNDDataGen.cmndgenline import CMND9Gen
from TextRender.buildtextmask import RenderText
from utils.params import cmndconfig
from LightAndShoot.shooteffect import ShootEffect
from LightAndShoot.colorize3_poisson import Layer, Colorize
import math, random

def createTextMask(gener, render, si):
    txt = gener.gen()
    m, txt = render.toMask(render.genFont(), txt=txt, otherlines = True)
    return m, txt

def createSample(gener, render, si):
    textmask, txt = createTextMask(gener, render, si)
    if textmask is None:
        return None, None
    newwidth = int(32.0/textmask.shape[0]*textmask.shape[1])
    textmask = cv2.resize(textmask,(newwidth, 32))
    rs = si.effect(textmask)
    return rs, txt

class Stack:
    
    def __init__(self, width, height):
        self.ellprender = RenderText(cmndconfig.fontdob)
        self.dobrender = RenderText(cmndconfig.fontdob)
        self.idrender = RenderText(cmndconfig.fontid)
        
        self.width = width
        self.height = height

        self.colorize = Colorize()
        self.si = ShootEffect()
        
 
    @staticmethod
    def sineWave(x0, x1, y0, phase, amp, wavelength):
        x = np.arange(x0, x1, 2)
        B = np.arange(x0, x1, 2)
        for i, x0 in enumerate(x):
            B[i] = math.sin(2.0*math.pi*(x0-x[0])/wavelength + phase)
        B *= amp
        datapoints = np.vstack((x,B)).astype(np.int32).T
        return datapoints
    
    def buildGuillocheBG(self):
        pass
    
        return Layer()
    
    def buildEllipse(self, y0, dx, includeKey=True):
        mask, txt = self.ellprender.toMask(self.ellprender.genFont(), txt=txt, otherlines = True)
        
        return Layer()
    
    def buildOtherLines(self):
        pass
        
        return Layer()
    
    def buildDOB(self, txt):
        mask, txt = self.dobrender.toMask(self.dobrender.genFont(), txt=txt, otherlines = True)
        #move mask
        adapted_mask = None
        
        return Layer()
    
    def buildID(self, renderer, txt):
        mask, txt = self.idrender.toMask(self.idrender.genFont(), txt=txt, otherlines = True)
        #move mask
        adapted_mask = None
        
        return Layer()
    
    def buildGuillocheBGSo(self, dy, x0, y0, col):
        alpha = 0
        
        return Layer()
    
    def gen(self):
        bg_col_hsv = (random.randint(122,190), random.randint(2,40), random.randint(50,90))
        guilloche_col_hsv = bg_col_hsv + (0, random.randint(0,10), random.randint(-20,-5))
        text_col_hsv = (bg_col_hsv[0], bg_col_hsv[1]*random.uniform(1.3,2.2), bg_col_hsv[2]/random.uniform(1.5,2.5))
        sodo_col_hsv = (random.randint(330,350), random.randint(25, 47), random.randint(52,80))
        l_bg = Layer(alpha=255*np.ones((self.height, self.width, 3),'uint8'), color=bg_col)
        if np.random.rand() < 0.5:
            #DOB Layers
            layers = [self.buildDOB(),
                          self.buildEllipse(y0, dx),
                          self.buildGuillocheBG(),
                          self.buildOtherLines(),
                          l_bg]
        else:
            #ID Layers
            layers = [self.buildID(),
                          self.buildGuillocheBGSo(dy, x0, y0, col),
                          self.buildGuillocheBG(),
                          self.buildOtherLines(),
                          l_bg]
        blends = ['normal'] * len(layers)
        return self.colorize.merge_down(layers, blends)
   
if __name__ == '__main__':
    root = '/home/loitg/Downloads/images_txt/'
    cmndgen = Stack(320, 320)
    
    with open(root + 'anno-train.txt', 'w') as annotation_train:
        with open(root + 'anno-test.txt', 'w') as annotation_test:
            for i in range(300000):
                print i, '-----------------------'
                rs, txt = cmndgen.gen()
                if rs is None: continue
                txt = txt.strip()
#                 cv2.imwrite(root + str(i) + '.jpg', rs)
                print '@@@'+txt+'@@@'
                cv2.imshow('hihi', rs)
                cv2.waitKey(-1)
                
#                 if i < 295000:
#                     annotation_train.write('./' + str(i) + '.jpg ' + txt + '\n')
#                 else:
#                     annotation_test.write('./' + str(i) + '.jpg ' + txt + '\n')
        
        

    
        
    
