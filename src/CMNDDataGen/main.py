'''
Created on Mar 17, 2018

@author: loitg
'''

import numpy as np
import cv2
import os, sys
from CMNDDataGen.cmndgenline import CMND9Gen
from TextRender.buildtextmask import RenderText
from utils.params import cmndconfig
from LightAndShoot.shooteffect import ShootEffect
from LightAndShoot.colorize3_poisson import Layer, Colorize
from TextRender.buildtextmask import rotate_bound
import math, random
from ParametersMatching.collections import Params

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
def hsv2bgr(col_hsv):
    assert len(col_hsv) == 3
    col = cv2.cvtColor(np.array(col_hsv, 'uint8').reshape((1,1,3)), cv2.COLOR_HSV2BGR)
    return col[0,0]

def bgr2hsv(col_hsv):
    assert len(col_hsv) == 3
    col = cv2.cvtColor(np.array(col_hsv, 'uint8').reshape((1,1,3)), cv2.COLOR_BGR2HSV)
    return col[0,0]
    
    
class Stack:
    
    def __init__(self, width, height, ):
        self.ellprender = RenderText(cmndconfig.fontdob)
        self.dobrender = RenderText(cmndconfig.fontdob)
        self.idrender = RenderText(cmndconfig.fontid)
        
        self.width = width
        self.height = height

        self.colorize = Colorize()
        self.si = ShootEffect()
        self.p = Params()
        
 
    @staticmethod
    def sineWave(x0, y0, length, amp, wavelength, angle=0, phase=0):
        x = np.arange(x0 - length/2, x0 + length/2, 2)
        B = np.array(x, 'float')
        o = np.ones_like(B, float)
        for i, xx in enumerate(x):
            B[i] = math.sin(2.0*math.pi*(xx-x[0])/wavelength + phase)
        B *= amp
        B += y0
        datapoints = np.vstack((x,B,o))
        rotM = cv2.getRotationMatrix2D((x0,y0),angle,1)
        datapoints = rotM.dot(datapoints)
        datapoints = datapoints[:2,].astype(np.int32).T
        return datapoints

    def putMask(self, srcMask, pos):
        height, x0, y0, angel = pos
        newwidth = int(srcMask.shape[1]*height*1.0/srcMask.shape[0])
        srcMask = cv2.resize(srcMask, (newwidth, height))
        srcMask = rotate_bound(srcMask, angel)
        x0 = x0 + srcMask.shape[1]/2
        y0 = y0 + srcMask.shape[0]/2
        adapted_mask = np.zeros((self.height + 2*srcMask.shape[0], self.width + 2*srcMask.shape[1]), 'uint8')
        adapted_mask[y0:(y0 + srcMask.shape[0]), x0:(x0+srcMask.shape[1])] = srcMask        
        adapted_mask = adapted_mask[srcMask.shape[0]:-srcMask.shape[0], srcMask.shape[1]:-srcMask.shape[1]]
        return adapted_mask
    
    def buildGuillocheBG(self, col):
        alpha= np.zeros((self.height, self.width),'uint8')
        amp = random.randint(self.height/6, self.height/3)
        wavelength = random.randint(self.height/8, self.height/3)
        thick = 1
        angle= random.uniform(0.0, 160.0)
        n = random.randint(20,50)
        y0 = random.randint(20,30)
        dy = (self.height - y0)/n
        x0 = random.randint(20,30)
        dx = (self.width - x0)/n
        
        for i in range(n):
            x0 += dx + random.randint(-2,2)
            y0 += dy + random.randint(-2,2)
            pts = self.sineWave(x0, y0, int(self.width*0.8), amp, wavelength, angle)
            cv2.polylines(alpha, [pts], isClosed=False, color=255, thickness=thick)
        
        return Layer(alpha=alpha, color=col)            
    
    def buildEllipse(self, y0, dx, includeKey=True):
        mask, txt = self.ellprender.toMask2(self.ellprender.genFont(), txt=txt)
        
        return Layer()
    
    def buildOtherLines(self, angle):
        mask, txt = self.idrender.toMask2(self.dobrender.genFont(), txt='CHUNG MINH NHAN DAN')
        #move mask
        #TODO randomize 
        adapted_mask = self.putMask(mask, (self.height/2, self.width/2,-self.height/4,angle))
        
        return Layer(alpha= adapted_mask, color=(0,0,255))
    
    def buildDOB(self, txt):
        mask, txt = self.dobrender.toMask2(self.dobrender.genFont(), txt=txt)
        #move mask
        adapted_mask = self.putMask(mask, (64, 20,20,5))
        
        return Layer(alpha= adapted_mask, color=(0,0,0))
    
    def buildID(self, txt, color, height, angel):
        x0 = self.p.new('center_x')
        y0 = self.p.new('center_y')
        afont = self.idrender.genFont()
        mask, txt = self.idrender.toMask2(afont, txt=txt)
        #move mask
        #move mask
        adapted_mask = self.putMask(mask, (height, x0.x, y0.x, angel))
        
        return Layer(alpha= adapted_mask, color=color)
    
    def buildGuillocheBGSo(self, height, angel, col):
        alpha= np.zeros((self.height, self.width),'uint8')
        dy = height*1.0/5
        x0 = self.p.new('center_x', self.width/2, paramrange=(0, self.width))
        y0 = self.p.new('center_y', self.height/2, paramrange=(0, self.height))
        thick = 1
        for i in range(6):
            pts = self.sineWave(x0.x, int(i*dy + y0.x - height/2), int(self.width*0.8), int(dy/2), int(9*dy))
            cv2.polylines(alpha, [pts], isClosed=False, color=255, thickness=thick)
            
        
        rotM = cv2.getRotationMatrix2D((x0.x,y0.x - height/2),angel,1)
        alpha = cv2.warpAffine(alpha,rotM,(alpha.shape[1], alpha.shape[0]))
        
        return Layer(alpha=alpha, color=col)
   

    def gen(self):
        bg_col_hsv = (random.randint(122,190)*180/360, random.randint(2,40)*255/100, random.randint(50,90)*255/100)
        guilloche_col_hsv = (bg_col_hsv[0], bg_col_hsv[1] + random.randint(0,10), bg_col_hsv[2] + random.randint(-20,-5))
        text_col_hsv = (bg_col_hsv[0], bg_col_hsv[1]*random.uniform(1.3,2.2), bg_col_hsv[2]/random.uniform(1.5,2.5))
        sodo_col_hsv = (random.randint(330,350)*180/360, random.randint(25, 47)*255/100, random.randint(45,75)*255/100)
        
        bg_col = hsv2bgr(bg_col_hsv)
        guilloche_col = hsv2bgr(guilloche_col_hsv)
        text_col = hsv2bgr(text_col_hsv)
#         sodo_col_hsv = (190,190,190)
        sodo_col = hsv2bgr(sodo_col_hsv)
        
#         print 'bgr', bg_col_hsv, bg_col
#         print 'oche', guilloche_col_hsv, guilloche_col
#         print 'sodo', sodo_col_hsv, sodo_col
        
        idheight = int(random.uniform(80.0/3, 80.0/2))
        idangle = random.uniform(-2, 2)
        
        lGuiBgSo = self.buildGuillocheBGSo(idheight, idangle, sodo_col)
        lId = self.buildID('012345678', sodo_col, int(idheight*1.2), idangle)
        lOtherLine = self.buildOtherLines(idangle)
        lGuiBG = self.buildGuillocheBG(guilloche_col)
        l_bg = Layer(alpha=255*np.ones((self.height, self.width),'uint8'), color=bg_col)

        lId.alpha = self.si.inkeffect(lId.alpha)
        lId.alpha = self.si.matnet(lId.alpha)
        lGuiBgSo.alpha = self.si.matnet(lGuiBgSo.alpha)


        layers = [lGuiBgSo, lId, lOtherLine, lGuiBG, l_bg]
        blends = ['normal'] * len(layers)
        idline = self.colorize.merge_down(layers, blends).color
        idline = self.si.addnoise(idline)
        idline = self.si.colorBlob(idline)
#         idline = self.si.lowresolution(idline)
        return idline
        
if __name__ == '__main__': 
    
    cmndgen = Stack(475,80)
    paramMgr = cmndgen.p   
    img = cmndgen.gen()
    #genShortKey
    desc = paramMgr.genShortKeys()
    #load Generatives if saved
    savedpath = '/tmp/params.pkl'
    if os.path.isfile(savedpath):
        paramMgr.load(savedpath)
    # TUNE/CHANGING pocess:
    while(True):
        print(desc)
        #updateFromUser
        paramMgr.updateFromUser()
        #apply and show 
        img = cmndgen.gen()
        cv2.imshow('hihi', img)
        k = cv2.waitKey(-1)
        if k == ord('1'):
            paramMgr.snapShot()
        elif k == ord('2'):
            break
            
    paramMgr.save(savedpath)
    
    sys.exit(0)  
        
    
        
     
    # GEN process:
    paramMgr.startGenerative()       
            
            
            
    root = '/home/loitg/Downloads/images_txt/'
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
        
        

    
        
    
