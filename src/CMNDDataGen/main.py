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
from TextRender.buildtextmask import rotate_bound
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
def hsv2bgr(col_hsv):
    print col_hsv
    assert len(col_hsv) == 3
    col = cv2.cvtColor(np.array(col_hsv, 'uint8').reshape((1,1,3)), cv2.COLOR_HSV2BGR)
    return col[0,0]

def bgr2hsv(col_hsv):
    print col_hsv
    assert len(col_hsv) == 3
    col = cv2.cvtColor(np.array(col_hsv, 'uint8').reshape((1,1,3)), cv2.COLOR_BGR2HSV)
    return col[0,0]
    
    
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
    def sineWave(x0, x1, y0, phase, amp, wavelength, angle=0):
        x = np.arange(x0, x1, 2)
        B = np.array(x, 'float')
        for i, x0 in enumerate(x):
            B[i] = math.sin(2.0*math.pi*(x0-x[0])/wavelength + phase)
        B *= amp
        B += y0
        datapoints = np.vstack((x,B))
        rotM = cv2.getRotationMatrix2D((x0,y0),angle,1)
        datapoints = rotM[:2,:2].dot(datapoints)
        datapoints = datapoints.astype(np.int32).T
        return datapoints

    def putMask(self, srcMask, pos):
        height, x0, y0, angel = pos    
        newwidth = int(srcMask.shape[1]*height*1.0/srcMask.shape[0])
        srcMask = cv2.resize(srcMask, (newwidth, height))
        srcMask = rotate_bound(srcMask, angel)
        adapted_mask = np.zeros((self.height + 2*srcMask.shape[0], self.width + 2*srcMask.shape[1]), 'uint8')
        x0 = x0 - srcMask.shape[1]/2 + srcMask.shape[1]
        y0 = y0 - srcMask.shape[0]/2 + srcMask.shape[0]
        adapted_mask[y0:(y0 + srcMask.shape[0]), x0:(x0+srcMask.shape[1])] = srcMask        
        adapted_mask = adapted_mask[srcMask.shape[0]:-srcMask.shape[0], srcMask.shape[1]:-srcMask.shape[1]]
        return adapted_mask
    
    def buildGuillocheBG(self):
        pass
    
        return Layer()
    
    def buildEllipse(self, y0, dx, includeKey=True):
        mask, txt = self.ellprender.toMask(self.ellprender.genFont(), txt=txt)
        
        return Layer()
    
    def buildOtherLines(self):
        pass
        
        return Layer()
    
    def buildDOB(self, txt):
        mask, txt = self.dobrender.toMask(self.dobrender.genFont(), txt=txt)
        #move mask
        adapted_mask = self.putMask(mask, (64, 20,20,5))
        
        return Layer(alpha= adapted_mask, color=(0,0,0))
    
    def buildID(self, txt, color, x0, y0, height, angel):
        mask, txt = self.idrender.toMask(self.idrender.genFont(), txt=txt)
        #move mask
        #move mask
        adapted_mask = self.putMask(mask, (height, x0, y0, angel))
        
        return Layer(alpha= adapted_mask, color=color)
    
    def buildGuillocheBGSo(self, dy, n, x0, y0, col, angel, thick):
        alpha= np.zeros((self.height, self.width),'uint8')
        for i in range(n):
            pts = self.sineWave(x0, self.width, i*dy + y0, 0, dy, 2*dy)
            cv2.polylines(alpha, [pts], isClosed=False, color=255, thickness=thick)
            
        
        rotM = cv2.getRotationMatrix2D((x0,y0),angel,1)
        alpha = cv2.warpAffine(alpha,rotM,(alpha.shape[1], alpha.shape[0]))
        
        return Layer(alpha=alpha, color=col)
    
    def gen(self):
        bg_col_hsv = (random.randint(122,190), random.randint(2,40), random.randint(50,90))
        guilloche_col_hsv = (bg_col_hsv[0], bg_col_hsv[1] + random.randint(0,10), bg_col_hsv[2] + random.randint(-20,-5))
        text_col_hsv = (bg_col_hsv[0], bg_col_hsv[1]*random.uniform(1.3,2.2), bg_col_hsv[2]/random.uniform(1.5,2.5))
        sodo_col_hsv = (random.randint(330,350), random.randint(25, 47), random.randint(52,80))
        l_bg = Layer(alpha=255*np.ones((self.height, self.width, 3),'uint8'), color=bg_col)
        
        bg_col = hsv2bgr(bg_col_hsv)
        guilloche_col = hsv2bgr(guilloche_col_hsv)
        text_col = hsv2bgr(text_col_hsv)
        sodo_col = hsv2bgr(sodo_col_hsv)
        
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
   

    def test(self):
        bg_col_hsv = (random.randint(122,190)*180/360, random.randint(2,40), random.randint(50,90))
        guilloche_col_hsv = (bg_col_hsv[0], bg_col_hsv[1] + random.randint(0,10), bg_col_hsv[2] + random.randint(-20,-5))
        text_col_hsv = (bg_col_hsv[0], bg_col_hsv[1]*random.uniform(1.3,2.2), bg_col_hsv[2]/random.uniform(1.5,2.5))
        sodo_col_hsv = (random.randint(330,350)*180/360, random.randint(25, 47), random.randint(45,75))
        
        bg_col = hsv2bgr(bg_col_hsv)
        guilloche_col = hsv2bgr(guilloche_col_hsv)
        text_col = hsv2bgr(text_col_hsv)
#         sodo_col_hsv = (190,190,190)
        sodo_col = hsv2bgr(sodo_col_hsv)
        
        print 'bgr', bg_col
        print 'oche', guilloche_col
        print 'sodo', sodo_col
        l_bg = Layer(alpha=255*np.ones((self.height, self.width),'uint8'), color=bg_col)
        layers = [self.buildGuillocheBGSo(28, 6, 60, 90, sodo_col, 0.5, 1),
                  self.buildID('0 2 4 5 2 6 3 5', sodo_col, 800,160, random.randint(170,230), -0.5),
                  l_bg]
        blends = ['normal'] * len(layers)
        return self.colorize.merge_down(layers, blends)
        
if __name__ == '__main__':
    root = '/home/loitg/Downloads/images_txt/'
    cmndgen = Stack(1900,320)
    for i in range(100):
        img = cmndgen.test()
        cv2.imshow('ff', img.color)
        cv2.waitKey(-1)
    exit(0)
    
    
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
        
        

    
        
    
