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
    
if __name__ == '__main__':
    c9gen = CMND9Gen()
    
    idgen = c9gen.idnumber
    idrender = RenderText(cmndconfig.fontid)
    idsi = ShootEffect()
    
    dobgen = c9gen.dob
    dobrender = RenderText(cmndconfig.fontdob)
    dobsi = ShootEffect()
    
    root = '/home/loitg/Downloads/images_txt/'
    with open(root + 'anno-train.txt', 'w') as annotation_train:
        with open(root + 'anno-test.txt', 'w') as annotation_test:
            for i in range(300000):
                print i, '-----------------------'
                if np.random.rand() < 0.5:
                    rs, txt = createSample(idgen, idrender, idsi)
                else:
                    rs, txt = createSample(dobgen, dobrender, dobsi)
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
        
        

    
        
    
