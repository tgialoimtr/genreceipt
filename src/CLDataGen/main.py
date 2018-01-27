'''
Created on Jan 24, 2018

@author: loitg
'''

import numpy as np
import rstr
import cv2
from genline.receiptgenline import CapitalandGen
from LightAndShoot.buildtextmask import RenderText
from LightAndShoot.shooteffect import ShootEffect
from time import time

def createTextMask(clgen, render, si):
#     txt = rstr.rstr('01', np.random.randint(3,8))
    tt = time()
    txt = clgen.gen()
    if len(txt) < 2 or len(txt) > 30:
        return None, None
#     print 'GenLine', time() - tt
#     print 'Text: ', txt
#     fontpath = 'general_fairprice/PRINTF Regular.ttf'
    tt = time()
    if np.random.rand() < 0.33:
        otherlines = True
    else:
        otherlines = False
    m = render.toMask(render.genFont(), txt=txt, otherlines = otherlines)
#     print 'ToMask', time() - tt
    return m, txt

def createSample(clgen, render, si):
    tt1=time()
    # Create texts
    textmask, txt = createTextMask(clgen, render, si)
    if textmask is None:
        return None, None
    newwidth = int(32.0/textmask.shape[0]*textmask.shape[1])
    textmask = cv2.resize(textmask,(newwidth, 32))
    
    tt=time()
    rs = si.effect(textmask)
#     print 'ShootEffect', time() - tt
#     print 'All', time() - tt1
#     print '-------------------'
    return rs, txt
    
if __name__ == '__main__':
    si = ShootEffect()
    render = RenderText()
    clgen = CapitalandGen()
    root = '/home/loitg/Downloads/images2/'
    with open(root + 'anno-train.txt', 'w') as annotation_train:
        with open(root + 'anno-test.txt', 'w') as annotation_test:
            for i in range(100):
                print i, '-----------------------'
                rs, txt = createSample(clgen, render, si)
                if rs is None: continue
                cv2.imwrite(root + str(i) + '.jpg', rs)
                if i < 70:
                    annotation_train.write('./' + str(i) + '.jpg ' + txt + '\n')
                else:
                    annotation_test.write('./' + str(i) + '.jpg ' + txt + '\n')
        
        

    
        
    