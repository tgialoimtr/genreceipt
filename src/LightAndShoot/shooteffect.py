
import os, sys
# Config PYTHONPATH and template folder relatively according to current file location
project_dir = os.path.dirname(__file__) + '/../'
sys.path.insert(0, project_dir)
# from synthgen import *
import cv2
import numpy as np
from Operations import Distort, RotateRange
from PIL import Image

class ShootEffect(object):
    def __init__(self):
        self.p_artifact = 0.3
        self.p_blur = 0.2
        self.p_distort = 0.3
        self.p_curved = 0.1
        self.otherline_noise = 0.4
        self.noise = 0.1
        self.fold_warp =  0.2
        self.p_matnet = 0
        self.p_ = 0
        self.distortor = Distort(0.8, 2,4, 4)
        self.rotator = RotateRange(0.8, 5,5)
        # resize, jpeg artifact , blur (horizen, vertical), matnet, distort/warp, curved line, noise(above/below), foldeffect, bg(logo))
        
    def matnet(self, mask):
        strength = np.random.rand()
        strength = mask.shape[1] / mask.shape[0] * 2 * strength ** 4
        strength = 8.4 * 0.5
        mask = inkeffect(mask, nplaces=int(strength), strength=strength/100.0, onlyshrink=False)
        return mask
    
    def blur(self, mask):
        bsz = 1.0 + 0.1*np.random.randn()
        sz = (np.random.randn(2)*10).astype(int)
        sz = 2*sz +1
        mask = cv2.GaussianBlur(mask,tuple(sz),bsz)
        return mask
 
    def fold(self, bg, fg):

        return bg, fg
    
    def addnoise(self, line):
        noises = np.random.randn(*line.shape[:2])*np.random.randint(1,10)
        noises = noises[:,:,np.newaxis] + np.random.randn(*line.shape)*np.random.randint(1,3)
        line = (line + noises)
        line = np.clip(line, 0, 255).astype(np.uint8)
        return line
    
    def combine(self, textmask, bg_cl, fg_cl):
        textmask = textmask/255.0
        rs = (((1-textmask)*1.0)[:,:,None] * bg_cl
                    + textmask[:,:,None]*fg_cl    )
        return np.clip(rs,0,255).astype(np.uint8)

    def lowresolution(self, img):
        realheight = np.random.randint(13,32)
        r = realheight*1.0/img.shape[0]
        img = cv2.resize(img, (0,0), fx = r, fy = r)
        img = cv2.resize(img, (0,0), fx = 1.0/r, fy = 1.0/r)
        return img
        
    def jpegartifact(self, img):
        q = np.random.randint(40,90)
        cv2.imwrite('/tmp/789789789.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
        img = cv2.imread('/tmp/789789789.jpg')
        return img
    
    def distort(self, img):
        pilimg = Image.fromarray(img)
        pilimg = self.rotator.perform_operation(pilimg)   
#         pilimg = self.distortor.perform_operation(pilimg)
             
        img = np.array(pilimg)
        return img
    

def inkeffect(oritext, nplaces=20, strength=0.1, onlyshrink=False):
    rmask = np.zeros_like(oritext, dtype=np.float)
    rs = np.zeros_like(oritext)
    (h,w) = oritext.shape[:2]
    
    print nplaces, strength
    for i in range(nplaces):
        r = np.random.randint(h/3.5)
        if not onlyshrink:
            inten = np.random.rand()*h*strength*2-h*strength
        else:
            inten = -np.random.rand()*h*strength
        y = np.random.randint(h)
        x = np.random.randint(w)
        cv2.circle(rmask,(x,y), r, inten, -1)
    
    bsz = 1.5 + 0.1*np.random.randn()
    rmask = cv2.GaussianBlur(rmask,(5,5),bsz)
     
#     cv2.imshow('h', rmask*1.0/np.amax(rmask))
#     cv2.waitKey(-1)
    
    for x in range(0,w):
        for y in range(0,h):
            r = int(rmask[y,x])
            if r == 0:
                rs[y,x] = oritext[y,x]
            elif r > 0:
                xmin = max(0, x - r)
                xmax = min(w-1, x + r)
                ymin = max(0, y - r)
                ymax = min(h-1, y + r)
                rs[y,x] = np.amax(oritext[ymin:ymax, xmin:xmax])
            elif r < 0:
                xmin = max(0, x + r)
                xmax = min(w-1, x - r)
                ymin = max(0, y + r)
                ymax = min(h-1, y - r)
                rs[y,x] = np.amin(oritext[ymin:ymax, xmin:xmax])

#     cv2.imshow('h', rs)
#     cv2.waitKey(-1)            
    
    return rs  


def init():
    si = ShootEffect()
    # Select colors
    bg_main_cl = np.random.randint(190,250) + np.random.randn(3)*np.random.randint(2,10)
    bg_main_cl = np.clip(bg_main_cl, 0, 255).astype(int)
    fg_main_cl = np.random.randint(2,150) + np.random.randn(3)*np.random.randint(2,10)
    fg_main_cl = np.clip(fg_main_cl, 0, 255).astype(int)
    # Create texts
    textmask = np.zeros((64, 270), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(textmask,'BEDOK MALL',(10,40), font, 1,255,2,cv2.LINE_AA)


    textmask = si.matnet(textmask)
    textmask = si.blur(textmask)
    rs = si.combine(textmask, bg_main_cl, fg_main_cl)
#     rs = si.lowresolution(rs)
#     rs = si.jpegartifact(rs)
    rs = si.distort(rs)

    cv2.imshow('ll',rs)
    cv2.waitKey(-1)
    
    print '------------'
    
    return None     
############################################################################    
if __name__ == '__main__':
    for i in range(30):
        line = init()
    
def feather(text_mask, min_h):
    # determine the gaussian-blur std:
    if min_h <= 15 :
        bsz = 0.25
        ksz=1
    elif 15 < min_h < 30:
        bsz = max(0.30, 0.5 + 0.1*np.random.randn())
        ksz = 3
    else:
        bsz = max(0.5, 1.5 + 0.5*np.random.randn())
        ksz = 5
    return cv2.GaussianBlur(text_mask,(ksz,ksz),bsz)
    