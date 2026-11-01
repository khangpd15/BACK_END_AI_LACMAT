"""Torch-only class-agnostic eye detector; no face, pupil or folder-label input."""
import math
import numpy as np
import torch
from torch import nn
from PIL import Image
from .resnet_reference import resnet

POLICY = 'learned_eye_roi_eso_exo_v1'
INPUT_SIZE = (384, 256)
STRIDE = 8


class EyeDetector(nn.Module):
    def __init__(self, initializer=None):
        super().__init__()
        base = resnet('resnet50')
        if initializer is not None:
            base.load_state_dict(initializer)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1, self.layer2, self.layer3 = base.layer1, base.layer2, base.layer3
        def lateral(ch):
            return nn.Sequential(nn.Conv2d(ch, 64, 1), nn.GroupNorm(8,64), nn.ReLU())
        self.lat2, self.lat3 = lateral(512), lateral(1024)
        self.head = nn.Sequential(nn.Conv2d(128,128,3,padding=1),nn.GroupNorm(8,128),nn.ReLU(),
                                  nn.Conv2d(128,64,3,padding=1),nn.GroupNorm(8,64),nn.ReLU())
        self.objectness = nn.Conv2d(64,1,1)
        self.distances = nn.Conv2d(64,4,1)
        self.offsets = nn.Conv2d(64,2,1)
        nn.init.constant_(self.objectness.bias, -2.19)

    def forward(self, x):
        a = self.layer1(self.stem(x)); b = self.layer2(a); c = self.layer3(b)
        features = self.head(torch.cat((self.lat2(b), nn.functional.interpolate(
            self.lat3(c),size=b.shape[-2:],mode='bilinear',align_corners=False)),1))
        return self.objectness(features), self.distances(features).sigmoid(), self.offsets(features).sigmoid()


def prepare(image):
    width,height = image.size; iw,ih = INPUT_SIZE
    scale = min(iw/width,ih/height); rw,rh = max(1,round(width*scale)),max(1,round(height*scale))
    left,top=(iw-rw)//2,(ih-rh)//2
    canvas=Image.new('RGB',INPUT_SIZE,(124,116,104))
    canvas.paste(image.convert('RGB').resize((rw,rh),Image.Resampling.BILINEAR),(left,top))
    array=np.array(canvas,dtype=np.float32)/255
    array=(array-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
    return array.transpose(2,0,1).copy(),dict(scale_x=rw/width,scale_y=rh/height,left=left,top=top,
                                          resized_width=rw,resized_height=rh,source_width=width,source_height=height)


def overlap(a,b):
    area=lambda r:max(0,r[2]-r[0])*max(0,r[3]-r[1])
    inter=area([max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])])
    return inter/max(1e-8,area(a)+area(b)-inter)


def decode(outputs, transform, threshold=.5):
    logits,distances,offsets=outputs
    probability=logits.sigmoid(); maxima=probability==nn.functional.max_pool2d(probability,3,1,1)
    values=(probability*maxima)[0,0].flatten();count=min(20,values.numel())
    scores,indices=values.topk(count);height,width=logits.shape[-2:];candidates=[];iw,ih=INPUT_SIZE
    for score,index in zip(scores.cpu().tolist(),indices.cpu().tolist()):
        if score<threshold:continue
        y,x=divmod(index,width);dx,dy=offsets[0,:,y,x].detach().cpu().tolist()
        l,t,r,b=distances[0,:,y,x].detach().cpu().tolist();cx,cy=(x+dx)*STRIDE,(y+dy)*STRIDE
        mapped=[(cx-l*iw-transform['left'])/transform['scale_x'],
                (cy-t*ih-transform['top'])/transform['scale_y'],
                (cx+r*iw-transform['left'])/transform['scale_x'],
                (cy+b*ih-transform['top'])/transform['scale_y']]
        x0,y0,x1,y1=mapped;sw,sh=transform['source_width'],transform['source_height']
        if not all(math.isfinite(v) for v in mapped) or x1-x0<8 or y1-y0<8:continue
        clipped=[max(0,x0),max(0,y0),min(sw,x1),min(sh,y1)]
        if max(0,clipped[2]-clipped[0])*max(0,clipped[3]-clipped[1])<.9*(x1-x0)*(y1-y0):continue
        if any(overlap(clipped,c['bbox'])>.3 for c in candidates):continue
        candidates.append(dict(bbox=[max(0,math.floor(clipped[0])),max(0,math.floor(clipped[1])),
                                     min(sw,math.ceil(clipped[2])),min(sh,math.ceil(clipped[3]))],score=float(score)))
    return candidates


def select_pair(candidates,width,height):
    pairs=[]
    for i,a in enumerate(candidates):
        for b in candidates[i+1:]:
            left,right=sorted((a,b),key=lambda c:c['bbox'][0]+c['bbox'][2]);l,r=left['bbox'],right['bbox']
            lw,rw=l[2]-l[0],r[2]-r[0];lh,rh=l[3]-l[1],r[3]-r[1]
            separation=((r[0]+r[2])-(l[0]+l[2]))/2
            if not (.15*width<=separation<=.85*width):continue
            if min(lw,rw)/max(lw,rw)<.45 or abs((l[1]+l[3]-r[1]-r[3])/2)>.5*max(lh,rh):continue
            if overlap(l,r)>.05:continue
            pairs.append((left,right))
    if len(pairs)!=1:return {},dict(status='ambiguous' if pairs else 'not_found',valid_pair_count=len(pairs))
    return {side:c['bbox'] for side,c in zip(('image_left','image_right'),pairs[0])},dict(
        status='paired_candidates',valid_pair_count=1,pair_scores=[c['score'] for c in pairs[0]])


class LearnedEyeROIs:
    def __init__(self,checkpoint,device='cpu'):
        data=torch.load(checkpoint,map_location=device,weights_only=True)
        if data.get('policy')!=POLICY:raise ValueError('Unknown eye detector checkpoint')
        self.model=EyeDetector().to(device);self.model.load_state_dict(data['state_dict']);self.model.eval()
        self.device=device;self.threshold=float(data['score_threshold'])

    def detect(self,image):
        array,transform=prepare(image)
        with torch.inference_mode():outputs=self.model(torch.from_numpy(array)[None].to(self.device))
        candidates=decode(outputs,transform,self.threshold);boxes,info=select_pair(candidates,*image.size)
        info.update(method=POLICY,policy=POLICY,candidates=candidates,score_threshold=self.threshold,
                    transform=transform,roi_padding_applied=True,source_pixels_only=True,
                    human_review_required=True,anatomical_sides_verified=False,
                    score_note='Uncalibrated detector scores, not disease probabilities')
        return boxes,info
