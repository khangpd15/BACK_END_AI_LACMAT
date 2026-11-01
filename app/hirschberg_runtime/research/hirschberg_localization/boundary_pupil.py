import torch,numpy as np,cv2
from PIL import Image
from app.hirschberg_runtime.research.hirschberg_localization.eyelid_segmentation import EyelidUNet
from torch import nn
class BoundaryModel(EyelidUNet):
 def __init__(self):
  super().__init__();self.visibility=nn.Linear(64,3)
 def forward(self,x):
  a=self.e1(x);b=self.e2(nn.functional.max_pool2d(a,2));c=self.e3(nn.functional.max_pool2d(b,2))
  d=self.d2(torch.cat((nn.functional.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b),1))
  mask=self.head(self.d1(torch.cat((nn.functional.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a),1)))
  return mask,self.visibility(c.mean((2,3)))

def predict(model,image,box,side,iris_four,torch,device,visualize=True):
 if not box:return None,{}, {'reason':'missing_roi','branch':'pupil_boundary'}
 x0,y0,x1,y1=box
 a=np.array(image.convert('RGB').crop(box).resize((192,192),Image.Resampling.BILINEAR))
 with torch.no_grad():logits,flags=model(torch.from_numpy(a).permute(2,0,1)[None].float().to(device)/255)
 mask=(logits.sigmoid()[0,0].cpu().numpy()>=.5).astype('uint8');scores=flags.softmax(1)[0].cpu().tolist();status=['visible','partial','hidden'][int(np.argmax(scores))]
 raw={'branch':'pupil_boundary','visibility':status,'visibility_probabilities':scores,'visibility_reliable':False,'ellipse_boundary_source':None,'peak_xy':None,'in_crop_and_source':False}
 cs,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
 if status!='visible' or not cs:return None,{},raw
 c=max(cs,key=cv2.contourArea)
 if len(c)<5 or cv2.contourArea(c)<3:return None,{},raw
 (cx,cy),(w,h),ang=cv2.fitEllipse(c);p=[x0+cx*(x1-x0)/192,y0+cy*(y1-y0)/192]
 valid=all(np.isfinite(p)) and x0<=p[0]<x1 and y0<=p[1]<y1
 t=np.linspace(0,2*np.pi,100);u=w/2*np.cos(t);v=h/2*np.sin(t);ang=np.deg2rad(ang)
 boundary=np.stack((x0+(cx+u*np.cos(ang)-v*np.sin(ang))*(x1-x0)/192,y0+(cy+u*np.sin(ang)+v*np.cos(ang))*(y1-y0)/192),1).tolist()
 raw.update(peak_xy=p,in_crop_and_source=bool(valid),ellipse_boundary_source=boundary)
 return p if valid else None,{},raw
