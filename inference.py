import os
import time
import argparse
from tqdm import tqdm

import numpy as np
import cv2
import torch

from data import transform,impro
from utils import util,ffmpeg

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", default=0, type=int,help="choose your device")
parser.add_argument("--model", default='./export/deep3d_v1.0.pt', type=str,help="input model path")
parser.add_argument("--video", default='./medias/wood.mp4', type=str,help="input video path")
parser.add_argument("--out", default='./results/wood.mp4', type=str,help="output video path")
parser.add_argument("--tmpdir", default='./tmp', type=str,help="output video path")
opt = parser.parse_args()


net = torch.jit.load(opt.model)
net.eval()
process = transform.PreProcess()

if 'cuda' in opt.model and torch.cuda.is_available():
    net.to(opt.gpu_id).half()
    process.to(opt.gpu_id).half()
else:
    opt.gpu_id = -1

out_width  = int(os.path.basename(opt.model).split('_')[2].split('x')[0])
out_height = int(os.path.basename(opt.model).split('_')[2].split('x')[1])

fps,duration,height,width = ffmpeg.get_video_infos(opt.video)
video_length = int(fps*duration)

util.clean_tempfiles(opt.tmpdir)
util.makedirs(os.path.split(opt.out)[0])
ffmpeg.video2voice(opt.video,os.path.join(opt.tmpdir, 'tmp.wav'))


#init
alpha = 5
cap = cv2.VideoCapture(opt.video)
frames_pool = []
output = np.zeros((out_height*1,out_width*2,3),np.uint8)
for i in range(alpha*2+1):
    ret, cur_frame = cap.read()
    if height != out_height or width != out_width:
        cur_frame = cv2.resize(cur_frame,(out_width,out_height),interpolation=cv2.INTER_LANCZOS4)
    frames_pool.append(torch.from_numpy(cur_frame))


x0 = frames_pool[0]
if opt.gpu_id >= 0:
    x0 = x0.to(opt.gpu_id).half()
x0 = process(x0)

print("start inference...")
for frame in tqdm(range(video_length)):
    if frame<alpha:
        beta = 0
    elif alpha<=frame<video_length-alpha:
        beta = -(frame-alpha)

    if alpha<frame<video_length-alpha:
        ret, cur_frame = cap.read()
        if height != out_height or width != out_width:
            cur_frame = cv2.resize(cur_frame,(out_width,out_height),interpolation=cv2.INTER_LANCZOS4)
        if not ret or cur_frame is None:
            break
        frames_pool.pop(0)
        frames_pool.append(torch.from_numpy(cur_frame))

    x1 = frames_pool[np.clip(frame-alpha+beta,0,alpha*2)]
    x2 = frames_pool[np.clip(frame-1+beta,0,alpha*2)]
    x3 = frames_pool[frame+beta]
    x4  = frames_pool[np.clip(frame+1+beta,0,alpha*2)]
    x5  = frames_pool[np.clip(frame+alpha+beta,0,alpha*2)]

    if opt.gpu_id >= 0:
        x1,x2,x3,x4,x5 = x1.to(opt.gpu_id).half(),x2.to(opt.gpu_id).half(),x3.to(opt.gpu_id).half(),x4.to(opt.gpu_id).half(),x5.to(opt.gpu_id).half()
    x1,x2,x3,x4,x5 = process(x1),process(x2),process(x3),process(x4),process(x5)
   
    input_data = torch.cat((x1,x2,x0,x3,x4,x5),dim=0)
    input_data = input_data.reshape(1,*input_data.shape)
    
    with torch.no_grad():
        out = net(input_data)
        x0 = out.clone().detach()[0]
    
    pred = torch.cat((x3,out[0]),dim=2)
    pred = transform.tensor2im(pred)
    impro.imwrite(os.path.join(opt.tmpdir, 'cvt','%06d'%(frame+1)+'.png'),pred,True)

print("start write to video...")
ffmpeg.image2video(fps,os.path.join(opt.tmpdir, 'cvt','%06d.png'),os.path.join(opt.tmpdir, 'tmp.wav'),opt.out)
cap.release()
util.clean_tempfiles(opt.tmpdir,tmp_init=False)