# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from c3d.utils.video_util import *


# def visualize_clip(clip, convert_bgr=False, save_gif=False, file_path=None):
#     num_frames = len(clip)
#     fig, ax = plt.subplots()
#     fig.set_tight_layout(True)

#     def update(i):
#         if convert_bgr:
#             frame = cv2.cvtColor(clip[i], cv2.COLOR_BGR2RGB)
#         else:
#             frame = clip[i]
#         plt.imshow(frame)
#         return plt

#     # FuncAnimation will call the 'update' function for each frame; here
#     # animating over 10 frames, with an interval of 20ms between frames.
#     anim = FuncAnimation(fig, update, frames=np.arange(0, num_frames), interval=1)
#     if save_gif:
#         anim.save(file_path, dpi=80, writer='imagemagick')
#     else:
#         # plt.show() will just loop the animation forever.
#         plt.show()


# def visualize_predictions(video_path, predictions, save_path):
#     frames = get_video_frames(video_path)
#     assert len(frames) == len(predictions)

#     fig, ax = plt.subplots(figsize=(5, 5))
#     fig.set_tight_layout(True)

#     fig_frame = plt.subplot(2, 1, 1)
#     fig_prediction = plt.subplot(2, 1, 2)
#     fig_prediction.set_xlim(0, len(frames))
#     fig_prediction.set_ylim(0, 1.15)

#     def update(i):
#         frame = frames[i]
#         x = range(0, i)
#         y = predictions[0:i]
#         fig_prediction.plot(x, y, '-')
#         fig_frame.imshow(frame)
#         return plt

#     # FuncAnimation will call the 'update' function for each frame; here
#     # animating over 10 frames, with an interval of 20ms between frames.

#     anim = FuncAnimation(fig, update, frames=np.arange(0, len(frames), 10), interval=1, repeat=False)

#     if save_path:
#         anim.save(save_path, dpi=200, writer='imagemagick')
#     else:
#         plt.show()

#     return

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from c3d.utils.video_util import *
from c3d.utils.video_util import get_video_frames
import numpy as np
from PIL import Image
import cv2,os

def visualize_clip(clip, convert_bgr=False, save_gif=False, file_path=None):
    num_frames = len(clip)
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    def update(i):
        if convert_bgr:
            frame = cv2.cvtColor(clip[i], cv2.COLOR_BGR2RGB)
        else:
            frame = clip[i]
        plt.imshow(frame)
        return plt

    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 20ms between frames.
    anim = FuncAnimation(fig, update, frames=np.arange(0, num_frames), interval=1)
    if save_gif:
        anim.save(file_path, dpi=80, writer='imagemagick')
    else:
        # plt.show() will just loop the animation forever.
        plt.show()

def image_to_video(img_array,directory,out_path,start,end):
    print("from: {},to: {}".format(start,end))
    height,width,layers=img_array[1].shape
    out_temp_path,out_path = out_path+"_temp.mp4",out_path+".mp4"
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video=cv2.VideoWriter(out_temp_path,0x7634706d,30,(width,height))
    
    cv2.imwrite(directory+f"/{start}-{end}.png",img_array[start])
    for j in range(start,end):
        img = cv2.cvtColor(img_array[j], cv2.COLOR_RGB2BGR)
        video.write(img)

    cv2.destroyAllWindows()
    video.release()
    print("done with temp_video writing")
    os.system(f"ffmpeg -i {out_temp_path} -vcodec libx264 {out_path}")
    print("done with orig_video writing")
    os.remove(out_temp_path)

def visualize_predictions(video_path, predictions, save_path):
    video_name = video_path.split("/")[-1].split(".")[0]
    frames = get_video_frames(video_path)
    # print("entered!")
    assert len(frames) == len(predictions)

    # fig, ax = plt.subplots(figsize=(5, 5))
    # fig.set_tight_layout(True)

    # fig_frame = plt.subplot(2, 1, 1)
    # fig_prediction = plt.subplot(2, 1, 2)
    # fig_prediction.set_xlim(0, len(frames))
    # fig_prediction.set_ylim(0, 1.15)

    exit_val,count = 0,0
    for i in range(10,len(frames),10):
        # frame = frames[i]
        isAnomaly = sum(predictions[i-10:i])>=0.2
        if(isAnomaly==True):
            count+=1
            if(count == 3):
                if((i-100) >= 0):
                    start = i-100
                elif((i-80) >= 0):
                    start = i-80
                elif((i-50) >= 0):
                    start = i-50
                else:
                    start = i-30
            exit_val=0
        elif(count>3):
            exit_val+=1
            if(exit_val==3):
                end = i
                count,exit_val=0,0
                directory = "media/output/"+video_name+"/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    os.makedirs(directory+"abnormal/")
                    os.makedirs(directory+"normal/")
                out_path = "media/output/"+video_name+f"/abnormal/{start}-{end}"
                image_to_video(frames,directory+"abnormal",out_path,start,end)
    
    return