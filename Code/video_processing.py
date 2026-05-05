import os
import numpy as np
np.float = np.float64
np.int = np.int_

import skvideo.io
import skvideo.motion
from skimage import io, morphology
import re

import matplotlib.pyplot as plt

from frame_processing import *

base_path = "../Videos/"

#video = [file for file in os.listdir(base_path) if re.match(r".*Heat.*Men.*mp4",file)][4]
#title_pattern = r"(\d+)\W.*-\W*(.+)\W+\((..+\W+[ABCDE]\W*)\W*\).*"
#format_pattern = r"\g<1>_\g<2>_\g<3>"
#print(video,re.sub(title_pattern,format_pattern,video))

#frame_path = "../KeyFrames/"
#video_folder_path = os.path.join(frame_path,re.sub(title_pattern,format_pattern,video))
#os.makedirs(video_folder_path,exist_ok=True)

# 30 frames per second
#video_reader = skvideo.io.vreader(os.path.join(base_path,video))

def make_image(array):
    image_range = 2*np.max([np.abs(array.max()),np.abs(array.min())])
    new_array = array/image_range+0.5
    return (new_array * 255)

def non_zero_count(frame):
    return (frame>0).sum()

def no_filter(metrics_data):
    return True

def timer_filter(metrics_data):
    metrics = metrics_data["metrics"]
    if len(metrics_data["filters"])==0:
        last_filter = False
    else:
        last_filter = metrics_data["filters"][-1]
    if last_filter == False and (metrics[-1] > 200) and (len(metrics) < 3000):
        return True
    else:
        return last_filter and (metrics[-1] > 200)
    
def transform_video(video_dir,
                    video_file,
                    transformed_dir,
                    transformed_pattern=r"\g<1>_\g<2>_\g<3>_transform.mp4",
                    transform_frame_function=identity,
                    transformed_metric_function=non_zero_count,
                    metric_filter=no_filter):
    title_pattern = r"(\d+)\W.*-\W*(.+)\W+\((..+\W+[ABCDE]\W*)\W*\).*"
    transformed_file_name = re.sub(title_pattern,transformed_pattern,video_file)
    os.makedirs(transformed_dir,exist_ok=True)
    video_reader = skvideo.io.vreader(os.path.join(video_dir,video_file))
    video_writer = skvideo.io.FFmpegWriter(os.path.join(transformed_dir,transformed_file_name))
    print(video_file,transformed_file_name)
    prev_frame = None
    metrics = []
    filters = []
    for i,frame in enumerate(video_reader):
        if i == 0:
            prev_frame = frame
        new_frame = transform_frame_function({"frame":frame,"prev_frame":prev_frame})
        metrics.append(transformed_metric_function({"frame":frame}))
        filters.append(metric_filter({"metrics":metrics,"filters":filters}))
        if filters[-1]==True:
            video_writer.writeFrame(new_frame)
        prev_frame = frame
    video_writer.close()
    metrics = np.array(metrics)
    return metrics, filters

def process_video_motion(video_reader,video_folder_path):
    os.makedirs(video_folder_path,exist_ok=True)
    overlay_sizes = []
    for i, frame in enumerate(video_reader):
        if i == 0:
            previous_frame = frame
        if i % 10 == 0:
            frames = np.stack([previous_frame,frame])
            mbSize = 8
            motion = skvideo.motion.blockMotion(frames,mbSize=mbSize)
            motion_frame = np.repeat(motion,mbSize,axis=1)
            motion_frame = np.repeat(motion_frame,mbSize,axis=2)[0]
            motion_frame = make_image(motion_frame)
            motion_frame = np.concat([motion_frame,np.zeros((*motion_frame.shape[:2],1))],axis=2).astype(frame.dtype)
            io.imsave(os.path.join(video_folder_path,f"frame{i}_motion.png"),motion_frame)
            previous_frame = frame

def process_video(video_reader,video_folder_path):
    os.makedirs(video_folder_path,exist_ok=True)
    overlay_sizes = []
    for i, frame in enumerate(video_reader):
        #overlay_sizes.append(overlay_mask.mean())
        if i == 1200:
        #if i % 10 == 0:
            overlay_mask = find_overlay(frame)
            inverse_mask = (1-overlay_mask)[:,:,np.newaxis]
            print(i,overlay_mask.mean())
            io.imsave(os.path.join(video_folder_path,f"frame{i}.png"),frame)
            io.imsave(os.path.join(video_folder_path,f"frame{i}_mask.png"),(frame*inverse_mask).astype(frame.dtype))

def bulk_videos(video_dir,
                    video_filter_pattern=r".*Heat.*Men.*mp4",
                    transformed_dir="../Trimmed_Videos/",
                    transformed_pattern=r"\g<1>_\g<2>_\g<3>_trimmed.mp4",
                    transform_frame_function=race_mask,
                    transformed_metric_function=timer_metric,
                    metric_filter=timer_filter):
    for video in os.listdir(video_dir):
        if re.match(video_filter_pattern,video):
            print(video)
            metrics, filters = transform_video(video_dir,
                                video,
                                transformed_dir,
                                transformed_pattern=transformed_pattern,
                                transform_frame_function=transform_frame_function,
                                transformed_metric_function=transformed_metric_function,
                                metric_filter=metric_filter)

#process_video(video_reader,video_folder_path)
#process_video_motion(video_reader,video_folder_path)

#metrics, filters = transform_video(base_path,
#                        video,
#                        "../Transformed_Videos/",
#                        transformed_pattern=r"\g<1>_\g<2>_\g<3>_trimmed.mp4",
#                        transform_frame_function=race_mask,
#                        transformed_metric_function=timer_metric,
#                        metric_filter=timer_filter)

bulk_videos(base_path)

#print(sum(filters))

#plt.plot(metrics)
#plt.show()
#plt.plot(filters)
#plt.show()