import numpy as np
import os
import re
from skimage import color, io, util, filters, morphology
from skimage.feature import canny, ORB, match_descriptors, plot_matched_features

from matplotlib import pyplot as plt

from collections import Counter

def edge_detector(data,sigma=1.5):
    gray = color.rgb2gray(data["frame"])
    return util.img_as_ubyte(canny(gray,sigma=sigma))[:,:,np.newaxis].repeat(3,axis=-1)

def mask_grey(data,threshold=0.3):
    frame = data["frame"]
    mask = 1-frame.min(axis=-1,keepdims=True)/frame.max(axis=-1,keepdims=True)
    mask = mask > threshold
    bright_mask =  (frame.mean(axis=-1,keepdims=True)/frame.max()>0.5)
    
    mask = (mask + bright_mask)>0

    dilation_size = morphology.footprint_rectangle((5,5,1))
    mask = morphology.dilation(mask,dilation_size)

    #erosion_size = morphology.footprint_rectangle((15,15,1))
    #mask = morphology.erosion(mask,erosion_size)
    
    mask = (1-mask).repeat(3,axis=-1)

    #return (mask * 255.0).astype(frame.dtype)
    return (mask * frame).astype(frame.dtype)
    #return edge_detector({"frame":(mask * frame).astype(frame.dtype)})

def find_constants(data):
    return data["frame"] == data["prev_frame"]

def apply_constant_mask(data):
    mask = find_constants(data)
    return data["frame"]*(1-mask)

def identity(data):
    return data["frame"]

def find_overlay(frame):
    has_blue = frame[:,:,2] > 1.6*frame.mean(axis=2)
    is_dark = frame.mean(axis=2) < 0.3*255
    mask = has_blue*is_dark
    erosion_size = morphology.footprint_rectangle((3,3))
    dilation_size = morphology.footprint_rectangle((18,18))
    mask = morphology.erosion(mask,erosion_size)
    mask = morphology.dilation(mask,dilation_size)
    #mask = morphology.erosion(mask,morph_size)
    return mask

def apply_overlay_mask(data):
    frame = data["frame"]
    mask = find_overlay(frame)
    inverse_mask = (1-mask)[:,:,np.newaxis]
    return (frame*inverse_mask).astype(frame.dtype)

def find_timer(data):
    frame = data["frame"]
    mask = np.zeros(frame.shape)
    mask[10:35,390:520,:] = 1
    return (mask*frame).astype(frame.dtype)

def timer_metric(data):
    frame = find_timer(data)
    return (frame>frame[frame>0].mean()).min(axis=-1).sum()

def race_mask(data):
    frame = data["frame"]
    mask = np.zeros(frame.shape)
    subpicture_test = (frame.mean(axis=-1,keepdims=True)<=15) # find dark lines
    if subpicture_test.sum() < 700:
        mask[40:,:,:] = 1 # if not subpicture
    else:
        # coords = [Counter(2*(dim//2)).most_common(2) for dim in subpicture_test.nonzero()]
        # mask[min(coords[0])[0]:max(coords[0])[0],:max(450,min(coords[1])[0]),:] = 1 # if subpicture
        mask[52:306,:450,:] = 1 # if subpicture, heuristic
    return (mask*frame).astype(frame.dtype)

def orb_points(data):
    frame = data["frame"]
    prev_frame = data["prev_frame"]
    
    # transforms
    frame = race_mask({"frame":frame})
    prev_frame = race_mask({"frame":prev_frame})

    orb_finder = ORB()
    orb_finder.detect_and_extract(color.rgb2gray(frame))
    keypoints_now = orb_finder.keypoints
    descriptors_now = orb_finder.descriptors

    orb_finder.detect_and_extract(color.rgb2gray(prev_frame))
    keypoints_prev = orb_finder.keypoints
    descriptors_prev = orb_finder.descriptors

    matches12 = match_descriptors(descriptors_now,descriptors_prev,cross_check=True)

    fig,ax = plt.subplots()

    plot_matched_features(prev_frame,
                            frame,
                            keypoints0=keypoints_prev,
                            keypoints1=keypoints_now,
                            matches=matches12,
                            ax=ax)

    plt.show()

    print(orb_finder.keypoints.astype(int).shape)
    #print(frame[orb_finder.keypoints.astype(int)].shape)
    return frame

def test_image(frame_name="frame3700.png",
                 prev_frame_name="frame3650.png",
                 transform_pattern=r"transformed_\g<1>.png",
                 transform_func=identity):
    frame_dir = "../KeyFrames/" 
    first_frame_vid = os.listdir(frame_dir)[0]
    frame_path = os.path.join(frame_dir,first_frame_vid)
    frame = io.imread(os.path.join(frame_path,frame_name))
    prev_frame = io.imread(os.path.join(frame_path,prev_frame_name))
    new_frame = transform_func({"frame":frame,"prev_frame":prev_frame})
    io.imsave(re.sub(r"(.*).png",transform_pattern,frame_name),new_frame)
    #io.imsave(re.sub(r"(.*).png",r"timer_mask_\g<1>.png",frame_name),timer_metric(new_frame))
    #print(timer_metric(new_frame).sum())
    return new_frame

test_image(transform_pattern=r"orb_points_\g<1>.png",transform_func=orb_points)