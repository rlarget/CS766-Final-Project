---
title: "CS766 Final Project"
author: "Riley Larget"
---

# Motivation

There is race footage (https://www.youtube.com/@CMUbuggy) that we would like to analyze.
It has pans and zooms within shots and cuts between different camera angles. We intend to extract both
3-D (course environment) and 4-D (racer progress) data.
The same techniques that will work on this example are broadly applicable to geolocation and
object tracking more generally. There is significant amounts of labeled video footage available in this
case.

The course looks like this:
![Course Map](Resources/frames/i-Hv7VPf8.png)

Details in the [project proposal](Resources/Project_Proposal.pdf)

# Challenges

The main challenges were isolating the relevant parts of the videos and frames. Methods expected to work in theory did not work in practice.

Many attempts almost worked or worked on most frames but not all. The edge case failures were detected via manual review, which took significant time.
For a specfic example, automatic sub-picture boundary detection worked most of the time, but occasionally erroneously detected phantom small main pictures. In the masking pipeline, this creates sudden largely black frames in the middle of a sequence for no clear reason.

These preprocessing and masking tasks absorbed most of the effort, after including the false starts and dead ends.

## Motion detection example

![Original frame 1300](Resources/frames/frame1300.png)

Optical flow does not work well in this case.

![Optical Flow frame 1300](Resources/frames/frame1300_motion.png)

## Timer extraction example

![Original frame 1300](Resources/frames/frame1300.png)

Frame level functions extract the timer and digits (to make sure timer is running).

![Timer filter 1300](Resources/frames/timer_frame1300.png)

![Digit detector 1300](Resources/frames/timer_mask_frame1300.png)

Once implemented, this was effective at identifying the main race parts of the video (timer exists and is running).

## Overlay elements dominate feature detection

Without masking the overlay: ![Timer filter 1300](Resources/frames/Orb_frames.png)

Masking the overlay: ![Timer filter 1300](Resources/frames/Orb_masked_frames.png)

# Results

Please see [Final Presentation](Resources/CS_766_Final_Presentation_Slides.pdf)

The code is available [here](Code/).
