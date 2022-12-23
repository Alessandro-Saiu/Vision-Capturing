# Vision-Capturing

## Description
The aim of the project is to generate text strings describing what happens in a video content so that timestamps can be created 

## Technical information
- The model is launched from the main and requests as input the path where the videos are contained 
- The model is a pretrained version of TimeSformer by Facebook AI, it analyzes 8 frames at a time and returns a descriptive string of what is happening in the frames
- All the videos are resized to 256x256 
- The output of the video_processing module is a generator for a list of numpy arrays
