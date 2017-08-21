'''test moveipy'''
from moviepy.editor import *

clip = VideoFileClip('project_video.mp4').rotate(180)

clip.ipython_display(width=200)
