import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import pandas as pd
import datetime

SAVE_PATH = "C:/Users/RNALAB/Downloads/ImageClassification/media/"

def frame_creation(video_saved):
    #link = "https://youtu.be/TOdeshrNmF4"
    #list_of_links_df = pd.read_excel(datafile)
    #list_of_links = list_of_links_df['Links'].tolist()
    #link = list_of_links[0]
    #yt = YouTube(link)
    #stream = yt.streams.first()
    #video_saved = stream.download()
    #print(video_saved,"="*50)
    vidcap1 = cv2.VideoCapture(video_saved)

    success, image = vidcap1.read()
    count = 0
    try:
        while success:
            cv2.imwrite("C:/Users/RNALAB/Documents/ImageClassification_test/media/frame{0}.jpg".format(count), image)
            success, image = vidcap1.read()
            #print('Read a new frame: ', success)
            count += 1
        return True
    except UnboundLocalError:
        print('no frame created')
        return False
