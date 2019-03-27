import pickle
import numpy as np
import pandas as pd
import os, sys
import json
from tqdm import tqdm
import webvtt
from datetime import datetime, timedelta
from sklearn.decomposition import PCA

__author__ = "Pieter Wolfert"

class DataProcessor():
    """
    Class for reading in pose data from the ted x dataset.
    Per frame there are 18 points (in COCO format).

    Outputs to a pickle file.
    """
    def __init__(self, json_dir, sub_dir, csv_dir, expsize):
        self.json_dir = json_dir
        self.sub_dir = sub_dir
        self.csv_dir = csv_dir
        self.expsize = expsize
        self.clips = []
        self.clip_captions = []

    def readPose(self):
        """
        Creates a list, with every item being a list:
        [file_name, clip_info, sequence of frames for that clip]
        """
        clip_sequences = []
        json_videos = os.listdir(self.json_dir)
        for x in tqdm(json_videos):
            #reading through every file, add it to our database
            with open(self.json_dir + x, "r") as f:
                tmp = json.load(f)
                for i in tmp:
                    if not i['clip_info'][2]:
                        continue
                    frames = []
                    for j in i['frames']:
                        if 'pose_keypoints' in j:
                            #we choose 24 since this covers the 8 joints
                            frames.append(j['pose_keypoints'][:24])
                        else:
                            frames.append([])
                    #here we add all clips to one big list containing filename,
                    #clip info and the frames.
                    clip_sequences.append([x, i['clip_info'], frames])
        self.clips = clip_sequences

    def readSubtitles(self):
        """
        Reads the captions for the videos.
        Saves a list with every item being a list:
        [file_name, captions]
        """
        clip_caption_sequences = []
        subtitle_files = [x for x in os.listdir(self.sub_dir) if 'vtt' in x]
        for x in tqdm(subtitle_files):
            captions = []
            for caption in webvtt.read(self.sub_dir + x):
                captions.append([caption.start, caption.text, caption.end])
            clip_caption_sequences.append([x, captions])
        self.clip_captions = clip_caption_sequences

    def readCSV(self):
        """
        Reads the CSV files.
        Saves a list with every item being a list:
        [file_name, frame_numbers, start_times]
        """
        csv_files = [x for x in os.listdir(self.csv_dir)]
        csv_data = []
        for x in tqdm(csv_files):
            tmp = pd.read_csv(self.csv_dir + x, skiprows=[0])
            frame_numbers = tmp['Frame Number (Start)'].tolist()
            start_times = tmp['Start Time (seconds)'].tolist()
            length = tmp['Length (seconds)'].tolist()
            csv_data.append([x, frame_numbers, start_times, length])
        self.csv_data = csv_data

    def generateSequences(self):
        """
        This method combines captions and poses per clip.
        """
        clip_text_poses = []
        for x in self.clips:
            filename = x[0]
            clip_info = x[1]
            frames = x[2] #contains pose info
            #clip_info[0] = start, clip_info[1] = end
            start_time, length = self.getTime(filename, clip_info[0])
            #the next for loops are per clip
            clip_subtitles = []
            for z in self.clip_captions:
                if filename[:-4] in z[0]:
                    for a in z[1]:
                        e = datetime.strptime(a[0], "%H:%M:%S.%f")
                        e_seconds = e.minute * 60 + e.second
                        if e_seconds > start_time and e_seconds < start_time + length:
                            clip_subtitles.append(a[1])
            clip_text_poses.append([clip_subtitles, frames])
        return clip_text_poses

    def getTime(self, filename, frame_number):
        """
        Returns the start seconds and length of a clip
        given filename and frame number
        """
        for x in self.csv_data:
            if filename[:-4] in x[0]:
                start_seconds = x[2][x[1].index(frame_number)]
                length = x[3][x[1].index(frame_number)]
                return start_seconds, length

    def getCaptions(self, filename):
        for x in self.clip_captions:
            if filename[:-4] in x[0]:
                return x[1]

    def normalizePose(self, pose_frames):
        """
        Normalize a pose so that the neck position is at the origin (0,0)
        x[x+0] = x coordinate, x[x+1] = y , [x+2] = reliability
        """
        #we're going to do this per frame
        temp = []
        for x in pose_frames:
            if len(x) != 0:
                del x[2::3]
                pose_list = x
                length_shoulders = pose_list[4] - pose_list[10]
                offset_x = pose_list[2]
                offset_y = pose_list[3]
                for i, item in enumerate(pose_list):
                    if i % 2 == 0:
                        pose_list[i] = (item - offset_x) / length_shoulders
                    else:
                        pose_list[i] = (item - offset_y) / length_shoulders
                if len(pose_list) is 16:
                    temp.append(pose_list)
        return temp

    def runPCA(self, pca_frames):
        print("Running PCA fitting")
        pca = PCA(n_components=10)
        pca.fit(pca_frames)
        return pca

    def generateTextPose(self):
        #this gets one caption and a sequence of poses
        clip_caption_poses = []
        pca_frames = []
        print("Generating Pickle File, clips length: {}".format(len(self.clips)))
        for x in tqdm(self.clips):
            filename = x[0]
            clip_info = x[1]
            frames = x[2]
            start_time, length = self.getTime(filename, clip_info[0])
            fps = len(frames)/length
            captions = self.getCaptions(filename)
            captions_time = []
            if captions is None:
                continue
            for c in captions:
                e_start = datetime.strptime(c[0], "%H:%M:%S.%f")
                e_end = datetime.strptime(c[2], "%H:%M:%S.%f")
                e_start_seconds = e_start.minute * 60 + e_start.second
                e_end_seconds = e_end.minute * 60 + e_end.second
                duration = e_end_seconds - e_start_seconds
                if e_start_seconds >= start_time and e_end_seconds < start_time + length:
                    frame_sec_start = e_start_seconds - start_time
                    frame_sec_end = e_end_seconds - start_time
                    ff = frames[int(frame_sec_start * fps):int(frame_sec_end * fps)]
                    pose_frames = self.normalizePose(ff)
                    text = c[1]
                    clip_caption_poses.append([text, pose_frames])
        with open('preprocessed_1295videos.pickle', 'wb') as fp:
            pickle.dump(clip_caption_poses, fp)

if __name__=="__main__":
    data = "./data/shots/JSON/"
    sub = "./data/videos_tedx_subtitles/"
    csv = "./data/shots/CSV/"
    t = DataProcessor(data, sub, csv, expsize=100)
    print("Reading CSV")
    t.readCSV()
    print("Reading Subtitles")
    t.readSubtitles()
    print("Reading pose information")
    t.readPose()
    t.generateTextPose()
