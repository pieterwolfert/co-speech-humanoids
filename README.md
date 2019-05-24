# Robots Learn Social Skills: End-to-End Learning of Co-Speech Gesture Generation for Humanoid Robots.
A PyTorch implementation of "Robots Learn Social Skills: End-to-End Learning of Co-Speech Gesture Generation for Humanoid Robots". 

    @article{yoon2018robots,
      title={Robots Learn Social Skills: End-to-End Learning of Co-Speech Gesture Generation for Humanoid Robots},
      author={Yoon, Youngwoo and Ko, Woo-Ri and Jang, Minsu and Lee, Jaeyeon and Kim, Jaehong and Lee, Geehyuk},
      journal={arXiv preprint arXiv:1810.12541},
      year={2018}
    }
[[Paper]](https://arxiv.org/pdf/1810.12541.pdf)

You can find the dataset and more information on the paper [here](https://sites.google.com/view/youngwoo-yoon/projects/co-speech-gesture-generation).

----
### Installation
The code is implemented in Python 3.7 and PyTorch 1.0

The implementation in this repository is based on the pre-print as it appeared on arXiv. I will update the implementation when more information becomes available. Please create an issue when you spot bugs. 

Update 23/5: Although training appears to provide fluent poses, I haven't been able yet to reproduce the same results as Youngwoo.

Check out [this repository](https://github.com/pieterwolfert/2d_to_3d_human_pose_converter) for translating 2D frontal pose to 3D frontal pose.

### Running the code

Please follow the following steps to run the code:

1- Create the following directories:

	./data/shots/JSON/
	./data/videos_tedx_subtitles/
	./data/shots/CSV/
	./models/

2- Place the youtube-gestures-dataset generated .csv, .json, and .vtt in their respective foldrs. Please take care to make sure that file names are formated exactly as <video_id>.ext to avoid file errors.

3- Run preprocessing.py to generate the pickle file "preprocessed_1295videos.pickle".

4- Move the file generated in step 3 to the ./data directory

5- If running for the first time, modify main.py so that the parameter pre_trained_file=None

6- Run main.py
