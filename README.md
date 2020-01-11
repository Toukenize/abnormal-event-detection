# Abnormal Event Detection
A project to detect and classify abnormal video clips (fighting and falling down) using various neural network architectures (Long-term Recurrent Convolutional Network, 3-Dimensional Convolutional Network).

Data were collected from multiple sources:

- The falling dataset consist of footage shot from multiple angles in a lab by [E. Auvinet et al from Université de Montréal](http://www.iro.umontreal.ca/~labimage/Dataset/), as well as other falling down footage found on various online sources. 
- The fighting dataset was a combination of Assault and Fighting videos from the UCF-Crime Dataset by [W. Sultani et al from the University of Central Florida](https://www.crcv.ucf.edu/projects/real-world/)

Each of the video clip is split into multiple subclips of 30 frames over 5 seconds, and the annotation of the 5-seconds clips was done manually.

## 2 types of Models

- To be updated

## 6 types of Features Representations

For each frame in the video, 6 different feature representations were generated:

| No.  | Name                               | Description                                                  | Sample                                                       |
|:----:|:----------------------------------:|:------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | Raw                                | 1-channel representation : Grayscale version the original image. | ![chute01_cam2_subclip007_frame008_raw](https://user-images.githubusercontent.com/43180977/72202329-15e21480-3499-11ea-8058-ee04a47a90b5.png) |
| 2    | Heatmap (HM)*                      | 1-channel representation : Grayscale version of OpenPose's raw output, with high intensity areas corresponding to high possibility of human body part. | ![chute01_cam2_subclip007_frame008_hm](https://user-images.githubusercontent.com/43180977/72202333-2db99880-3499-11ea-8cfd-eddc9643c6c5.png) |
| 3    | Keypoint (KP)*                     | 1-channel representation : Grayscale version of the skeletons representation of the detected human (processed from thresholded OpenPose heatmap). | ![chute01_cam2_subclip007_frame008_kp](https://user-images.githubusercontent.com/43180977/72202340-3a3df100-3499-11ea-980a-bd54341f1e3f.png) |
| 4    | Raw, Heatmap, Backsub (RHB)**      | 3-channels representation : Raw (1), Heatmap (2), Background-subtracted frame, positive regions (non-zero in RGB color space) represents region of new motions relative to the previous 5 frames. | ![chute01_cam2_subclip007_frame008_rhb](https://user-images.githubusercontent.com/43180977/72202345-49bd3a00-3499-11ea-9702-c650f2a3c75f.png) |
| 5    | Heatmap, Keypoint, Backsub (HKB)** | 3-channels representation : Heatmap (2), Keypoint (3), Background-subtracted frame. | ![chute01_cam2_subclip007_frame008_hkb](https://user-images.githubusercontent.com/43180977/72202351-593c8300-3499-11ea-9880-fc08a9efc640.png) |
| 6    | Heatmap, Heatmap, Backsub (HHB)**  | 3-channels representation : Heatmap (2), Heatmap (2), Background-subtracted frame. | ![chute01_cam2_subclip007_frame008_hhb](https://user-images.githubusercontent.com/43180977/72202358-60fc2780-3499-11ea-9606-45b59551c4d3.png) |

Note \* : Extracted using [OpenPose application](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

Note \*\* : Background subtraction with past 5 frames as history, using [OpenCV's Gaussian Mixture-based Background/Foreground Segmentation Algorithm](https://docs.opencv.org/3.4/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html#details)

## Modelling Enviroment:
- OS : Windows 10
- GPU : NVIDIA RTX2070
- RAM : 16 GB
- Package Manager : Anaconda 4.8.0

# How to Use
## Download these files:
1. [Processed Data with 6 different representations](https://www.kaggle.com/chewzy/assault-falldown-data)
2. [Annotations](https://www.kaggle.com/chewzy/assault-fight-data-train-test-files/)
3. [Extracted Features](https://www.kaggle.com/chewzy/assault-fall-extracted-features) (Optionally, you can run the `extract_features.py` to obtain the same set of features)
4. [Pretrained Weights for C3D](https://drive.google.com/file/d/1rlZ-xTkTMjgWKiQFUedRnHlDgQwx6yTm/view)
5. The scripts and yaml from this repository

## Organise them into the following structure:

![image](https://user-images.githubusercontent.com/43180977/72202540-a588c280-349b-11ea-9fc1-acef8a50737f.png)

## Recreate the conda environment

Using anaconda prompt, change directory to your `project_folder` from the previous step.

Create the environment (make necessary changes the **prefix** in the yml file to your Anaconda directory):

`conda env create -f tf_gpu_115.yml`

Activate the environment:

`conda activate tf_gpu_115`

## Feature Extractions

Extract features from the `assault-fall-data` folder.

Run `python extract_features.py`. New folders `c3d`, `mobilenet`, `resnet50v2` will be created, with all the extracted features

*Note : Only run this if you have not download the **Extracted Features** from previous section.*

## Model Training

Train the select model using selected features (image type), over a specified N-fold cross validation for M times.

Run the script `model_training.py` with these flags (available options in `{}`): 

```
  --model {mobilenet,resnet50v2,c3d,all}
                        Select model you want to build.
  --folds {1,2,3,4,5,6,7,8,9,10}
                        Specify N-fold validations.
  --runs {1,2,3,4,5,6,7,8,9,10}
                        Specify N runs. During each run, N-fold CV is done.
  --imgtype {raw,hm,kp,hhb,rhb,hkb,all}
                        Specify feature representation.
```

For example, if you want to train all models, with all image types using a 5-fold cross validation for 10 times, run:

`python model_training.py --model all --folds 5 --runs 10 --imgtype all`

A training results folder will be created for each model, with subfolders for each image type. The training metrics for each model (1 model is created for each fold during each run) is saved as an image in the respective subfolders, which looks like this:

![mobilenet_hhb_run1_fold1_metrics](https://user-images.githubusercontent.com/43180977/72201845-4030d380-3493-11ea-8cbb-c1e784e3f82a.png)

A prediction csv file is also generated, which consist of the test predictions of every model trained.

| is_fight | is_fall | raw_run1_fold1_fight | raw_run1_fold1_fall | raw_run1_fold2_fight | raw_run1_fold2_fall |
| -------- | ------- | -------------------- | ------------------- | -------------------- | ------------------- |
| 0        | 1       | 0.1323               | 0.6754              | 0.2351               | 0.1231              |
| 1        | 0       | 0.3245               | 0.1234              | 0.7231               | 0.3275              |
| ...      | ...     | ...                  | ...                 | ...                  | ...                 |

## Model Results Analysis

See the analysis notebook [here](https://github.com/Toukenize/abnormal-event-detection/blob/master/Model%20Results%20Analysis.ipynb).
