# PredictiveMA-WGAN

Code for the papers "3D Skeleton-based Human Motion Prediction with Manifold-Aware GAN " and "Human Motion Prediction Using Manifold-Aware Wasserstein GAN"

![](https://user-images.githubusercontent.com/105372137/225350201-eab5952e-efa9-4055-94a2-1fc5ab58363c.png)


Predict the end of a motion sequence using its beginning

https://user-images.githubusercontent.com/105372137/225054588-88c44fed-9446-4560-8d0f-89d13466814a.mp4
 
 
 
**Dependencies :**

For training and prediction 
- python 3.7 
- tensorflow 2.2.0 
- numpy 1.19.0 
- scipy 1.4.1 

For quantitative results and visualisation 
- Matlab R2020a

**Instructions:**

**Preparation**

```
- Download the Human3.6M database at [https://arxiv.org/pdf/2207.01685.pdf]([[https://arxiv.org/pdf/2207.01685.pdf](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip)](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip)
- Download the CMU-MoCap data from [https://github.com/chaneyddtt/Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics/tree/master/data/cmu_mocap](https://github.com/chaneyddtt/Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics/tree/master/data/cmu_mocap)
- In PMAWGAN-<dataset name> run create_folders.sh
- Download CMU weight at """""""""""" and put them in PMAWGAN-CMU/Checkpoint_long and PMAWGAN-CMU/Checkpoint_short
- Download Human3.6M weight at """""""""""" and put them in PMAWGAN-Human3.6M/Checkpoint_long and PMAWGAN-Human3.6M/Checkpoint_short
- Put the Human3.6M data in a folder named 'Human3.6M' at the root of the project
- Put the CMU Mocap data in a folder named 'CMU_MoCaP' at the root of the project
- run Create_Database.m In the script you can change 'DB_choice' between 'CMU_short', 'CMU_long', 'HM36_short', 'HM36_long' to select the dataset type (long = long term prediction)
- In PMAWGAN-<dataset name>/Data_skeleton copy the content of the '3D' folder contained in the newly created folder ('CMU_short', 'CMU_long', 'HM36_short', 'HM36_long')
```
**Running the model**
```
- launch WGAN.ipynb with notebook (parameters are in the second cell)
   or
- use 'python Main.py' (parameters are in the file. default Training = False,Short_Term = False for testing on long term prrediction)
```

**Getting quantitative results** 
```
- after predicting motion in python 
- in the Matlab terminal 
- use Quantitative_results(false) to get results for long term prediction (need prediction with Short_Term=False) 
- use Quantitative_results(true) to get results for short term prediction (need prediction with Short_Term=True)
```

**Visualisation of results** 
```
- after predicting motion in python 
- in the Matlab terminal 
- use Visualisation(class,N_samples,Short_Term) to see the results in a Matlab figure and save the results in a video file 
  - class : char of the name of the action to visualize, choose one of the following : 'Directions','Discussion','Eating','Greeting','Phoning','Photo','Posing','Purchases','Sitting','SittingDown','Smoking','Waiting','WalkDog','Walking','WalkTogether' 
  - N_samples : an integer representing the number of sample to visualize 
  - Short_Term =false : visualize long term prediction. Short_Term =true : visualize short term prediction. 
  - example : Visualisation('Walking',8,false) 
- Visualisation presents on the left side the ground truth and on the right the predicted motion 
- First the 15 last frames of the historical sequence are shown in blue 
- Then 10 or 25 frames (short term or long term) of predicted motion are shown in red on the right side 
- A video file is then saved. Example : Walking_long.avi 
- We recommend to visualize long term prediction as short term visualisation is rather fast.
```


**Citations:**

```
@INPROCEEDINGS {9667071,
author = {B. Chopin and N. Otberdout and M. Daoudi and A. Bartolo},
booktitle = {2021 16th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2021)},
title = {Human Motion Prediction Using Manifold-Aware Wasserstein GAN},
year = {2021},
volume = {},
issn = {},
pages = {1-8},
```
```
@ARTICLE{9924242,
  author={Chopin, Baptiste and Otberdout, Naima and Daoudi, Mohamed and Bartolo, Angela},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science}, 
  title={3D Skeleton-based Human Motion Prediction with Manifold-Aware GAN}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TBIOM.2022.3215067}}
```

**Acknowledgements**

this project has recieved financial support from the CNRS  through the 80-prime  program.
