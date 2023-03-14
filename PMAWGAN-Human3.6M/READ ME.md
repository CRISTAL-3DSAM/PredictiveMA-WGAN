Dependencies : 
For training and prediction
	- python 3.7
	- tensorflow 2.2.0
	- numpy 1.19.0
	- scipy 1.4.1
For quantitative results and visualisation
	- Matlab R2020a
	

predicting motion
	- launch in jupyter
	- parameters are in the second cell

	
Getting quantitative results
	- after predicting motion in python
	- in the Matlab terminal
	- use Quantitative_results(false) to get results for long term prediction (need prediction with Short_Term=False)
	- use Quantitative_results(true) to get results for short term prediction (need prediction with Short_Term=True)

Visualisation of results
	- after predicting motion in python
	- in the Matlab terminal
	- use Visualisation(class,N_samples,Short_Term) to see the results in a Matlab figure and save the results in a video file
		- class : char of the name of the action to visualize, choose one of the following :
			'Directions','Discussion','Eating','Greeting','Phoning','Photo','Posing','Purchases','Sitting','SittingDown','Smoking','Waiting','WalkDog','Walking','WalkTogether'
		- N_samples : an integer representing the number of sample to visualize
		- Short_Term =false : visualize long term prediction. Short_Term =true : visualize short term prediction. 
		- example : Visualisation('Walking',8,false)
	- Visualisation presents on the left side the ground truth and on the right the predicted motion
	- First the 15 last frames of the historical sequence are shown in blue
	- Then 10 or 25 frames (short term or long term) of predicted motion are shown in red on the right side
	- A video file is then saved. Example : Walking_long.avi
	- We recommend to visualize long term prediction as short term visualisation is rather fast.