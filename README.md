# Lyft Motion Prediction for Autonomous Vehicles
## Summary

This repository contains code for the [Lyft Motion Prediction](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles) Kaggle competition.

The goal of this competition was to predict the motion of surrounding traffic agents (cars, pedestrians, cyclists) for the next 5 sec. 

 <img src="https://user-images.githubusercontent.com/68122114/199073197-8943fc83-cf69-474a-aba9-4f0108dfcff0.jpg" width="468" height="230"> 
 <img width="390" alt="Screenshot 2022-10-31 105222" src="https://user-images.githubusercontent.com/68122114/199075824-c70c3972-2e07-42f3-b9cb-d589da22ea33.png">
 
For each agent, three trajectories and their respective confidence scores were predicted. 
The predictions were evaluated with negative log-likelihood loss. 

This ResNet based model achieves negative log-likelihood loss **18.915**.
 
## Sources 
* https://github.com/woven-planet/l5kit
