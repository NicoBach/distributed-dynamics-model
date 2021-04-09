# Distributed-supervised-learning-of-dynamic-models 

This is the codebase for  a  distributed  supervised  learning  architecture, which  can  generate  trajectory  data  conditioned  by  control  commands and  learned  from  demonstrations.  The  architecture  consists  of  an  ensemble of neural networks (NNs) which learns the dynamic model and a separate addressing NN that decides from which NN to draw a prediction. This work also introduces an error-based method for automatic assignment of data subsets to the ensemble NNs for training using the loss profile of the ensemble.

![training](https://github.com/NicoBach/distributed-dynamics-model/blob/master/pictures/image-train.png)


 Each subnetwork will be trained only on data points from its bin. An equal number of samples in bins ensures good competitiveness of  subnetwork  specialization.  Not  providing  equally  sized  bins  for  the  ensemble  networks  will  preclude  specialization  if  at  one  point  an  ensemble  network becomes strictly better than all the others and then wins all training data for itself.
 
![2d-lossspace-before-training](https://github.com/NicoBach/distributed-dynamics-model/blob/master/pictures/image-ls1.png)

In the pictures above the loss profiles for a 2-NNs ensemble in the Acrobot environment are shown. On the left: loss  profiles  of  data  points  of  untrained  subnetworks  in  the  ensemble.  We apply clustering of these data points into two bins (green and black). On the right, the alignment of the data points along the axes (loss profile) is reflecting the successful specialization of the subnetworks after training of the ensemble ofthe two subnetworks (green and black).

## Installation
  
  - at the end of environment.yml prefix: /PATH/TO/anaconda3/envs/abgabe
  - conda env create -f environment.yml
  - git clone https://github.com/NicoBach/dsl_custom_gym.git
  - cd dsl_custom_gym
  - pip install -e .

## How2run
  
  - conda activate dist_supervised_learning
  - For running e.g. mountaincar-script
       cd adn_mountaincar
       python run_mountaincar.py
       run_mountaincar.py has a predetermined trajectorie,
       which can be turned off in the file
       use keys LEFT, RIGHT, DOWN to control agent
   - Alternatively (for Mac, pynput could not be working):
       python run_mountaincar_alternative_keyboard_control.py
       use keys 0, 1, 2 to control agent

  - For running e.g. acrobot-script
       cd adn_acrobot
       python run_acrobot.py
       use keys LEFT, RIGHT, DOWN to control agent
   - Alternatively (for Mac, pynput could not be working):
       python run_acrobot_alternative_keyboard_control.py
       use keys 0, 1, 2 to control agent




