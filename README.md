# Distributed-supervised-learning-of-dynamic-models 

## Installation
  
  - at the end of environment.yml prefix: /PATH/TO/anaconda3/envs/abgabe
  - conda env create -f environment.yml
  - git clone https://github.com/NicoBach/dsl_custom_gym.git
  - cd custom_gym
  - pip install -e

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




