This assignment contains two parts. Each requires different dependencies. The conda environment for both assignments are stored in environment.yml file under each folder.

For robomimic, please follow the official robomimic installation guide. (https://robomimic.github.io/docs/introduction/installation.html) The pre-trained models I trained were stored inside "imitation assignment/robomimic/robomimic/core/bc/lift/ph/low_dim/trained_models" and recorded lift videos are stored inside as well. The repo was originally cloned from "https://github.com/ARISE-Initiative/robomimic"

For locomotion tasks. Please use conda to install environment.yml "conda env create -f environment.yml". the "motions" folder and txt files in the folder are motion captured dataset. 

"a1_il_mocap.py" mimics the data provided by unitree
"dog_simple_IL_a1" use a1 robot in pybullet to mimic dataset "dog_pace"
"dog_simple_IL_laikago" use a1 robot in pybullet to mimic dataset "dog_pace"
Sadly, they all failed by the time of the assignment deadline.

Other locomotion pretrained models are from the published github repo "https://github.com/erwincoumans/motion_imitation", same conda environment can be used to execute the pre-trained models there. Note that some apt dependencies may need installation.