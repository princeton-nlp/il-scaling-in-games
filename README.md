# Scaling Laws for Imitation Learning in NetHack

Official code repo for the paper "Scaling Laws for Imitation Learning in NetHack". While we unfortunately cannot release the full code, we do release the model weights of our forecasted model below along with a sample script to run it.

## NetHack Forecasted Model Weights
Please download the model weights for our forecasted NetHack experiment from the following links:
- Model weights: https://drive.google.com/file/d/1tWxA92qkat7Uee8SKMNsj-BV1K9ENExl/view?usp=share_link
- Model flags: https://drive.google.com/file/d/1yyvbhq-yBF6q3lWCtfrIiyB1NdhTr5l2/view?usp=share_link

Make sure to place these files in a folder named `nethack_files` in the root directory of this repo.

NOTE: You can use `gdown` to download these using the command line. Simply install `gdown` using `pip install gdown` and then run the following commands:
- `gdown 'https://drive.google.com/uc?id=1tWxA92qkat7Uee8SKMNsj-BV1K9ENExl'` for the model weights
- `gdown 'https://drive.google.com/uc?id=1yyvbhq-yBF6q3lWCtfrIiyB1NdhTr5l2'` for the model flags

## Installation
Clone the repository and run the following command in the root directory:
```
pip install -e .
```

Then run install from requirements.txt:
``` 
pip install -r requirements.txt
```

## Running the sample script
Once everything is installed, you can simply run
```
python3 -m il_scale.nethack.rollout --parameter_file conf/rollout_example.parameters
```
The hyperparameters in `conf/rollout_example.parameters` are set to reproduce the numbers in the paper. You can also change the hyperparameters to run your own experiments.

NOTE: the `num_actors` flag is currently set to 1. If you want to run multiple actors in parallel to speed things up (recommended), you can simply increase this number.


