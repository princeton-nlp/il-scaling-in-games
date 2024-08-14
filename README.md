# Scaling Laws for Imitation Learning in Single-Agent Games

Official code repo for the paper "Scaling Laws for Imitation Learning in Single-Agent Games".

## Downloading NetHack Weights ⬇️

Please download the model weights for our forecasted NetHack experiment from the following links:
- [LSTM (30M) model weights](https://drive.google.com/file/d/1tWxA92qkat7Uee8SKMNsj-BV1K9ENExl/view?usp=share_link), [LSTM model flags](https://drive.google.com/file/d/1yyvbhq-yBF6q3lWCtfrIiyB1NdhTr5l2/view?usp=share_link)
- [Mamba (167M) model weights](https://drive.google.com/file/d/1LbgA-yTDOe3VDd3-iGHyJrh0FXgDKlJ3/view?usp=share_link)

You can use `gdown` to download these using the command line. Simply install `gdown` using `pip install gdown` and then run the following commands:
- LSTM model: `gdown 'https://drive.google.com/uc?id=1tWxA92qkat7Uee8SKMNsj-BV1K9ENExl'`
- LSTM flags: `gdown 'https://drive.google.com/uc?id=1yyvbhq-yBF6q3lWCtfrIiyB1NdhTr5l2'`
- Mamba model: `gdown 'https://drive.google.com/uc?id=1LbgA-yTDOe3VDd3-iGHyJrh0FXgDKlJ3'`

Make sure to place these files in a folder named `nethack_files` in the root directory of this repo.
> [!NOTE]
> The Mamba model weights originate from follow-up experiments and are not used in the paper. 
> However, they achieve strong performance (~9.5k on Human Monk) and are provided here in case they are helpful for the community.

## Installation 🔌
Clone the repository and run the following command in the root directory:
```
pip install -e .
```

Then run install from requirements.txt:
``` 
pip install -r requirements.txt
```

If you plan on using the Mamba model, make sure to also install the following packages:
```
pip install causal-conv1d==1.2.0
pip install mamba-ssm==1.1.1
pip install hydra-core --upgrade
```

## Running the sample script 🚀

### LSTM
Once everything is installed, you can simply run
```
python3 -m il_scale.nethack.v1.rollout --parameter_file conf/rollout_example.parameters
```
The hyperparameters in `conf/rollout_example.parameters` are set to reproduce the numbers in the paper. You can also change the hyperparameters to run your own experiments.

> [!NOTE]
> The `num_actors` flag is currently set to 1. If you want to run multiple actors in parallel to speed things up (recommended), you can simply increase this number.

### Mamba 🐍

To run the Mamba model, you can run the following command:
```
python3 -u -m il_scale.nethack.v2.rollout +nethack/rollouts=rollout_mamba
```

## Training your own models 🏋️‍♂️
To start training your own models, you can use the following command:
```
python3 -m il_scale.nethack.v2.train_nethack_il +nethack/exp=train_sample
```

> [!TIP]
> In order to successfully run the training script, you'll need to have a dataset of AutoAscend trajectories named `nld-aa-human-monk`. 

