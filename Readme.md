# CQL
Self-contained implementation of CQL and S4RL (CQL-Noise). Based on SAC. Copyright Matthias Weissenbacher.

# Requirements
Tested on [D4RL](https://github.com/rail-berkeley/d4rl) and [Mujoco](http://www.mujoco.org/) continuous control tasks in [OpenAI gym](https://gym.openai.com/). 


# Usage
To run the experiments run the following shell scripts, respectively.
```
./run_cql_mujoco.sh
./run_cql_antmaze.sh
```

Insert the number of aviable GPU's as:
```
num_devices=2 # Enter the number of available GPU's
