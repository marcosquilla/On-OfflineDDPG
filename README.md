# On-OfflineDDPG
Offline algorithm trained on the data generated by the same algorithm but online

This algorithms require to have the [pytorch_sac_ae repository](https://github.com/denisyarats/pytorch_sac_ae)  (for making the video and replay buffer), but most importantly [dmc2gym](https://github.com/denisyarats/dmc2gym) to create and operate the environment. We also took in some parts inspiration from [MinimalRL](https://github.com/seungeunrho/minimalRL).

## Online
First, the online algorithm must be run. It will generate data in the specified directory, as well as save the parameters of the policy and a video for the last evaluation of the policy. 
In this step, it can be chosen to include random actions (ie. actions that are not calculated by the policy). Currently it is set to include them gradually (no actions at the beginning of training and increase the amount linearly until 50%).

## Offline
Then, the data generated will be used by the offline version of the algorithm. Based on my experience, the learning rates should be 10 times smaller for it to work. 
Here it can be chosen to balance the dataset. Should this be the case, classes will be created based on the rewards (currently classes are up to the first decimal) and then an equal amount of records will be taken randomly from each class. The amount of decimals can be changed, but pay attention to df2 as it determines how many records to take (it doesn't automatically take the maximum amount).
This algorithm also saves the the policy parameters and a video for the best performing evaluation.

## Comparing results
Worth to mention when plotting the rewards against the episodes, is that in the online version there are 20 updates per episode, while in the offline there is a single one.
