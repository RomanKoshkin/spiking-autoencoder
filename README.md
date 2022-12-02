# LATEST
- metric: EMA of reward 
- metric: correlation of the paddle middle and the ball's y-position
- fixed reward logging and stimulation on reward
- `sh make_video.sh` converts data snapshots in tmp/ to .png files in assets/ (using multiprocessing), makes a video to videos/ and purges both the data snapshots in tmp/ and pictures in assets/.
- the fact that I don't have to make pictures at every data snapshot speeds up the simulation, the pictures are generated on multiple cores, which is way more efficient.
- `run.py` - runs the experiment. Don't forget to `cd notebooks`.

# Usage
`python run.py --config=configs/config_1.yaml`
`cd scripts && sh make_video.sh`

# TODO

- log the ball's position in the state_dict
- try changing the speed of the ball in network time (step between the snapshots)
- try a denser weight matrix
- try disinhibiting the network
- ? The matrix becomes more diagonal over time, which is good: place cells emerge.
- You don't need spurious assemblies. Turn down inhibition -- see what happens.
- to see if the ball forms tracks, make a move of in+outdegree deltas
- turn off rewards.

## Trainer
The network is "embodied" in an enviroment in which it can move the paddle up and down. If the ball bounces off the paddle, the network receives a positive _reward_ and if it misses the ball, it recieves a negative _reward_

## Exp1
See if activity of a group of briefly stimulated neurons propagates.

## Notes
- if you turn off the paddle, the network might not have enough stimulus to get going, i.e. the ball will not elicit any spikes in the place cells it flies over.

- Kick-start the network by turning on the paddle
- after the ball produces activity reliably, you can turn off the paddle.

## Exp2 - Set EE weights to some uniform value
- Still no particular direction of the vector field after a constant unidirectinal movement of the ball
- With uniform EE weights (set at 0.2), you need at least 3s with `paddle_on=True` to jump-start the network activity
- the weights do get biased in the direction opposite to the ball's movement

