# TODO

- The matrix becomes more diagonal over time, which is good: place cells emerge.
- You don't need spurious assemblies. Turn down inhibition -- see what happens.
- to see if the ball forms tracks, make a move of in+outdegree deltas
- turn off rewards.

## Trainer
The network is "embodied" in an enviroment in which it can move the paddle up and down. If the ball bounces off the paddle, the network receives a positive _reward_ and if it misses the ball, it recieves a negative _reward_

## Exp1
See if activity of a group of briefly stimulated neurons propagates.