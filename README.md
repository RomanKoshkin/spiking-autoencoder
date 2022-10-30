# TODO

- The matrix becomes more diagonal over time, which is good: place cells emerge.
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
