# ChaoticSystemModelling
This is an experiment to see how well a Feed-forward Neural Network can model a chaotic system.

- [EntireSimulation](EntireSimulation) attempts to predict every sphere position at each timestep of the simulation.
- [CollisionsSimulation](CollisionsSimulation) attempts to simulate just the collisions by predicting the reflected unit vector of a sphere when a collision occurs.

The two code bases are almost identical I could have used an `#ifdef` flag to switch between the two, but it felt right to seperate them both to avoid potential confusion as each needs seperate datasets and models. I had high hopes for the CollisionsSimulation but no dice it would seem. So far atleast.

[EntireSimulation](EntireSimulation) is probably the most interesting of the two, while it resembles nothing to the original simulation it does try which produces some speculatively "artistic" results particularly the [EntireSimulation/models/ODD/selu_adam_32_16_16_shuf](EntireSimulation/models/ODD) dataset.
