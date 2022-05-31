# ChaoticSystemModelling
This is an experiment to see how well a Feed-forward Neural Network can model a chaotic system.

- [EntireSimulation](EntireSimulation) attempts to predict every sphere position at each timestep of the simulation.
- [CollisionsSimulation](CollisionsSimulation) attempts to simulate just the collisions by predicting the reflected unit vector of a sphere when a collision occurs.

The two code bases are almost identical I could have used an `#ifdef` flag to switch between the two, but it felt right to seperate them both to avoid potential confusion as each needs seperate datasets and models. I had high hopes for the CollisionsSimulation but no dice it would seem. So far atleast.

[EntireSimulation](EntireSimulation) is probably the most interesting of the two, while it resembles nothing to the original simulation it does try; which produces some speculatively "artistic" results particularly the [EntireSimulation/models/ODD/selu_adam_32_16_16_shuf](EntireSimulation/models/ODD) dataset.

It would seem the problem in [EntireSimulation](EntireSimulation) is that the model only gets taught about the simulation of spheres while are are inside the unit sphere because in the deterministic training model spheres can never leave the unit sphere. However, in the imperfect and generalised neural model it is highly likely these spheres will leave the unit sphere and thus leave the computational range of the neural networks training, this is my proposition to explain why the models tend to reach a freezing point when all of the spheres seem to reach a maximum radius from the center point (slightly beyond the bounds of the unit sphere).
