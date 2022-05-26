# ChaoticSystemModelling
This is an experiment to see how well a Feed-forward Neural Network can model a chaotic system.

- [EntireSimulation](EntireSimulation) attempts to predict every sphere position at each timestep of the simulation.
- [CollisionsSimulation](CollisionsSimulation) attempts to simulate just the collisions by predicting the reflected unit vector of a sphere when a collision occurs.

The two code bases are almost identical I could have used an `#ifdef` flag to switch between the two, but it felt right to seperate them both to avoid potential confusion as each needs seperate datasets and models. I had high hopes for the CollisionsSimulation but no dice it would seem. So far atleast.
