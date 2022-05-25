# ChaoticSystemModelling
This is an experiment to see how well a Feed-forward Neural Network could model a chaotic system.

- [EntireSimulation](EntireSimulation) attempts to simulate every position of a sphere at each timestep of the simulation.
- [CollisionsSimulation](CollisionsSimulation) attempts to simulate just the collisions and the reflected vector for every sphere.

The two code bases are almost identical I could have used an `#ifdef` flag to switch between the two, but it felt right to seperate them both. I had high hopes for the CollisionsSimulation but no dice it would seem. So far atleast.
