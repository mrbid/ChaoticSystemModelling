# ChaoticSystemModelling
This is an experiment to see how well a Feed-forward Neural Network could model a chaotic system.

This version attempts to simulate just the collisions and the reflected vector for every sphere.

## how

This is developed to execute under a Linux operating system.

As always I tend to put important notes in the header of `main.c` ([main.c](main.c) & [/cli/main.c](/cli/main.c)).

If you would just like to see the result execute `launch_neuralsim.sh` you will need [Python3](https://www.python.org/downloads/), [TensorFlow](https://www.tensorflow.org/), and [XTERM](https://invisible-island.net/xterm/) installed. You may need to re-compile the binary first depending on the Linux distribution you are using by executing `release.sh`.

## results

This version attempts to output the reflected direction vector when a collision occurs.

The results are abysmal, the [EntireSimulation](../EntireSimulation) version had better results and even that is in the range of being abysmal.

The supplied models have been trained from a ~15GB dataset produced by executing `./cli/go.sh` which launched 64 instances of the cli dataset logging program. It took only a few seconds to generate said dataset.

## inputs

keyboard input for the `uc` program
- `N` = New simulation.
- `F` = FPS to console.
- `P` = Toggle CPU and NEURAL modes.
- `O` = Reset positions of spheres to outside the unit sphere.

