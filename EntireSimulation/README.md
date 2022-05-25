# ChaoticSystemModelling
This is an experiment to see how well a Feed-forward Neural Network could model a chaotic system.

This version attempts to predict every sphere position at each timestep of the simulation.

## how

This is developed to execute under a Linux operating system.

As always I tend to put important notes in the header of `main.c` ([main.c](main.c) & [cli/main.c](cli/main.c)).

If you would just like to see the result execute `launch_neuralsim.sh` you will need [Python3](https://www.python.org/downloads/), [TensorFlow](https://www.tensorflow.org/), and [XTERM](https://invisible-island.net/xterm/) installed. You may need to re-compile the binary first depending on the Linux distribution you are using by executing `release.sh`.

## results

This version attempts to output the next position of the sphere for the current step of the simulation.

I trained a few models as you can see in the [/models](models) directory but the neural network did not learn a very good representation of the chaotic system. Also it's scalability is very limited as not many home computers have the resources to train networks of more than 16 bouncing spheres. I would have probably had better results with a recurrent neural network but I wanted to test to see how well a traditional FNN would perform using large input and output buffers. _(I have since experimented with an LSTM [here](train_lstm.py))_

I did try to teach the network to output only direction vectors but the network had a harder time minimising the loss in these instances. The network trained better using sphere position as the target output and then from the last position and the new position from the neural network I calculate a new direction vector. This works much better. But as you can see, it's not close to replicating the original simulation at all.

The network does learn some concept of the outer boundaries of the unit sphere but it seems more of a cubic representation than a spherical one.

The supplied models have been trained from a ~15GB dataset produced by executing `./cli/go.sh` which launched 64 instances of the cli dataset logging program. It took only a few seconds to generate said dataset.

## inputs

keyboard input for the `uc` program
- `N` = New simulation.
- `F` = FPS to console.
- `P` = Toggle CPU and NEURAL modes.
- `O` = Reset positions of spheres to outside the unit sphere.

