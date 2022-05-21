# ChaoticSystemModelling
This was an experiment to see how well a Feed-forward Neural Network could model a chaotic system.

## how

This is developed to execute under a Linux operating system.

If you would just like to see the result execute `launch_neuralsim.sh` you will need [python3](https://www.python.org/downloads/) and [tensorflow](https://www.tensorflow.org/) installed. You may need to re-compile the binary first depending on the Linux distribution you are using by executing `release.sh`.

## results

I trained a few models as you can see in the [/models](models) directory but the neural network did not learn a very good representation of the chaotic system. Also it's scalability is very limited as not many home computers have the resources to train networks of more than 16 bouncing spheres. I would have probably had better results with a recurrent neural network but I wanted to test to see how well a traditional FNN would perform using large input and output buffers.

I did try to teach the network to output only direction vectors but the networks had a harder time minimising the loss in these instances, I assume this is because when a collision occurs the direction vector updates very frequently as two spheres lock together maintaining intersection briefly in some instances. The network trained better using sphere position as the target output and then from the last position and the new position from the neural network I calculate a new direction vector. This works much better. But as you can see, it's not close to replicating the original simulation at all.

The network does learn some concept of the outer boundaries of the unit sphere but it seems more of a cubic representation then a spherical one.

The supplied models have been trained from a ~15GB dataset produced by executing `./cli/go.sh` which launched 64 instances of the cli dataset logging program. It took only a few seconds to generate said dataset.

*"It's not that interesting. I tried it and here is the outcome."*

## inputs

keyboard input for the `uc` program
- `N` = New simulation.
- `F` = FPS to console.
- `P` = Toggle CPU and NEURAL modes.

