# ChaoticSystemModelling
This was an experiment to see how well a Feed-forward Neural Network could model a chaotic system.

## how

If you would just like to see the result execute `launch_neuralsim.sh` you will need python3 and tensorflow installed. You may need to re-compile the binary first depending on the Linux distribution you are using by executing `release.sh`.

## results

I trained a few models as you can see in the [/models](models) directory but the neural network did not learn a very good representation of the chaotic system. Also it's scalability is very limited as not many home computers have the resources to train networks of more than 16 bouncing spheres.

I did try to teach the network to output only direction vectors but the networks had a harder time minimising the loss in these instances, I assume this is because when a collision occurs the director vector updates very frequently as the two spheres maintain intersection in some instances. The network trained better using position as outputs and then from the last position and the new position from the neural network I calculate a new direction vector. This works better. But as you can see, it's not close to replicating the original simulation at all.

The network does learn some concept of the outer boundaries of the unit sphere but it seems more of a cubic representation then a spherical one.

## inputs

input for the `uc` program
- `N` = New simulation.
- `F` = FPS to console.
- `P` = Toggle CPU and NEURAL modes.

