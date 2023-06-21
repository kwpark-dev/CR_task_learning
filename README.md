# CR_task_learning
Simple task learning implementation for stacking blocks. The main task is compoased of three different sub-tasks. Simulation was done by MoveIt. (but only stacking positions were used as data to train the model for sub-task02) 

1. sub-task01: Learning picking order of the blocks
2. sub-task02: Learning stacking position of the blocks
3. sub-task03: Estimating trajectory of the blocks to move it from the initial position to stacking position.

## sub-task01
Count the frequency of the block's order.

## sub-task02
The stacking position of the first picking block would be randomly chosen. The neural network trained by stacking position would return two candidate positions (x,y,z --> x,y,z+l,x,y,z+h).

## sub-task03
Assume that robot perfectly detects each block and identifies the coordinate. Then 3 observation points are already given (there are 3 blocks initially distributed) and another two points are able to estimate by sub-task02. In total, there are 5 points where the robot's arm should pass through. Thus, trajectory can be inferred by gaussian process.

## Discussion
1. The neural network is overfitting. If the test point is located at out of intrinsic boundary, the stacking positions don't obey the stacking rule.
2. Trajectory offered by GP has no idea about hidden constraints of height. The objects are located on the table, z = 0.225. The height of trajectories, however, are not always higher than the table! Please check the images.
