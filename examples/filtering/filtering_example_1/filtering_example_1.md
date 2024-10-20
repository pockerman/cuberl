# EXample 1: Kalman filtering

In this example we will the ```KalmaFilter``` class in order to estimate the state
of simple mobile robot. The example is taken from <a href="https://github.com/AtsushiSakai/PythonRobotics">PythonRobotics</a>.
In particular, we will implement the example in <a href="https://atsushisakai.github.io/PythonRobotics/modules/localization/extended_kalman_filter_localization_files/extended_kalman_filter_localization.html#id3">Extended Kalman Filter Localization</a>.
However, we will be using Kalman filtering instead of the extended Kalman filter.

In the simulation below the state of the is described by four variables. 
The 2D position ```p=(x,y)```, the orientation ```phi! and the velocity ```v```.

We will assume that the robot has a speed and a gyro sensors. Hence at each time step the input will be 
a 2D vector describing the velocity and the angular velocity. In addition, the robot has a GNSS sensor providing 
information about the positionof the robot. 

## Driver code

```

       

```

Running the code above produces the following output:

```

```

The average per epoch loss is shown in the figure below

| ![average-per-epoch-loss](./average_loss.png) |
|:--:|
| **Figure: Average loss per epoch.**|

