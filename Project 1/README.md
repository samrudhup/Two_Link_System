
# Gradient Adaptive Control

Derived the dynamics of a two-link system using Euler-Legrange dynamics. Considering that the mass 'm' and length 'l' are unknown but have known bounds. The solved dynamics can be found in the twoLinkDynamics.pdf file.

Derived a gradient based adaptive controller that regulates the system to the desired trajectory. Proved that the designed controller is Lyapunov stable. 

Implimented this controller to show the behavior of the control using gradient update law. Angle plots (angels.py, angularVelocity.py and angularAcceleration.py) show the plots of desired angle, angular velocity and angular acceleration of joint angles theta1 and theta2 vs the simulated results. The error graphs confirm the stability analysis of the controller as they are all converging to zero with time. The behaivor of parametric errors are show by the thetaHat.pdf, thetaTilda.pdf and thetaTildeNorm.pdf plots, which show that the measured value of theta trace the theta estimates.

The input.pdf graph shows the behavior of control over time along with its feedback and feedforward components. 

The data used to plot these graphs are stored in the data.csv file.


## Documentation

[Dynamics](https://github.com/samrudhup/Two_Link_System/blob/main/Project%201/twoLinkDynamics.pdf)

