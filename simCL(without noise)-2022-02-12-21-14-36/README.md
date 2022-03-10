
# Concurrent Learning adaptive control      

Implimented Concurrent Learning adaptive controller that regulates the two-link system to the desired trajectory and proved satbility using Lyapunov-based method. 

The Two_link_sim.py file call both Concurrent Learning.py and Two_link_dynammics.py file. The control gaines are tuned to be alpha=2, beta=1 and gamma=0.5. Only data with eigenvalues greater that a lambdaCL=1 and different enough form the already stored data was used to impliment concurrent to satisfy the condition of persistance of excitation. 

Angle plots (angels.py, angularVelocity.py and angularAcceleration.py) show the plots of desired angle, angular velocity and angular acceleration of joint angles theta1 and theta2 vs the simulated results. The error graphs confirm the stability analysis of the controller as they are all converging to zero with time. The behaivor of parametric errors are show by the thetaHat.pdf, thetaTilda.pdf and thetaTildeNorm.pdf plots, which show that the measured value of theta trace the theta estimates.

The input.pdf graph shows the behavior of control over time along with its feedback and feedforward components. 

The data used to plot these graphs are stored in the data.csv file.




