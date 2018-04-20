# Report for project CarND-Controls-MPC

[//]: # (Image References)

[image1]: ./writeup_images/kinematic_eqs.png "Kinematic equations"
[image2]: ./writeup_images/cte_error.png "Cte error"
[image3]: ./writeup_images/orientation_error.png "Orientation error"


---

## Compilation

The code should compile just fine!

## Implementation

### The Model

In order to predict the system states, a kinematic vehicle model was used. The state of this model consists of the following:

* `state = [x, y, psi, v]`

where 

* `(x, y)` is the position
* `(psi)` is the heading
* `(v)` is the velocity

The state variables are calulated using the following set of equations

![alt text][image1]

where `a(t)` and `delta(t)` are the acturator variables. There are two error tems of interest. 

The cross track error `(cte)` given by

![alt text][image2]

The heading error (or orientation error) `(epsi)` given by

![alt text][image3]

### Timestep Length and Elapsed Duration (N & dt)

Choosing `N` and `dt` essentially boils down to a tradeoff between computational load and accuracy. From what I've seen, there is essentialy an "inverse" relationship between `N` and `dt` since:

* Increasing `N` causes `computational load AND accuracy` to INCREASE
* Increasing `dt` cause `computational load AND accuracy` to DECREASE

Some values I've tried:

* `N = 30` and `dt = 0.01` caused the car to run off the track due to the computational load (system couldn't keep up). 
* `N = 25` and `dt = 0.01` caused the car to run off the track due to the computational load (system couldn't keep up).
* I kept decreasing `N` in decrements of 5 until i got to `N = 10`. To begin with, it worked OK, but the car quickly drove of the road aswell. 
* I set `dt = 0.1` instead to speed the system up. My thinking was that a faster loop will lead to more stable driving since MPC sends the first prediction to the car (i.e the speed and steering is set more often, even if it is less accurate). 

With `N = 10` and `dt = 0.1` the preformance was acceptable. 


### Polynomial Fitting and MPC Preprocessing

A 3rd order polynomial was fitted to the waypoints provided by the simulator. The `polyfit` function provided by Udacity was used. Before the waypoints were fitted, they were transformed to the vehicle coordinate system so that the initial values of `(x, y)` and `(psi)` could be assumed to be zero. The coefficients from the fit is used to evaluate the polynomial at the desired location to compute the `cte`.   

### Model Predictive Control with Latency

The MPC works fairly well with a latency of 100ms. To handle the latency, I tried to change the weights for the cost parameters in `MPC.cpp`. My thinking was that penalizing the steering angle paramters would "punish" the system for overly agressive steering, but it still didn't result in a stable system. After looking for tips at the Udacity forums, I changed how i initialized the MPC state. I used the kinematic equations to compute the future values for `(x,y)` and `(psi)` with a delay `dT` and also updated `cte` and `epsi` based on the kinematic equations. This causes (as far as I can tell) the MPC to account for the latency.

## Simulation

As far as I can tell, this criteria is met. 
