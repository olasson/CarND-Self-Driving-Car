# Report for project CarND-Controls-MPC

[//]: # (Image References)

[image1]: ./writeup_images/kinematic_eqs.png "Kinematic equations"


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

### Timestep Length and Elapsed Duration (N & dt)

### Polynomial Fitting and MPC Preprocessing

### Model Predictive Control with Latency

## Simulation

As far as I can tell, this criteria is met. 
