# Report for project: CarND-Controls-PID

## Compilation

The code should compile just fine!

## Implementation

The basic PID equation is implemented in `PID::CaculateControllerOutput()` which is a function I added. `PID::TotalError()` did not really fit into how I implemented the PID. 

## Reflection

### Proportional part of controller (P)

The purpose of the P-term is essentially to reach a desired value as fast as possible. If this value is set to high, one can experience overshoot or even severe oscillations around the target value. 

This term did exactly what i expected. If I increased it, the car would preform more "agressive" corrections in steering angle, but if I set it too high, the car would wobble or even drive of the track. 

### Integral part of controller (I)

A P-controller can never achieve zero error on its own, making the I-term useful. This term helps drive the system error as close to zero as possible by summing up the errors over the current and all previous samples. This is all well and good in theory, but it can also result in the system drifiting away from the desired value, since the sum of errors will keep growing over time. 

I took two measures to combat this. First, I kept the gains on the I-terms low. Second, I implemented a very simple "anti windup" mechanism, which effectively saturates the I-term, keeping it from growing indefinitely.



### Derivative part of controller (D)

The main purpose of this term is to "predict" errors. Since it is a function of the current and past errors, it can provide a sharp correction when necessary. It is very useful when the P-term is unable to stabilize the system during large error margins. 

One very important considerations of the D-term, is how it behaves with very noisy signals. If the amplitude of the noise is too large, the D-term can end up amplifying it and make the system unstable. Since this project runs on a simulator, the D-term is fairly well behaved. However,in the real world, implementing some kind of filtering or tracking mechanism before using this term would be wise. 



## Simulation
