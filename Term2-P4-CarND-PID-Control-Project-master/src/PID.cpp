#include "PID.h"
#include <iostream>

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp_, double Ki_, double Kd_) {


	// Controller gains
	Kp = Kp_;
	Kd = Kd_;
	Ki = Ki_;

	// Errors
	p_error = 0;
	i_error = 0;
	d_error = 0;
	current_error = 0;


	// Flags
	anti_windup = false;

	// Samples
	current_sample_number = 0;
	max_sample_number = 5000;

	// Limits 
	controller_output_MAX = 1.0;

}

void PID::UpdateError(double cte) {

	// Derivative part of controller
	d_error = cte - p_error;

	// Proportional part of controller
	p_error = cte;

	// Integral part of controller

	// Check anti-windup
	if (anti_windup) {

		i_error = i_error;
		//anti_windup = false;
	}
	else {

		i_error += cte;
	}

	// Use MSE of cte for current error
	current_error += (cte * cte);

	// Calculate average over the pre-defined number of samples
	if (current_sample_number == max_sample_number) {
		current_error = current_error / max_sample_number;
	}
	
}

double PID::TotalError() {
	return 0;
}


// Replaces TotalError()
double PID::CaculateControllerOutput(){

	// Basically the PID equation
	double controller_output = ((-Kp * p_error) + (-Ki * i_error) + (-Kd * d_error));

	// Anti-windup on integral component of controller
	if (controller_output >= controller_output_MAX) {
		controller_output = controller_output_MAX;
		anti_windup = true;
	}
	else if (controller_output <= -controller_output_MAX) {
		controller_output = -controller_output_MAX;
		anti_windup = true;
	}
	return controller_output;

}
