#ifndef PID_H
#define PID_H

class PID {
public:
	/*
	* Errors
	*/
	double p_error;
	double i_error;
	double d_error;

	/*
	* Coefficients
	*/ 
	double Kp;
	double Ki;
	double Kd;

	//******************
	// Added by me START
	//******************

	// Error(s)
	double current_error; 

	// Samples
	int current_sample_number; 
	int max_sample_number; 

	// Flags
	bool anti_windup; 

	// Limits
	int controller_output_MAX;


	//******************
	// Added by me START
	//******************

	/*
	* Constructor
	*/
	PID();

	/*
	* Destructor.
	*/
	virtual ~PID();

	/*
	* Initialize PID.
	*/
	void Init(double Kp, double Ki, double Kd);

	/*
	* Update the PID error variables given cross track error.
	*/
	void UpdateError(double cte);

	/*
	* Calculate the total PID error.
	*/
	double TotalError();

	//******************
	// Added by me START
	//******************

	// Replaces TotalError()
	double CaculateControllerOutput();

	//******************
	// Added by me END
	//******************

};

#endif /* PID_H */
