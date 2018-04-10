#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <math.h>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

// Constants
const float NOISE_AX = 9;
const float NOISE_AY = 9;


// Constructor
FusionEKF::FusionEKF() {
	is_initialized_ = false;
	previous_timestamp_ = 0;

	// Initializing matricies
	R_laser_ = MatrixXd(2,2);
	R_radar_ = MatrixXd(3,3);
	H_laser_ = MatrixXd(2,4);
	Hj_ = MatrixXd(3,4);
	P_ = MatrixXd(4, 4);
	F_ = MatrixXd(4, 4);
	Q_ = MatrixXd(4,4);

	// Initializing vector(s)
    x_in = VectorXd(4);

	// Measurement covariance matrix - laser
	R_laser_ << 0.0225, 0,
				0, 0.0225;

    // Measurement covariance matrix - radar
    R_radar_ << 0.09, 0, 0,
                0, 0.0009, 0,
                0, 0, 0.09;

    H_laser_ << 1,0,0,0,
                0,1,0,0;

    Hj_ << 1, 1, 0, 0,
           1, 1, 0, 0,
           1, 1, 1, 1; 

    // The initial transition matrix F_
    F_ << 1, 0, 1, 0,
          0, 1, 0, 1,
          0, 0, 1, 0,
          0, 0, 0, 1;

    // State covariance matrix P
    P_ << 1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1000, 0,
          0, 0, 0, 1000;
    
    // State vector
    x_in <<1,1,1,1;
}

// Destructor
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
	//**************
	//Initialization
	//**************

	if (!is_initialized_){

    	// First measurement
     	cout << "EKF: " << endl;
     	ekf_.Init(x_in,P_,F_,H_laser_,R_laser_,Q_);

     	if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
     	    // Initialize state
     		ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0.0, 0.0;
    	} else if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    	    //Convert radar from polar to cartesian coordinates and initialize state.x
    		float rho = measurement_pack.raw_measurements_[0];
    		float phi = measurement_pack.raw_measurements_[1];
    		float rho_dot = measurement_pack.raw_measurements_[2];


        	ekf_.x_(0) = rho * cos(phi);
        	ekf_.x_(1) = rho * sin(phi);
        	ekf_.x_(2) = rho_dot * cos(phi);
        	ekf_.x_(3) = rho_dot * sin(phi); 
    	}
    	// Done initializing, no need to predict or update
    	is_initialized_ = true;

    	previous_timestamp_ = measurement_pack.timestamp_;
    	return;
	}

	//**************
    //Prediction
    //**************

    float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
  	previous_timestamp_ = measurement_pack.timestamp_;

	float dt_2 = dt   * dt;
	float dt_3 = dt_2 * dt;
	float dt_4 = dt_3 * dt;

	// Modify the F matrix so that the time is integrated
	ekf_.F_(0, 2) = dt;
	ekf_.F_(1, 3) = dt;

	// Set the process covariance matrix Q
	ekf_.Q_ << ((dt_4 / 4) * NOISE_AX), 0, ((dt_3 / 2) * NOISE_AX), 0,
				0, ((dt_4 / 4) * NOISE_AY), 0, ((dt_3 / 2) * NOISE_AY),
	            ((dt_3 / 2) * NOISE_AX), 0, (dt_2 * NOISE_AX), 0,
	            0, ((dt_3 / 2) * NOISE_AY), 0, (dt_2 * NOISE_AY);

	ekf_.Predict();

	//**************
    //Update
    //**************

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        // Radar updates
    	Hj_= tools.CalculateJacobian(ekf_.x_);
    	ekf_.H_ = Hj_;
    	ekf_.R_= R_radar_;
    	ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  	} else{
  	    // Laser updates
  		ekf_.H_= H_laser_;
    	ekf_.R_= R_laser_;
    	ekf_.Update(measurement_pack.raw_measurements_);
  	}
}
