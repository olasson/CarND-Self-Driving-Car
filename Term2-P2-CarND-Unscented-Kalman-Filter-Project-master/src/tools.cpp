#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
							  const vector<VectorXd> &ground_truth){
    // Initialize RMSE vector
	VectorXd rmse(4);
	rmse << 0,0,0,0;

	// Preform validity check
	if(estimations.size()!=ground_truth.size() || estimations.size()==0){
		cout<<"ERROR--Tools::CalculateRMSE: Vectors estimations and ground_truth are not the same size OR vector estimations is empty!" << endl;
		return rmse;
	}
	
	VectorXd residual; 

	for(unsigned int i = 0; i < estimations.size(); i++){
		residual = estimations[i] - ground_truth[i];
		residual = residual.array() * residual.array();
		rmse += residual;
	}

	// Calculate the mean
	rmse = rmse / estimations.size();

	// Calculate the squared root
	rmse = rmse.array().sqrt();

	// Return the result
	return rmse;
}