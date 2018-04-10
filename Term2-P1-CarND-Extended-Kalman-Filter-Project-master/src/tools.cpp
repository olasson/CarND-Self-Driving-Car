#include <iostream>
#include "tools.h"
#include <math.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;


Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                                     const vector<VectorXd> &ground_truth) {

  // Initialize RMSE vector
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  // Preform validity check
  if(estimations.size()!=ground_truth.size() || estimations.size()==0){
    cout<<"ERROR--Tools::CalculateRMSE: Vectors estimations and ground_truth are not the same size OR vector estimations is empty!" << endl;
    return rmse;
  }
  for(unsigned int i=0; i<estimations.size(); i++){
    VectorXd residual = estimations[i] - ground_truth[i];
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

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state) {


  MatrixXd Hj(3,4);
  Hj <<  1,1,0,0,
         1,1,0,0,
         1,1,1,1;

  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);


  // Pre-calculated values for convenience
  float c1 = (px * px) + (py * py);
  float c2 = sqrt(c1);
  float c3 = c1 * c2;

  if(fabs(c1) < 0.001){
		cout << "ERROR--Tools::CalculateRMSE: Division by zero" << endl;
		return Hj;
	}

  Hj<<(px / c2), (py / c2), 0, 0,
     -(py / c1), (px / c1), 0, 0,
      py * ((vx * py) - (vy * px)) / c3, px * ((vy * px) - (vx * py)) / c3, (px / c2), (py / c2);

  return Hj;

}
