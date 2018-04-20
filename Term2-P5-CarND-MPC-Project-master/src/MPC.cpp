
#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

// *******************
// Changed by me START
// *******************

size_t N = 10; // Changed to current value from 0
double dt = 0.1; // Changed to current value from 0

// *******************
// Changed by me END
// *******************

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

// *****************
// Added by me START
// *****************

// Define reference values
const double REF_CTE = 0.0;
const double REF_EPSI = 0.0;
const double REF_VELOCITY = 40.0;

// Define weights for cost calculations
const double WEIGHT_CTE = 2000.0;
const double WEIGHT_EPS = 2000.0;
const double WEIGHT_ACT = 5.0;
const double WEIGHT_DELTA_DIFF = 200.0;
const double WEIGHT_A_DIFF = 10.0;

// Define index positions to simplify accessing certain variables
const size_t IDX_X_START = 0.0;
const size_t IDX_Y_START = IDX_X_START + N;
const size_t IDX_PSI_START = IDX_Y_START + N;
const size_t IDX_V_START = IDX_PSI_START + N;
const size_t IDX_CTE_START = IDX_V_START + N;
const size_t IDX_EPS_START = IDX_CTE_START + N;
const size_t IDX_DELTA_START = IDX_EPS_START + N;
const size_t IDX_A_START = IDX_DELTA_START + N - 1;

// Define max values
const double MAX_NON_ACT = 1.0e19;
const double MAX_DELTA = 0.436332 * Lf;
const double MAX_A = 1.0;

// *****************
// Added by me END
// *****************

class FG_eval {
public:
	// Fitted polynomial coefficients
	Eigen::VectorXd coeffs;
	FG_eval(Eigen::VectorXd coeffs) { this->coeffs = coeffs; }

	typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
	void operator()(ADvector& fg, const ADvector& vars) {


		// fg is a vector of the cost constraints
		fg[0] = 0;


		// Calculate the reference state cost
		for (unsigned int t = 0; t < N; t++) {
			fg[0] += WEIGHT_CTE * CppAD::pow(vars[IDX_CTE_START + t] - REF_CTE, 2);
			fg[0] += WEIGHT_EPS * CppAD::pow(vars[IDX_EPS_START + t] - REF_EPSI, 2);
			fg[0] += CppAD::pow(vars[IDX_V_START + t] - REF_VELOCITY, 2);
		}

		// Calculate cost to minimize the use of actuators
		for (unsigned int t = 0; t < N - 1; t++) {
			fg[0] += WEIGHT_ACT * CppAD::pow(vars[IDX_DELTA_START + t], 2);
			fg[0] += WEIGHT_ACT * CppAD::pow(vars[IDX_A_START + t], 2);
		}

		// Minimize the value gap between sequential actuations.
		for (unsigned int t = 0; t < N - 2; t++) {
			fg[0] += WEIGHT_DELTA_DIFF  * CppAD::pow(vars[IDX_DELTA_START + t + 1] - vars[IDX_DELTA_START + t], 2);
			fg[0] += WEIGHT_A_DIFF * CppAD::pow(vars[IDX_A_START + t + 1] - vars[IDX_A_START + t], 2);
		}

		// Constraints setup
		// Since fg[0] is reserved for cost, bump all other indices up by 1
		fg[1 + IDX_X_START] = vars[IDX_X_START];
		fg[1 + IDX_Y_START] = vars[IDX_Y_START];
		fg[1 + IDX_PSI_START] = vars[IDX_PSI_START];
		fg[1 + IDX_V_START] = vars[IDX_V_START];
		fg[1 + IDX_CTE_START] = vars[IDX_CTE_START];
		fg[1 + IDX_EPS_START] = vars[IDX_EPS_START];

		// Set the rest of the constraints
		for (unsigned int t = 0; t < N - 1; t++) {
			// Constrain to 0

			// The state at time t
			AD<double> x0 = vars[IDX_X_START + t];
			AD<double> psi0 = vars[IDX_PSI_START + t];
			AD<double> v0 = vars[IDX_V_START + t];
			AD<double> y0 = vars[IDX_Y_START + t];
			AD<double> cte0 = vars[IDX_CTE_START + t];
			AD<double> epsi0 = vars[IDX_EPS_START + t];

			// The actuations at time t
			AD<double> delta0 = vars[IDX_DELTA_START + t];
			AD<double> a0 = vars[IDX_A_START + t];
			AD<double> f0 = coeffs[0] + (coeffs[1] * x0) + (coeffs[2] * x0 * x0) + (coeffs[3] * x0 * x0 * x0);
			AD<double> psides0 = CppAD::atan((3 * coeffs[3] * x0 * x0) + (2 * coeffs[2] * x0) + coeffs[1]);

			// The state at time t + 1
			AD<double> x1 = vars[IDX_X_START + t + 1];
			AD<double> y1 = vars[IDX_Y_START + t + 1];
			AD<double> psi1 = vars[IDX_PSI_START + t + 1];
			AD<double> v1 = vars[IDX_V_START + t + 1];
			AD<double> cte1 = vars[IDX_CTE_START + t + 1];
			AD<double> epsi1 = vars[IDX_EPS_START + t + 1];

			// Based on vehicle model equations from udacity lessons
			fg[2 + IDX_X_START + t] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
			fg[2 + IDX_Y_START + t] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
			fg[2 + IDX_PSI_START + t] = psi1 - (psi0 - (v0 * delta0 * dt / Lf));
			fg[2 + IDX_V_START + t] = v1 - (v0 + (a0 * dt));
			fg[2 + IDX_CTE_START + t] = cte1 - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * dt));
			fg[2 + IDX_EPS_START + t] = epsi1 - ((psi0 - psides0) - ((v0 * dt * delta0) / Lf));
		}
	}
};

//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
	bool ok = true;
	typedef CPPAD_TESTVECTOR(double) Dvector;

	size_t n_vars = (6 * N) + (2 * (N - 1)); // Changed to current value from 0
	size_t n_constraints = 6 * N; // Changed to current value from 0

	// Unpack the state variables
	double x = state[0];
	double y = state[1];
	double psi = state[2];
	double v = state[3];
	double cte = state[4];
	double epsi = state[5];

	// Set the initial value of the independent variables.
	Dvector vars(n_vars);
	for (unsigned int i = 0; i < n_vars; i++) {
		vars[i] = 0;
	}
	// Set the initial values of the variables
	vars[IDX_X_START] = x;
	vars[IDX_Y_START] = y;
	vars[IDX_PSI_START] = psi;
	vars[IDX_V_START] = v;
	vars[IDX_CTE_START] = cte;
	vars[IDX_EPS_START] = epsi;

	Dvector vars_lowerbound(n_vars);
	Dvector vars_upperbound(n_vars);

	// Set the upper and lower limits on non-actuators
	for (unsigned int i = 0; i < IDX_DELTA_START; i++) {
		vars_lowerbound[i] = -MAX_NON_ACT;
		vars_upperbound[i] = MAX_NON_ACT;
	}

	// Set the uppper and lower limits on delta
	for (unsigned int i = IDX_DELTA_START; i < IDX_A_START; i++) {
		vars_lowerbound[i] = -MAX_DELTA;
		vars_upperbound[i] = MAX_DELTA;
	}

	// Set the upper and lower limits on acceleration
	for (unsigned int i = IDX_A_START; i < n_vars; i++) {
		vars_lowerbound[i] = -MAX_A;
		vars_upperbound[i] = MAX_A;
	}

	// Set lower and upper limits for the constraints
	Dvector constraints_lowerbound(n_constraints);
	Dvector constraints_upperbound(n_constraints);
	for (unsigned int i = 0; i < n_constraints; i++) {
		constraints_lowerbound[i] = 0;
		constraints_upperbound[i] = 0;
	}
	constraints_lowerbound[IDX_X_START] = x;
	constraints_lowerbound[IDX_Y_START] = y;
	constraints_lowerbound[IDX_PSI_START] = psi;
	constraints_lowerbound[IDX_V_START] = v;
	constraints_lowerbound[IDX_CTE_START] = cte;
	constraints_lowerbound[IDX_EPS_START] = epsi;

	constraints_upperbound[IDX_X_START] = x;
	constraints_upperbound[IDX_Y_START] = y;
	constraints_upperbound[IDX_PSI_START] = psi;
	constraints_upperbound[IDX_V_START] = v;
	constraints_upperbound[IDX_CTE_START] = cte;
	constraints_upperbound[IDX_EPS_START] = epsi;

	// object that computes objective and constraints
	FG_eval fg_eval(coeffs);

	//
	// NOTE: You don't have to worry about these options
	//
	// options for IPOPT solver
	std::string options;
	// Uncomment this if you'd like more print information
	options += "Integer print_level  0\n";
	// NOTE: Setting sparse to true allows the solver to take advantage
	// of sparse routines, this makes the computation MUCH FASTER. If you
	// can uncomment 1 of these and see if it makes a difference or not but
	// if you uncomment both the computation time should go up in orders of
	// magnitude.
	options += "Sparse  true        forward\n";
	options += "Sparse  true        reverse\n";
	// NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
	// Change this as you see fit.
	options += "Numeric max_cpu_time          0.5\n";

	// place to return solution
	CppAD::ipopt::solve_result<Dvector> solution;

	// solve the problem
	CppAD::ipopt::solve<Dvector, FG_eval>(
	options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
	constraints_upperbound, fg_eval, solution);

	// Check some of the solution values
	ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

	
	vector<double> result;

	result.push_back(solution.x[IDX_DELTA_START]);
	result.push_back(solution.x[IDX_A_START]);

	for (unsigned int t = 0; t < N - 1; t++) {
		result.push_back(solution.x[IDX_X_START + t + 1]);
		result.push_back(solution.x[IDX_Y_START + t + 1]);
	}

	return result;
}