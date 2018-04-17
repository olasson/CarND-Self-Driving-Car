/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

	//cout << "Initializing particle filter..." << endl;
	
	// For generating pseudo-random numbers
	// From <random>
	default_random_engine random_number_gen;

	// Create a normal Gaussian distribution for x, y and theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	num_particles = 10;

	// Initialize particles and weights
	for (signed int i = 0; i < num_particles; i++) {


		Particle particle;
		particle.id = i;
		
		// Sample x,y,theta particles from the normal distrubtions
		particle.x = dist_x(random_number_gen);
		particle.y = dist_y(random_number_gen);
		particle.theta = dist_theta(random_number_gen);
		

		particle.weight = 1.0;
		
		particles.push_back(particle);
		weights.push_back(1.0);
	}

	// Set init flag.
	is_initialized = true;
	//cout << "Initialized partifcle filter with " << num_particles << " particles." << endl;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	
	// For generating pseudo-random numbers
	// From <random>
	default_random_engine random_number_gen;

	// Based on bicycle model
	for (signed int i = 0; i < num_particles; i++) {

		// Get pose
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;

		// Avoid division by zero and predict
		if (fabs(yaw_rate) > 0.001) {
			particles[i].x = x + ((velocity / yaw_rate) * (sin(theta + (yaw_rate * delta_t)) - sin(theta)));
			particles[i].y = y + ((velocity / yaw_rate) * (cos(theta) - cos(theta + (yaw_rate * delta_t))));
			particles[i].theta = theta + (yaw_rate * delta_t);
		}
		else {
			particles[i].x = x + (velocity * delta_t * cos(theta));
			particles[i].y = y + (velocity * delta_t * sin(theta));
			particles[i].theta = theta;
		}

		// Add random Gaussian noise


		// Create a normal Gaussian distribution for x 
		normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
		// Random sample
		particles[i].x = dist_x(random_number_gen);

		// Create a normal (Gaussian) distribution for y
		normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
		// Random sample
		particles[i].y = dist_y(random_number_gen);

		// Create a normal (Gaussian) distribution for theta
		normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);
		// Random sample
		particles[i].theta = dist_theta(random_number_gen);

	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

	LandmarkObs nearest_landmark;

	for (unsigned int i = 0; i < observations.size(); i++) {
		double dist_min = numeric_limits<double>::max();
		for (unsigned int j = 0; j < predicted.size(); j++) {

			// Calculate distance to all landmarks
			double dist_diff = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);

			// Iteratively get minimum and leave with the correct association
			if (dist_diff < dist_min)
			{
				dist_min = dist_diff;
				observations[i].id = predicted[j].id;
				nearest_landmark = predicted[j];
			}
			
		}

	}


}

// Two useful links for implementing this function provided by Udacity
// 1. https://www.willamette.edu/~gorr/classes/random_number_generalGraphics/Transforms/transforms2d.htm
// 2. http://planning.cs.uiuc.edu/node99.html (equation 3.33)

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
		
	for (signed int i = 0; i < num_particles; i++) {

		Map::single_landmark_s single_landmark;

		// For in-range landmarks
		LandmarkObs in_range_landmark;
		vector<LandmarkObs> map_in_range_landmarks;

		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {

			single_landmark = map_landmarks.landmark_list[j];

			// Check if the x,y landmarks are in range of particles
			if ((fabs(single_landmark.x_f - particles[i].x) <= sensor_range) &&
				(fabs(single_landmark.y_f - particles[i].y) <= sensor_range))
			{
				// Set values of in-range landmark and push it into in-range vector
				in_range_landmark.id = single_landmark.id_i;
				in_range_landmark.x = single_landmark.x_f;
				in_range_landmark.y = single_landmark.y_f;
				map_in_range_landmarks.push_back(in_range_landmark);
			}
		}
		
		
		// For landmark observations
		LandmarkObs observation;
		vector<LandmarkObs> map_observations;

		for (unsigned int j = 0; j < observations.size(); j++) {

			LandmarkObs transformed_observation;
			observation = observations[j];

			// Create a vector of transformed observations
			transformed_observation.x = particles[i].x + ((observation.x * cos(particles[i].theta)) - (observation.y * sin(particles[i].theta)));
			transformed_observation.y = particles[i].y + ((observation.x * sin(particles[i].theta)) + (observation.y * cos(particles[i].theta)));
			transformed_observation.id = observations[j].id;
			map_observations.push_back(transformed_observation);
						
			}

		// Associate global observations to nearest landmarks
		dataAssociation(map_in_range_landmarks, map_observations);

		// Reinitialize weights before updating them
		particles[i].weight = 1.0;
		weights[i] = 1.0;

		// Init vectors for associations, sense_x and sense_y
		vector<int> associations;
		vector<double> sense_x;
		vector<double> sense_y;

		// Loop over map_observations and in_range landmarks 
		for (unsigned int j = 0; j < map_observations.size(); j++){
			for (unsigned int k = 0; k < map_in_range_landmarks.size(); k++) {

				// Check matching pairs of observations and in-range landmarks
				if (map_observations[j].id == map_in_range_landmarks[k].id) {

					double gaussian_normalizer = 1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]);

					double prob_x = pow((map_observations[j].x - map_in_range_landmarks[k].x) / std_landmark[0], 2) / 2.0;
					double prob_y = pow((map_observations[j].y - map_in_range_landmarks[k].y) / std_landmark[1], 2) / 2.0;
					
					double prob_w = gaussian_normalizer * exp(-(prob_x + prob_y));

					particles[i].weight *= prob_w;
					weights[i] = particles[i].weight;

				}
				
			}
			// Update vectors for associations, sense_x and sense_y
			associations.push_back(map_observations[j].id);
			sense_x.push_back(map_observations[j].x);
			sense_y.push_back(map_observations[j].y);
		}
		// Assign particle associations
		SetAssociations(particles[i], associations, sense_x, sense_y);

	}

}

// Useful link for implementing this function provided by Udacity
// 1. http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

void ParticleFilter::resample() {
		
	// For generating pseudo-random numbers
	// From <random>
	default_random_engine random_number_gen;

	// Discrete distribution for weights
	discrete_distribution<int> dist_w(weights.begin(), weights.end());

	vector<Particle> resampled_particles;

	for (signed int i = 0; i < num_particles; i++) {

		resampled_particles.push_back(particles[dist_w(random_number_gen)]);
	}

	particles = resampled_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
