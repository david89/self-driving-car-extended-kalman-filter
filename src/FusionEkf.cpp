#include "FusionEkf.h"

#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEkf::FusionEkf() {
  previous_timestamp_ = 0;

  F_ = MatrixXd(4, 4);
  F_ << 1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1;

  Q_ = MatrixXd(4, 4);
  // We are going to use different values for Q_ in every measurement, so it's
  // not worth it initializing it.

  // Measurement covariance matrix - laser
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  // Measurement covariance matrix - radar
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;
}

const VectorXd&
FusionEkf::ProcessMeasurement(const MeasurementPackage& measurement_pack) {
  /**
   * Initialization
   */
  if (ekf_ == nullptr) {
    previous_timestamp_ = measurement_pack.timestamp;

    // first measurement
    VectorXd x(4);

    switch (measurement_pack.sensor_type) {
      case MeasurementPackage::RADAR:
        x << measurement_pack.raw_measurements(0) * std::sin(measurement_pack.raw_measurements(1)),
             measurement_pack.raw_measurements(0) * std::cos(measurement_pack.raw_measurements(1)),
             0,
             0;
        break;
      case MeasurementPackage::LASER:
        x << measurement_pack.raw_measurements(0),
             measurement_pack.raw_measurements(1),
             0,
             0;
        break;
      default:
        // TODO: maybe add a logging message as we don't know how to handle
        // this new source yet.
        break;
    }

    // State covariance matrix.
    MatrixXd P(4, 4);
    P << 1, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 1000, 0,
         0, 0, 0, 1000;

    // Measurement matrix.
    MatrixXd H(2, 4);
    H << 1, 0, 0, 0,
         0, 1, 0, 0;

    ekf_.reset(new KalmanFilter(std::move(x), std::move(P), std::move(H)));

    // done initializing, no need to predict or update
    return ekf_->x();
  }

  float dt = (measurement_pack.timestamp - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp;

  /**
   * Prediction
   */
  F_(0, 2) = dt;
  F_(1, 3) = dt;

  // We avoid creating the covariance matrix Q in every measurement to avoid
  // the number of allocations + deallocations.
  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;
  Q_ << dt_4 / 4 * noise_ax_, 0, dt_3 / 2 * noise_ax_, 0,
        0, dt_4 / 4 * noise_ay_, 0, dt_3 / 2 * noise_ay_,
        dt_3 / 2 * noise_ax_, 0, dt_2 * noise_ax_, 0,
        0, dt_3 / 2 * noise_ay_, 0, dt_2 * noise_ay_;
  ekf_->Predict(F_, Q_);

  /**
   * Update
   */
  if (measurement_pack.sensor_type == MeasurementPackage::RADAR) {
    ekf_->UpdateEkf(measurement_pack.raw_measurements, R_radar_);
  } else {
    ekf_->Update(measurement_pack.raw_measurements, R_laser_);
  }

  // print the output
  cout << "x_ = " << ekf_->x() << endl;
  cout << "P_ = " << ekf_->P() << endl;
  return ekf_->x();
}
