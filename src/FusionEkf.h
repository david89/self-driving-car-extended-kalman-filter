#ifndef FusionEkf_H_
#define FusionEkf_H_

#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "Eigen/Dense"
#include "kalman_filter.h"
#include "measurement_package.h"
#include "tools.h"

class FusionEkf {
 public:
  /**
   * Constructor.
   */
  FusionEkf();

  /**
   * Destructor.
   */
  virtual ~FusionEkf() = default;

  /**
   * Run the whole flow of the Kalman Filter from here.
   */
  const Eigen::VectorXd&
  ProcessMeasurement(const MeasurementPackage& measurement_pack);

private:
  // previous timestamp
  long long previous_timestamp_;

  // State transition matrix.
  Eigen::MatrixXd F_;

  // Process covariance matrix
  Eigen::MatrixXd Q_;

  // Measurement covariance matrices.
  Eigen::MatrixXd R_laser_;
  Eigen::MatrixXd R_radar_;

  // Acceleration noise components.
  const float noise_ax_ = 9;
  const float noise_ay_ = 9;

  /**
   * Kalman Filter update and prediction math lives in here.
   */
  std::unique_ptr<KalmanFilter> ekf_;
};

#endif // FusionEkf_H_
