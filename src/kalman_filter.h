#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_

#include "Eigen/Dense"

class KalmanFilter {
 public:
  KalmanFilter() = default;
  virtual ~KalmanFilter() = default;

  // By default this class is copyable and movable.
  KalmanFilter(const KalmanFilter&) = default;
  KalmanFilter(KalmanFilter&&) = default;
  KalmanFilter& operator=(const KalmanFilter&) = default;
  KalmanFilter& operator=(KalmanFilter&&) = default;

  /**
   * Init Initializes Kalman filter
   * @param x_in Initial state
   * @param P_in Initial state covariance
   * @param H_in Measurement matrix
   * @param R_in Measurement covariance matrix
   * @param Q_in Process covariance matrix
   */
  void Init(Eigen::VectorXd& x_in, Eigen::MatrixXd& P_in, Eigen::MatrixXd& H_in,
            Eigen::MatrixXd& R_in, Eigen::MatrixXd& Q_in);

  /**
   * Prediction Predicts the state and the state covariance
   * using the process model
   */
  void Predict(const Eigen::MatrixXd& F);

  /**
   * Updates the state by using standard Kalman Filter equations
   * @param z The measurement at k+1
   */
  void Update(const Eigen::VectorXd &z);

  /**
   * Updates the state by using Extended Kalman Filter equations
   * @param z The measurement at k+1
   */
  void UpdateEkf(const Eigen::VectorXd &z);

  const Eigen::VectorXd x() const { return x_; }
  const Eigen::MatrixXd P() const { return P_; }

private:
  // state vector
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;

  // process covariance matrix
  Eigen::MatrixXd Q_;

  // measurement matrix
  Eigen::MatrixXd H_;

  // measurement covariance matrix
  Eigen::MatrixXd R_;
};

#endif // KALMAN_FILTER_H_
