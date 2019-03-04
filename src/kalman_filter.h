#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_

#include "Eigen/Dense"

class KalmanFilter {
 public:
   KalmanFilter(Eigen::VectorXd x, Eigen::MatrixXd P, Eigen::MatrixXd H);

   virtual ~KalmanFilter() = default;

   // By default this class is copyable and movable.
   KalmanFilter(const KalmanFilter&) = default;
   KalmanFilter(KalmanFilter&&) = default;
   KalmanFilter& operator=(const KalmanFilter&) = default;
   KalmanFilter& operator=(KalmanFilter&&) = default;

   /**
    * Prediction Predicts the state and the state covariance
    * using the process model
    */
   void Predict(const Eigen::MatrixXd& F, const Eigen::MatrixXd& Q);

   /**
    * Updates the state by using standard Kalman Filter equations
    * @param z The measurement at k+1
    */
   void Update(const Eigen::VectorXd& z, const Eigen::MatrixXd& R);

   /**
    * Updates the state by using Extended Kalman Filter equations
    * @param z The measurement at k+1
    */
   void UpdateEkf(const Eigen::VectorXd& z, const Eigen::MatrixXd& R);

   const Eigen::VectorXd x() const { return x_; }
   const Eigen::MatrixXd P() const { return P_; }

 private:
   // Common method for updating the internal state of the Kalman filter after
   // analyzing a measurement.
   void UpdateImpl(const Eigen::VectorXd& y, const Eigen::MatrixXd& R,
                   const Eigen::MatrixXd& H);

   // state vector
   Eigen::VectorXd x_;

   // state covariance matrix
   Eigen::MatrixXd P_;

   // measurement matrix
   Eigen::MatrixXd H_;
};

#endif // KALMAN_FILTER_H_
