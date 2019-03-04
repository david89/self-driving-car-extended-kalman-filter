#include "kalman_filter.h"

#include <cmath>
#include "tools.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {
const double kPi = 4.0 * std::atan(1.0);

float normalizeAngle(float angle) {
  while (angle > kPi) angle -= 2 * kPi;
  while (angle < -kPi) angle += 2 * kPi;
  return angle;
}
}  // namespace

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

void KalmanFilter::Init(VectorXd& x_in, MatrixXd& P_in, MatrixXd& H_in) {
  x_ = x_in;
  P_ = P_in;
  H_ = H_in;
}

void KalmanFilter::Predict(const MatrixXd& F, const MatrixXd& Q) {
  x_ = F * x_;
  P_ = F * P_ * F.transpose() + Q;
}

void KalmanFilter::Update(const VectorXd &z, const MatrixXd& R) {
  VectorXd y = z - H_ * x_;
  UpdateImpl(y, R, H_);
}

void KalmanFilter::UpdateEkf(const VectorXd &z, const MatrixXd& R) {
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);
  float nor = sqrt(px * px + py * py);

  if (abs(px) < 1e-9) {
    throw std::invalid_argument("px cannot be 0.");
  }

  VectorXd h(3);
  h << nor, std::atan2(py, px), (px * vx + py * vy) / nor;
  VectorXd y = z - h;
  // The angle needs to be normalized after any addition/substraction.
  y(1) = normalizeAngle(y(1));
  MatrixXd Hj = tools::CalculateJacobian(x_);

  UpdateImpl(y, R, Hj);
}

void KalmanFilter::UpdateImpl(const VectorXd& y, const MatrixXd& R,
                              const MatrixXd& H) {
  MatrixXd Ht = H.transpose();
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  MatrixXd S = H * P_ * Ht + R;
  MatrixXd K = P_ * Ht * S.inverse();

  x_ = x_ + (K * y);
  P_ = (I - K * H) * P_;
}
