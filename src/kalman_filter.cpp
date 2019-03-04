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

void KalmanFilter::Init(VectorXd& x_in, MatrixXd& P_in, MatrixXd& H_in,
                        MatrixXd& R_in, MatrixXd& Q_in) {
  x_ = x_in;
  P_ = P_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict(const MatrixXd& F) {
  x_ = F * x_;
  P_ = F * P_ * F.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  MatrixXd Ht = H_.transpose();
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());

  VectorXd y = z - H_ * x_;
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();

  x_ = x_ + (K * y);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEkf(const VectorXd &z) {
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
  MatrixXd Hjt = Hj.transpose();
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  MatrixXd S = Hj * P_ * Hjt + R_;
  MatrixXd K = P_ * Hjt * S.inverse();

  x_ = x_ + (K * y);
  P_ = (I - K * Hj) * P_;
}
