#include "kalman_filter.h"

#include <vector>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace {

using Eigen::VectorXd;
using Eigen::MatrixXd;

TEST(KalmanFilter, KalmanFilterPrediction) {
  VectorXd x(4);
  x << 1, 2, 3, 4;

  MatrixXd P(4, 4);
  P << 1, 0, 0, 0,
       0, 1, 0, 0,
       0, 0, 2, 0,
       0, 0, 0, 2;

  MatrixXd F(4, 4);
  F << 1, 1, 1, 1,
       0, 1, 1, 1,
       0, 0, 1, 1,
       0, 0, 0, 1;

  MatrixXd H(2, 4);
  H << 1, 0, 0, 0,
       0, 1, 0, 0;

  MatrixXd R(2, 2);
  R << 0.5, 0,
       0, 0.5;

  MatrixXd Q(4, 4);
  Q << 1, 1, 1, 1,
       1, 1, 1, 1,
       1, 1, 1, 1,
       1, 1, 1, 1;

  KalmanFilter kf;
  kf.Init(x, P, F, H, R, Q);
  kf.Predict();

  // x' = F * x
  // And considering the way F is set up, x' will be the sum of the suffixes of:
  // [1, 2, 3, 4].
  VectorXd result_x = kf.x();

  VectorXd expected_x(4);
  expected_x << 10, 9, 7, 4;

  EXPECT_TRUE(result_x.isApprox(expected_x, 1e-6));

  // P' = F * P * transpose(F) + Q.
  MatrixXd result_P = kf.P();

  MatrixXd expected_P(4, 4);
  expected_P << 7, 6, 5, 3,
                6, 6, 5, 3,
                5, 5, 5, 3,
                3, 3, 3, 3;

  EXPECT_TRUE(result_P.isApprox(expected_P, 1e-6));
}

TEST(KalmanFilter, KalmanFilterUpdate) {
  VectorXd x(2);
  x << 1, 1;

  MatrixXd P(2, 2);
  P << 1000, 0,
       0, 1000;

  MatrixXd F(2, 2);
  F << 1, 1,
       0, 1;

  MatrixXd H(1, 2);
  H << 1, 0;

  MatrixXd R(1, 1);
  R << 0.5;

  MatrixXd Q(2, 2);
  Q << 0, 0,
       0, 0;

  VectorXd z(1);
  z << 2;

  KalmanFilter kf;
  kf.Init(x, P, F, H, R, Q);
  kf.Update(z);

  // x' = x + K * y, where
  // K = P * transpose(H) * inverse(S),
  // S = H * P * transpose(H) + R
  // y = z - H * x
  VectorXd result_x = kf.x();

  VectorXd expected_x(2);
  expected_x << 1.9995, 1;

  EXPECT_TRUE(result_x.isApprox(expected_x, 1e-6));

  // P' = (I - K * H) * P.
  MatrixXd result_P = kf.P();

  MatrixXd expected_P(2, 2);
  expected_P << 0.5, 0,
                0, 1000;

  EXPECT_TRUE(result_P.isApprox(expected_P, 1e-6));
}

TEST(KalmanFilter, KalmanFilterUpdateEkf) {
  VectorXd x(4);
  x << 1, 1, 1, 1;

  MatrixXd P(4, 4);
  P << 1, 0, 0, 0,
       0, 1, 0, 0,
       0, 0, 1000, 0,
       0, 0, 0, 1000;

  MatrixXd F(4, 4);
  F << 1, 1, 1, 1,
       0, 1, 1, 1,
       0, 0, 1, 1,
       0, 0, 0, 1;

  MatrixXd H(2, 4);
  H << 1, 0, 0, 0,
       0, 1, 0, 0;

  MatrixXd R(3, 3);
  R << 0.5, 0, 0,
       0, 0.5, 0,
       0, 0, 0.5;

  MatrixXd Q(4, 4);
  Q << 1, 1, 1, 1,
       1, 1, 1, 1,
       1, 1, 1, 1,
       1, 1, 1, 1;

  VectorXd z(3);
  z << 2.8284, 0.7853, 1;

  KalmanFilter kf;
  kf.Init(x, P, F, H, R, Q);
  kf.UpdateEkf(z);

  // x' = x + K * y, where
  // K = P * transpose(Hj) * inverse(S),
  // S = Hj * P * transpose(Hj) + R
  // Hj = Jacobian(x)
  // y = z - h(x)
  VectorXd result_x = kf.x();

  VectorXd expected_x(4);
  expected_x << 1.666703, 1.666605, 0.707253, 0.707253;

  EXPECT_TRUE(result_x.isApprox(expected_x, 1e-6));

  // P' = (I - K * Hj) * P.
  MatrixXd result_P = kf.P();

  MatrixXd expected_P(4, 4);
  expected_P << 0.416667, -0.083333, 0.000000, 0.000000,
                -0.083333, 0.416667, 0.000000, 0.000000,
                0.000000, 0.000000, 500.249875, -499.750125,
                0.000000, 0.000000, -499.750125, 500.249875;

  EXPECT_TRUE(result_P.isApprox(expected_P, 1e-6));
}

TEST(KalmanFilter, KalmanFilterUpdateEkfThrowWhenZeroPx) {
  VectorXd x(4);
  x << 0, 1, 1, 1;

  MatrixXd P(4, 4);
  P << 1, 0, 0, 0,
       0, 1, 0, 0,
       0, 0, 1000, 0,
       0, 0, 0, 1000;

  MatrixXd F(4, 4);
  F << 1, 1, 1, 1,
       0, 1, 1, 1,
       0, 0, 1, 1,
       0, 0, 0, 1;

  MatrixXd H(2, 4);
  H << 1, 0, 0, 0,
       0, 1, 0, 0;

  MatrixXd R(3, 3);
  R << 0.5, 0, 0,
       0, 0.5, 0,
       0, 0, 0.5;

  MatrixXd Q(4, 4);
  Q << 1, 1, 1, 1,
       1, 1, 1, 1,
       1, 1, 1, 1,
       1, 1, 1, 1;

  VectorXd z(3);
  z << 2.8284, 0.7853, 1;

  KalmanFilter kf;
  kf.Init(x, P, F, H, R, Q);
  EXPECT_THROW(kf.UpdateEkf(z), std::invalid_argument);
}
}  // namespace
