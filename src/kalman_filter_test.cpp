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

}  // namespace