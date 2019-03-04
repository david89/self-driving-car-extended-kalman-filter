#include "tools.h"

#include <stdexcept>
#include <iostream>

namespace tools {
namespace {
using Eigen::VectorXd;
using Eigen::MatrixXd;
}  // namespace

VectorXd CalculateRmse(const std::vector<VectorXd>& estimations,
                       const std::vector<VectorXd>& ground_truth) {
  if (estimations.empty()) {
    throw std::invalid_argument("estimations shouldn't be empty");
  }
  if (estimations.size() != ground_truth.size()) {
    throw std::invalid_argument(
        "estimations and ground_truth should be of the same size");
  }
  // TODO: more validations if the size of any row is not 4.

  // Implementing the Root Mean Squared Error (RMSE):
  // https://en.wikipedia.org/wiki/Root-mean-square_deviation
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  for (size_t i = 0; i < estimations.size(); i++) {
    VectorXd diff = estimations[i] - ground_truth[i];
    diff = diff.array() * diff.array();
    rmse += diff;
  }

  rmse = rmse / estimations.size();
  return rmse.array().sqrt();
}

MatrixXd CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3, 4);

  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  float den2 = px * px + py * py;
  if (abs(den2) < 1e-9) {
    // TODO: is it OK to return an exception?
    throw std::invalid_argument("px**2 + py**2 is 0");
  }

  float den1 = sqrt(den2);
  float den3 = den1 * den2;
  float vp = vx * py - vy * px;

  Hj << px / den1, py / den1, 0, 0,
        -py / den2, px / den2, 0, 0,
        py * vp / den3, px * -vp / den3, px / den1, py / den1;
  return Hj;
}

}  // namespace tools
