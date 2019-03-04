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
}
}  // namespace tools
