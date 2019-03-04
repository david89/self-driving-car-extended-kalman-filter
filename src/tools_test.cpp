#include "tools.h"

#include <stdexcept>
#include <vector>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace {

using Eigen::VectorXd;
using Eigen::MatrixXd;

VectorXd getVector(const std::vector<float>& elements) {
  VectorXd ans(elements.size());
  for (size_t i = 0; i < elements.size(); i++) ans(i) = elements[i];
  return ans;
}

TEST(Tools, CalculateRmse) {
  std::vector<VectorXd> estimations = {
      getVector({1, 1, .2, .1}),
      getVector({2, 2, .3, .2}),
      getVector({3, 3, .4, .3}),
  };
  std::vector<VectorXd> ground_truth = {
      getVector({1.1, 1.2, .3, .2}),
      getVector({2.1, 2.2, .4, .3}),
      getVector({3.1, 3.2, .5, .4}),
  };

  VectorXd expected(4);
  expected << .1, .2, .1, .1;

  VectorXd result = tools::CalculateRmse(estimations, ground_truth);
  EXPECT_TRUE(expected.isApprox(result, 1e-6));
}

TEST(Tools, CalculateRmseExceptionWhenEstimationsIsEmpty) {
  EXPECT_THROW(tools::CalculateRmse(/*estimations=*/{}, /*ground_truth=*/{}),
               std::invalid_argument);
}

TEST(Tools, CalculateRmseWhenDifferentSizes) {
  std::vector<VectorXd> estimations = {
      getVector({1, 1, 0.2, 0.1}),
  };
  std::vector<VectorXd> ground_truth = {
      getVector({1.1, 1.1, 0.3, 0.2}),
      getVector({2.1, 2.1, 0.4, 0.3}),
  };

  EXPECT_THROW(tools::CalculateRmse(estimations, ground_truth),
               std::invalid_argument);
}

TEST(Tools, CalculateJacobian) {
  VectorXd x_state = getVector({1, 2, 0.2, 0.4});
  MatrixXd expected(3, 4);
  expected << 0.447214, 0.894427, 0, 0,
              -0.4, 0.2, 0, 0,
              0, 0, 0.447214, 0.894427;

  MatrixXd result = tools::CalculateJacobian(x_state);
  EXPECT_TRUE(expected.isApprox(result, 1e-6));
}

TEST(Tools, CalculateJacobianDivisionByZero) {
  VectorXd x_state = getVector({0, 0, 0.2, 0.4});
  EXPECT_THROW(tools::CalculateJacobian(x_state), std::invalid_argument);
}

}  // namespace
