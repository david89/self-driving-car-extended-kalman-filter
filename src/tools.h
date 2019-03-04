#ifndef TOOLS_H_
#define TOOLS_H_

#include <vector>
#include "Eigen/Dense"

namespace tools {
/**
 * A helper method to calculate RMSE.
 */
Eigen::VectorXd CalculateRmse(const std::vector<Eigen::VectorXd>& estimations,
                              const std::vector<Eigen::VectorXd>& ground_truth);

/**
 * A helper method to calculate Jacobians.
 */
Eigen::MatrixXd CalculateJacobian(const Eigen::VectorXd& x_state);
}  // namespace tools

#endif  // TOOLS_H_
