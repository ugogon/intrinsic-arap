#pragma once

#include <Eigen/Sparse>

Eigen::SparseMatrix<double> externalized_intrinsic_Laplace(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);
Eigen::SparseMatrix<double> positive_externalized_intrinsic_Laplace(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);