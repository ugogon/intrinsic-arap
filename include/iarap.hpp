#pragma once

#include "iarap_data.hpp"

void initMesh(const Eigen::Matrix<double, -1, 3>& V, const Eigen::Matrix<int, -1, 3>& F, iARAPData& data);
void build_intrinsic_mesh(const Eigen::Matrix<double, -1, 3>& V, const Eigen::Matrix<int, -1, 3>& F, iARAPData& data);
void iarap_precompute(const Eigen::VectorXi& constraints_indices, iARAPData& data);
void iarap_solve(const Eigen::MatrixXd& constraints, iARAPData& data, Eigen::MatrixXd& U);
double iarap_energy(const iARAPData& data, const Eigen::MatrixXd& U);
