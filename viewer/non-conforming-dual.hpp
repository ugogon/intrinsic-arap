#pragma once

#include <Eigen/Sparse>
#include <vector>

struct ARAPData {
  std::vector<Eigen::SparseMatrix<double,Eigen::RowMajor>> Adjacency;
  std::vector<Eigen::Matrix<double, -1, 3>> edges;
  std::vector<Eigen::VectorXd> weights;
  Eigen::MatrixXd RAll;
};

void transformed_dual(
  const Eigen::MatrixXd& V, 
  const Eigen::MatrixXd& U, 
  const Eigen::MatrixXi& F, 
  ARAPData& data, 
  Eigen::MatrixXd& Vout, 
  Eigen::MatrixXi& Fout
);
  
void precomputation(
  const Eigen::MatrixXd& V, 
  const Eigen::MatrixXi& F, 
  ARAPData& data
);
