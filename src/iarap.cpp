#include "iarap.hpp"

#include <iostream>
#include <fstream>

#include <igl/parallel_for.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/polar_svd3x3.h>

#include <Eigen/Sparse>

#include "segments.hpp"

typedef Eigen::SparseVector<double>::InnerIterator SVIter;


void initMesh(const Eigen::Matrix<double, -1, 3>& V, const Eigen::Matrix<int, -1, 3>& F, iARAPData& data){
  data.V = V;
  data.F = F;
  data.inputMesh.reset(new gcs::ManifoldSurfaceMesh(F));
  data.inputGeometry.reset(new gcs::VertexPositionGeometry(*data.inputMesh, V));
  data.intTri.reset(new gcs::SignpostIntrinsicTriangulation(*data.inputMesh, *data.inputGeometry));
  data.intTri->flipToDelaunay();
  data.intTri->requireEdgeCotanWeights();
  data.intTri->requireEdgeLengths();
}

void build_intrinsic_mesh(const Eigen::Matrix<double, -1, 3>& V, const Eigen::Matrix<int, -1, 3>& F, iARAPData& data){
  initMesh(/*In:*/V, F, /*Out:*/data);

  calc_segments(/*In:*/data.V, data.intTri, /*Out:*/data.all_segments);
  calc_segments_per_vertex(/*In:*/data.V, data.all_segments, /*Out:*/data.K, data.L, data.Bs, data.segments, data.weights);
}

void iarap_precompute(const Eigen::VectorXi& constraints_indices, iARAPData& data){
  igl::min_quad_with_fixed_precompute(/*In:*/data.L, constraints_indices, Eigen::SparseMatrix<double>(), false, /*Out:*/data.solver_data);
  
  data.RAll = Eigen::Matrix<double,-1, 3>::Zero(3*data.V.rows(),3);
  for (int i = 0; i < 3*data.V.rows(); i++) data.RAll(i,i%3) = 1.0;
}

void local_step(const Eigen::MatrixXd& U, iARAPData& data){
  igl::parallel_for(U.rows(), [&data, &U](const int i) {
    Eigen::Matrix3d SB = (data.Bs[i]*U).transpose() * data.weights[i].asDiagonal() * data.segments[i];
    // shift matrices closer to 0 to get higher resolution (necessary because polar_svd3x3 operates on floats)
    SB /= SB.array().abs().maxCoeff();
    Eigen::Matrix3d Rik;
    igl::polar_svd3x3(SB, Rik);
    Rik.transposeInPlace();
    data.RAll.block<3,3>(i*3,0) = Rik;
  }, 1000);
}

void global_step(const Eigen::MatrixXd& constraints, const iARAPData& data, Eigen::MatrixXd& U){
  Eigen::VectorXd Uc, bc;
  Eigen::MatrixXd b;
  b = data.K*data.RAll;
  igl::min_quad_with_fixed_solve(/*In:*/data.solver_data, b, constraints, Eigen::VectorXd(), /*Out:*/U);
}

void iarap_solve(const Eigen::MatrixXd& constraints, iARAPData& data, Eigen::MatrixXd& U){
  local_step(U, data);
  global_step(constraints, data, U);
}

double iarap_energy(const iARAPData& data, const Eigen::MatrixXd& U){
  double result = 0;
  int nVertices = U.rows();
  for (size_t i = 0; i < nVertices; i++) {
    auto newSegments = (data.Bs[i]*U);
    for (size_t j = 0; j < data.segments[i].rows(); j++) {
      result += data.weights[i](j)*(newSegments.row(j) - data.segments[i].row(j)*data.RAll.block<3,3>(i*3,0)).squaredNorm();
    }
  }
  return result;
}
