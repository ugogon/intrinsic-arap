#pragma once

#include <Eigen/Core>
#include <vector>
#include "geometrycentral/surface/signpost_intrinsic_triangulation.h"
#include <igl/min_quad_with_fixed.h>

namespace gc = geometrycentral;
namespace gcs = gc::surface;

struct Segment {
  int startV;
  int endV;
  Eigen::SparseVector<double> bary;
  double weight;
  double length;
  Eigen::Vector3d vector;
};

struct iARAPData {
  // input
  Eigen::Matrix<double, -1, 3> V;
  Eigen::Matrix<int, -1, 3> F;
  
  // intrinsic Triangulation
  std::unique_ptr<gcs::IntrinsicTriangulation> intTri;
  std::unique_ptr<gcs::ManifoldSurfaceMesh> inputMesh;
  std::unique_ptr<gcs::VertexPositionGeometry> inputGeometry;

  // all Segments
  std::vector<Segment> all_segments;

  // per vertex data
  Eigen::MatrixXd RAll; // ARAP rotations
  
  // per Vertex -> per Segment data
  std::vector<Eigen::SparseMatrix<double,Eigen::RowMajor>> Bs; // barycentric weights for subvertices of segments per vertex
  std::vector<Eigen::Matrix<double, -1, 3>> segments; // list of all segment vectors per vertex
  std::vector<Eigen::VectorXd> weights; // weight list of all segments per vertex

  // solver stuff
  Eigen::SparseMatrix<double> L; // Laplace
  igl::min_quad_with_fixed_data<double> solver_data;
  Eigen::SparseMatrix<double> K; // precomputed rhs of LSE such that K*RAll = b;
};
