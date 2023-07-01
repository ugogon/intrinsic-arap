#include "laplace.hpp"

#include "iarap.hpp"
#include "segments.hpp"

typedef Eigen::SparseVector<double>::InnerIterator SVIter;

Eigen::SparseMatrix<double> externalized_intrinsic_Laplace(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
  iARAPData data;
  initMesh(V, F, data);
  
  std::vector<Segment> all_segs;
  calc_segments(data.V, data.intTri, all_segs);
  int nSegments = all_segs.size();
  
  typedef Eigen::Triplet<double> T;
  std::vector<T> BarysList;
  // conservative estimation
  BarysList.reserve(nSegments*4);
  
  Eigen::VectorXd W;
  W.resize(nSegments);
  
  for (size_t i = 0; i < nSegments; i++) {
    W[i] = all_segs[i].weight;
    for (SVIter iter(all_segs[i].bary); iter; ++iter){
      BarysList.push_back(T(i, iter.index(), iter.value()));
    }
  }
  
  Eigen::SparseMatrix<double> Barys;
  Barys.resize(nSegments, V.rows());
  Barys.setZero();
  Barys.setFromTriplets(BarysList.begin(), BarysList.end());
  return Barys.transpose()*W.asDiagonal()*Barys;
}


Eigen::SparseMatrix<double> positive_externalized_intrinsic_Laplace(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
  iARAPData data;
  initMesh(V, F, data);
  
  const int nVertices = data.V.rows();
  typedef Eigen::Triplet<double> T;
  std::vector<T> LaplaceList;
  // conservative estimation
  const int nEdges = data.intTri->intrinsicMesh->nEdges();
  LaplaceList.reserve(nEdges*8);
  
  for(gcs::Edge e : data.intTri->intrinsicMesh->edges()) {
    const double weight = data.intTri->edgeCotanWeights[e];
    const double edge_length = data.intTri->edgeLengths[e];
    const int start = e.firstVertex().getIndex();
    const int end = e.secondVertex().getIndex();
    const std::vector<gcs::SurfacePoint> pointVec = data.intTri->traceIntrinsicHalfedgeAlongInput(e.halfedge());
    Eigen::SparseVector<double> second, second_to_last;
    
    second = b(pointVec[1], nVertices);
    Eigen::Vector3d first = (second.transpose()*V)-V.row(start);
    double w_first = (edge_length/first.norm())*weight;
    LaplaceList.push_back(T(start, start, -w_first));
    for (SVIter iter(second); iter; ++iter){
      LaplaceList.push_back(T(start, iter.index(), iter.value()*w_first));
    }   
    
    second_to_last = b(pointVec[pointVec.size()-2], nVertices);
    Eigen::Vector3d last = (second_to_last.transpose()*V)-V.row(end);
    double w_last = (edge_length/last.norm())*weight;  
    LaplaceList.push_back(T(end, end, -w_last));
    for (SVIter iter(second_to_last); iter; ++iter){
      LaplaceList.push_back(T(end, iter.index(), iter.value()*w_last));
    }
  }
  
  Eigen::SparseMatrix<double> L;
  L.resize(V.rows(), V.rows());
  L.setZero();
  L.setFromTriplets(LaplaceList.begin(), LaplaceList.end());
  return L;
}
