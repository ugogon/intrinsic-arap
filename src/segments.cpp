#include "segments.hpp"

#include "iarap_data.hpp"


typedef Eigen::SparseVector<double>::InnerIterator SVIter;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator SVIterRow;


Eigen::SparseVector<double> b(const gcs::SurfacePoint& pt, const int size){
  Eigen::SparseVector<double> result;
  result.resize(size);
  if (pt.type == gcs::SurfacePointType::Vertex){
    result.insert(pt.vertex.getIndex()) = 1;
  } else {
    result.insert(pt.edge.firstVertex().getIndex()) = (1-pt.tEdge);
    result.insert(pt.edge.secondVertex().getIndex()) = pt.tEdge;
  }
  return result;
}

void calc_segments(
    const Eigen::Matrix<double, -1, 3>& V,
    const std::unique_ptr<gcs::IntrinsicTriangulation>& intTri,
    std::vector<Segment>& all_segments
){
  const int nEdges = intTri->intrinsicMesh->nEdges();
  // conservative estimation:
  all_segments.reserve(nEdges*2);
  
  const int nVertices = V.rows();
  int seg_cnt = 0;
  for(gcs::Edge e : intTri->intrinsicMesh->edges()) {
    const double weight = intTri->edgeCotanWeights[e];
    const double edge_length = intTri->edgeLengths[e];
    const int start = e.firstVertex().getIndex();
    const int end = e.secondVertex().getIndex();
    const std::vector<gcs::SurfacePoint> pointVec = intTri->traceIntrinsicHalfedgeAlongInput(e.halfedge());
    Eigen::SparseVector<double> current, last;
    last = b(pointVec[0], nVertices);
    for (int k = 1; k < pointVec.size(); k++) {
      current = b(pointVec[k], nVertices);
      Segment segment;
      segment.startV = start;
      segment.endV = end;
      segment.bary = current-last;
      segment.vector = segment.bary.transpose()*V;
      segment.length = segment.vector.norm();
      segment.weight = (edge_length/segment.length)*weight;
      all_segments.push_back(segment);
      last = current;
    }
  }
}


void calc_segments_per_vertex(
  const Eigen::Matrix<double, -1, 3>& V,
  const std::vector<Segment>& all_segs,
  Eigen::SparseMatrix<double>& K,
  Eigen::SparseMatrix<double>& L,
  std::vector<Eigen::SparseMatrix<double,Eigen::RowMajor>>& Bs,
  std::vector<Eigen::Matrix<double, -1, 3>>& segments,
  std::vector<Eigen::VectorXd>& weights
){
  const int nVertices = V.rows();
  Bs.resize(nVertices);
  segments.resize(nVertices);
  weights.resize(nVertices);
  
  // reorder segments
  int nSegments = all_segs.size();
  // get Bucket sizes per Vertex
  Eigen::VectorXi buckets = Eigen::VectorXi::Zero(nSegments);
  for (size_t i = 0; i < nSegments; i++) {
    buckets(all_segs[i].startV) += 1;
    buckets(all_segs[i].endV) += 1;
  }
  // Init weights, segments and barycentric weights
  for (size_t i = 0; i < nVertices; i++) {
    weights[i] = Eigen::VectorXd::Zero(buckets(i));
    segments[i] = Eigen::Matrix<double, -1, 3>::Zero(buckets(i),3);
    Bs[i].resize(buckets(i), nVertices);
  }
  
  Eigen::VectorXd W;
  W.resize(nSegments);
  
  typedef Eigen::Triplet<double> T;
  std::vector<T> BarysList;
  // conservative estimation
  BarysList.reserve(nSegments*4);
  
  std::vector<T> KList;
  // conservative estimation
  KList.reserve(nSegments*4*6);
  
  Eigen::VectorXi pos = Eigen::VectorXi::Zero(nSegments,buckets.maxCoeff());
  for (size_t i = 0; i < nSegments; i++) {
    const int u = all_segs[i].startV;
    const int v = all_segs[i].endV;
    
    weights[u](pos(u)) = all_segs[i].weight;
    weights[v](pos(v)) = all_segs[i].weight;
    
    segments[u].row(pos(u)) = all_segs[i].vector;
    segments[v].row(pos(v)) = -all_segs[i].vector;
    
    W[i] = all_segs[i].weight;
    const Eigen::Vector3d wseg = (1./2.)*all_segs[i].weight*all_segs[i].vector;
    for (SVIter iter(all_segs[i].bary); iter; ++iter){
      Bs[u].insert(pos(u), iter.index()) = iter.value();
      Bs[v].insert(pos(v), iter.index()) = -iter.value();
      
      BarysList.push_back(T(i,iter.index(), iter.value()));
      
      const auto bwseg = iter.value()*wseg;
      KList.push_back(T(iter.index(), u*3+0, bwseg(0)));
      KList.push_back(T(iter.index(), u*3+1, bwseg(1)));
      KList.push_back(T(iter.index(), u*3+2, bwseg(2)));
      KList.push_back(T(iter.index(), v*3+0, bwseg(0)));
      KList.push_back(T(iter.index(), v*3+1, bwseg(1)));
      KList.push_back(T(iter.index(), v*3+2, bwseg(2)));
    }
    pos(u) += 1;
    pos(v) += 1;
  }
  
  Eigen::SparseMatrix<double> Barys;
  Barys.resize(nSegments, nVertices);
  Barys.setZero();
  Barys.setFromTriplets(BarysList.begin(), BarysList.end());
  L = -Barys.transpose()*W.asDiagonal()*Barys;
  
  K.resize(nVertices, nVertices*3);
  K.setZero();
  K.setFromTriplets(KList.begin(), KList.end());
}
