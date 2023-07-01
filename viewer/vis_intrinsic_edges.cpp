#include "vis_intrinsic_edges.hpp"

Eigen::VectorXd interpolate(const gcs::SurfacePoint& pt, const Eigen::MatrixXd& V){
  if (pt.type == gcs::SurfacePointType::Vertex){
    return V.row(pt.vertex.getIndex());
  } else {
    return V.row(pt.edge.firstVertex().getIndex())*(1-pt.tEdge) + V.row(pt.edge.secondVertex().getIndex())*pt.tEdge;
  }
}

void intrinsicEdges(const std::unique_ptr<gcs::IntrinsicTriangulation>& intTri, const Eigen::MatrixXd& V, std::vector<Eigen::VectorXd>& points, std::vector<std::array<int, 2>>& edges, Eigen::MatrixXd& P1, Eigen::MatrixXd& P2) {
  edges.clear();
  points.clear();
  intTri->flipToDelaunay();
  for(gcs::Edge e : intTri->intrinsicMesh->edges()) {
    std::vector<gcs::SurfacePoint> pointVec = intTri->traceIntrinsicHalfedgeAlongInput(e.halfedge());
    for (int k = 0; k < pointVec.size(); k++) {
      points.push_back(interpolate(pointVec[k], V));
      if (k == 0) continue;
      int t = points.size();
      std::array<int, 2> tmp{t-2, t-1};
      edges.push_back(tmp);
    }
  }
  P1.resize(edges.size(),3);
  P2.resize(edges.size(),3);
  int i = 0;
  for(auto e : edges){
    auto p1 = points[e[0]];
    auto p2 = points[e[1]];
    P1.row(i) << p1[0], p1[1], p1[2];
    P2.row(i) << p2[0], p2[1], p2[2];
    i++;
  }
}
