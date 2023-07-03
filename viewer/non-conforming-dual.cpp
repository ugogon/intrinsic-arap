#include "non-conforming-dual.hpp"

#include <igl/parallel_for.h>
#include <igl/adjacency_list.h>
#include <igl/polar_svd3x3.h>
#include <igl/cotmatrix.h>


void local_fits(const Eigen::MatrixXd& U, ARAPData& data){
  igl::parallel_for(U.rows(), [&data, &U](const int i) {
    Eigen::Matrix3d SB = (data.Adjacency[i]*U).transpose() * data.weights[i].asDiagonal() * data.edges[i];
    SB /= SB.array().abs().maxCoeff();
    Eigen::Matrix3d Rik;
    igl::polar_svd3x3(SB, Rik);
    Rik.transposeInPlace();
    data.RAll.block<3,3>(i*3,0) = Rik;
  }, 1000);
}

void precomputation(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, ARAPData& data) {
  int Vcount = V.rows();
  data.Adjacency.resize(Vcount);
  data.weights.resize(Vcount);
  data.edges.resize(Vcount);
  std::vector<std::vector<int>> A;
  data.RAll.resize(Vcount*3,3);
  igl::adjacency_list(F, A);
  Eigen::SparseMatrix<double> L;
  igl::cotmatrix(V, F, L);
  for (size_t i = 0; i < Vcount; i++) {
    int Ncount = A[i].size();
    data.Adjacency[i].resize(Ncount, Vcount);
    data.Adjacency[i].reserve(Eigen::VectorXi::Constant(Ncount,2));
    data.weights[i].resize(Ncount);
    data.edges[i].resize(Ncount, 3);
    for (size_t k = 0; k < Ncount; k++) {
      int j = A[i][k];
      data.Adjacency[i].insert(k,i) = -1;
      data.Adjacency[i].insert(k,j) = 1;
      data.weights[i](k) = L.coeff(i,j);
      data.edges[i].row(k) = V.row(j)-V.row(i);
    }
  }
}

Eigen::Vector3d circumcenter(Eigen::Vector3d a, Eigen::Vector3d b, Eigen::Vector3d c) {
  Eigen::Vector3d ac = c - a;
  Eigen::Vector3d ab = b - a;
  Eigen::Vector3d abXac = ab.cross(ac);

  Eigen::Vector3d toCircumsphereCenter = (abXac.cross( ab )*ac.transpose()*ac + ac.cross( abXac )*ab.transpose()*ab) / (2.f*abXac.transpose()*abXac);

  Eigen::Vector3d ccs = a + toCircumsphereCenter;
  return ccs;
}

void transformed_dual(const Eigen::MatrixXd& V, const Eigen::MatrixXd& U, const Eigen::MatrixXi& F, ARAPData& data, Eigen::MatrixXd& Vout, Eigen::MatrixXi& Fout) {
  Vout.resize(F.rows()*12,3);
  Fout.resize(F.rows()*6,3);
  local_fits(U, data);
  const size_t faces = F.rows();
  for (size_t n=0; n < faces; n++){
    const size_t i = F(n,0);
    const size_t j = F(n,1);
    const size_t k = F(n,2);
    Eigen::Vector3d ccs = circumcenter(Eigen::Vector3d(V.row(i)), Eigen::Vector3d(V.row(j)), Eigen::Vector3d(V.row(k)));
    Eigen::Vector3d mij = (V.row(i)+V.row(j))/2;
    Eigen::Vector3d mjk = (V.row(j)+V.row(k))/2;
    Eigen::Vector3d mki = (V.row(k)+V.row(i))/2;
    Vout.row(n*12+0) = U.row(i);
    Vout.row(n*12+1) = U.row(j);
    Vout.row(n*12+2) = U.row(k);
    Vout.row(n*12+3) = (U.row(j)+U.row(k))/2;
    Vout.row(n*12+4) = (U.row(k)+U.row(i))/2;
    Vout.row(n*12+5) = (U.row(i)+U.row(j))/2;
    Vout.row(n*12+6) = Vout.row(n*12+3)+(data.RAll.block<3,3>(j*3,0)*(ccs-mjk)).transpose();
    Vout.row(n*12+7) = Vout.row(n*12+3)+(data.RAll.block<3,3>(k*3,0)*(ccs-mjk)).transpose();
    Vout.row(n*12+8) = Vout.row(n*12+4)+(data.RAll.block<3,3>(k*3,0)*(ccs-mki)).transpose();
    Vout.row(n*12+9) = Vout.row(n*12+4)+(data.RAll.block<3,3>(i*3,0)*(ccs-mki)).transpose();
    Vout.row(n*12+10) = Vout.row(n*12+5)+(data.RAll.block<3,3>(i*3,0)*(ccs-mij)).transpose();
    Vout.row(n*12+11) = Vout.row(n*12+5)+(data.RAll.block<3,3>(j*3,0)*(ccs-mij)).transpose();
    Fout.row(n*6+0) << n*12+0, n*12+5, n*12+10;
    Fout.row(n*6+1) << n*12+5, n*12+1, n*12+11;
    Fout.row(n*6+2) << n*12+1, n*12+3, n*12+6;
    Fout.row(n*6+3) << n*12+3, n*12+2, n*12+7;
    Fout.row(n*6+4) << n*12+2, n*12+4, n*12+8;
    Fout.row(n*6+5) << n*12+4, n*12+0, n*12+9;
  }
}
