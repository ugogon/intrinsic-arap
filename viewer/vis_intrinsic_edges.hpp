#pragma once

#include "geometrycentral/surface/signpost_intrinsic_triangulation.h"

namespace gc = geometrycentral;
namespace gcs = gc::surface;

void intrinsicEdges(const std::unique_ptr<gcs::IntrinsicTriangulation>& intTri, const Eigen::MatrixXd& V, std::vector<Eigen::VectorXd>& points, std::vector<std::array<int, 2>>& edges, Eigen::MatrixXd& P1, Eigen::MatrixXd& P2);
