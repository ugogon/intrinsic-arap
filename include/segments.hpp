#pragma once

#include "iarap_data.hpp"

Eigen::SparseVector<double> b(const gcs::SurfacePoint& pt, const int size);

void calc_segments(
		const Eigen::Matrix<double, -1, 3>& V,
		const std::unique_ptr<gcs::IntrinsicTriangulation>& intTri,
		std::vector<Segment>& all_segments
);

void calc_segments_per_vertex(
	const Eigen::Matrix<double, -1, 3>& V,
	const std::vector<Segment>& all_segs,
	Eigen::SparseMatrix<double>& K,
	Eigen::SparseMatrix<double>& L,
	std::vector<Eigen::SparseMatrix<double,Eigen::RowMajor>>& Bs,
	std::vector<Eigen::Matrix<double, -1, 3>>& segments,
	std::vector<Eigen::VectorXd>& weights
);