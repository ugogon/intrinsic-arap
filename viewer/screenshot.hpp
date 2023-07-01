#pragma once
#include <igl/opengl/glfw/Viewer.h>
#include <igl/png/writePNG.h>

inline void screenshot(igl::opengl::glfw::Viewer& viewer, std::string filename, int res) {
  int resx, resy;
  glfwGetWindowSize(viewer.window, &resx, &resy);
  resx *= res;
  resy *= res;
  // Allocate temporary buffers
  Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R(resx, resy);
  Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> G(resx, resy);
  Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> B(resx, resy);
  Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> A(resx, resy);

  // Draw the scene in the buffers
  viewer.core().draw_buffer(viewer.data(), false, R, G, B, A);

  // Save it to a PNG
  igl::png::writePNG(R, G, B, A, filename);
}
