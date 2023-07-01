#define IGL_VIEWER_VIEWER_QUIET
// #define MIN_QUAD_WITH_FIXED_CPP_DEBUG
#include <igl/opengl/glfw/Viewer.h>
#include <igl/AABB.h>
#include <igl/screen_space_selection.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/SelectionWidget.h>
#include <igl/arap.h>
#include <igl/png/readPNG.h>

#include "iarap_data.hpp"
#include "iarap.hpp"
#include "non-conforming-dual.hpp"
#include "screenshot.hpp"
#include "vis_intrinsic_edges.hpp"


int main(int argc, char *argv[])
{
  if (argc<2) {
    std::cout << "Please pass an input mesh as first argument" << std::endl;
    exit(1);
  }

  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  igl::read_triangle_mesh(argv[1], V, F);

  Eigen::MatrixXd U = V;

  // settings
  int mode = 0;
  bool show_f = false;
  bool virgin = true;
  bool view_all_points = false;

  // non-conformal-dual
  Eigen::MatrixXd Vout;
  Eigen::MatrixXi Fout;
  ARAPData Adata;
  precomputation(V, F, Adata);
  transformed_dual(V, U, F, Adata, Vout, Fout);

  // iARAP
  iARAPData data;
  build_intrinsic_mesh(V, F, data);
  // OG ARAP
  igl::ARAPData arap_data;
  arap_data.max_iter = 1;
  arap_data.energy = igl::ARAP_ENERGY_TYPE_SPOKES;
  // S+R ARAP
  igl::ARAPData arap_data_rims;
  arap_data_rims.max_iter = 1;
  arap_data_rims.energy = igl::ARAP_ENERGY_TYPE_SPOKES_AND_RIMS;

  // Viewer
  igl::opengl::glfw::Viewer viewer;
  igl::opengl::glfw::imgui::ImGuiPlugin plugin;

  Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R,G,B,A;
  igl::png::readPNG(argc > 2 ? argv[2] : "../textures/texture.png", R,G,B,A);

  Eigen::MatrixXd CM = (Eigen::MatrixXd(9,3)<<
      //0.8,0.8,0.8,                        // 0 gray
      1,1,1,
      1.0/255.0,31.0/255.0,255.0/255.0,   // 1 blue
      0/255.0,201.0/255.0,195.0/255.0,    // 2 cyan
      7.0/255.0,152.0/255.0,59.0/255.0,   // 3 green
      190.0/255.0,255.0/255.0,0.0/255.0,  // 4 lime
      255.0/255.0,243.0/255.0,0.0/255.0,  // 5 yellow
      245.0/255.0,152.0/255.0,0.0/255.0,  // 6 orange
      245.0/255.0,152.0/255.0,0.0/255.0,  // 7 orange
      // 224.0/255.0,18.0/255.0,0.0/255.0,   // 7 red
      220.0/255.0,13.0/255.0,255.0/255.0  // 8 magenta
    ).finished();

  auto init_viewer_data = [&](){
    viewer.data().set_face_based(true);
    viewer.data().set_texture(R,G,B,A);
    viewer.data().show_texture = true;
    viewer.data().use_matcap = true;
    viewer.data().point_size = 15;
    viewer.data().line_width = 1;
    viewer.data().show_lines = F.rows() < 20000;
  };

  // Selection
  bool translate = false;
  bool only_visible = false;
  igl::opengl::glfw::imgui::SelectionWidget selection;
  Eigen::Array<double,Eigen::Dynamic,1> W = Eigen::Array<double,Eigen::Dynamic,1>::Zero(V.rows());
  Eigen::Array<double,Eigen::Dynamic,1> and_visible = Eigen::Array<double,Eigen::Dynamic,1>::Zero(V.rows());

  Eigen::MatrixXd all_constraints = V;

  Eigen::Array<int,-1,1> constraints_mask = Eigen::Array<int,-1,1>::Zero(V.rows());
  Eigen::Array<int,-1,1> selected_mask = Eigen::Array<int,-1,1>::Zero(V.rows());
  Eigen::VectorXi selected_ids;

  Eigen::MatrixXd constraints;
  Eigen::VectorXi bi;
  Eigen::VectorXi indices = Eigen::VectorXd::LinSpaced(V.rows(), 0.0, V.rows() - 1).cast<int>();


  igl::AABB<Eigen::MatrixXd, 3> tree;
  tree.init(V,F);

  const auto update_points = [&]()
  {
    // constrained
    viewer.data().set_points(igl::slice_mask(all_constraints, (selected_mask * constraints_mask).cast<bool>(), 1), CM.row(6));
    viewer.data().add_points(igl::slice_mask(all_constraints, (selected_mask * (1-constraints_mask)).cast<bool>(), 1), CM.row(5));
    viewer.data().add_points(igl::slice_mask(all_constraints, ((1-selected_mask) * constraints_mask).cast<bool>(), 1), CM.row(7));
    if (view_all_points) {
      viewer.data().add_points(igl::slice_mask(all_constraints, ((1-selected_mask) * (1-constraints_mask)).cast<bool>(), 1), CM.row(2));
    }
    viewer.draw();viewer.draw();
  };

  auto refresh_mesh = [&](){
    viewer.data().clear();
    if (show_f) {
      viewer.data().set_mesh(Vout, Fout);
    } else{
      viewer.data().set_mesh(U,F);
    }
    init_viewer_data();
    update_points();
  };

  selection.callback = [&]()
  {
    screen_space_selection(all_constraints, F, tree, viewer.core().view, viewer.core().proj, viewer.core().viewport, selection.L, W, and_visible);
    if (only_visible){
      W = (W * and_visible);
    }
    selected_mask = (W.abs()>0.5).cast<int>();
    selected_ids = igl::slice_mask(indices, selected_mask.cast<bool>(), 1);
    selection.mode=selection.OFF;
    update_points();
  };

  auto update_constraints = [&](){
    if (virgin) return;
    bi = igl::slice_mask(indices, constraints_mask.cast<bool>(), 1);
    constraints = igl::slice(all_constraints, bi, 1);
    // does all precomputations at once
    igl::parallel_for(3, [&](const int i) {
      switch (i) {
        case 0:
          igl::arap_precomputation(V,F,V.cols(),bi,arap_data);
        case 1:
          igl::arap_precomputation(V,F,V.cols(),bi,arap_data_rims);
        case 2:
          iarap_precompute(bi, data);
      }
    }, 1);
    update_points();
  };

  viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
  {
    translate = false;
    selection.mode=selection.OFF;
    update_points();
    return false;
  };

  Eigen::RowVector3f last_mouse;
  viewer.callback_mouse_move = [&](igl::opengl::glfw::Viewer& viewer, int x, int y)
  {
    if (!translate) return false;
    Eigen::RowVector3f drag_mouse(viewer.current_mouse_x, viewer.core().viewport(3) - viewer.current_mouse_y, last_mouse(2));
    Eigen::RowVector3f drag_scene,last_scene;
    igl::unproject(drag_mouse, viewer.core().view, viewer.core().proj, viewer.core().viewport, drag_scene);
    igl::unproject(last_mouse, viewer.core().view, viewer.core().proj, viewer.core().viewport, last_scene);
    for (size_t i = 0; i < selected_ids.size(); i++) {
      all_constraints.row(selected_ids(i)) += (drag_scene-last_scene).cast<double>();
    }
    constraints = igl::slice(all_constraints, bi, 1);
    last_mouse = drag_mouse;
    update_points();
    return false;
  };



  viewer.callback_key_pressed = [&](decltype(viewer) &,unsigned int key, int mod)
  {
    switch(key)
    {
      case ' ':
        if (virgin) return true;
        if (mode == 0){
          iarap_solve(constraints, data, U);
        } else if(mode == 1) {
          igl::arap_solve(constraints, arap_data, U);
        } else if (mode == 2) {
          igl::arap_solve(constraints, arap_data_rims, U);
        }
        if (show_f) {
          transformed_dual(V, U, F, Adata, Vout, Fout);
          viewer.data().set_mesh(Vout, Fout);
          return true;
        }
        viewer.data().set_mesh(U,F);
        return true;
      case '1':  mode = 0; return true;
      case '2':  mode = 1; return true;
      case '3':  mode = 2; return true;
      case 'H': case 'h': only_visible = !only_visible; update_points(); return true;
      case 'C': case 'c':
        virgin = false;
        constraints_mask = ((constraints_mask+selected_mask)>0).cast<int>();
        update_constraints();
        // std::cout << constraints << "\n\n\n";
        // std::cout << bi << "\n\n\n";
        return true;
      case 'U': case 'u':
        constraints_mask = ((constraints_mask-selected_mask)==1).cast<int>();
        if (constraints_mask.sum() == 0){
          virgin = true;
          return true;
        }
        update_constraints();
        return true;
      case 'G': case 'g':
        translate = !translate;
        {
          Eigen::MatrixXd CP;
          Eigen::Vector3d mean = constraints.rowwise().mean();
          igl::project(mean, viewer.core().view, viewer.core().proj, viewer.core().viewport, CP);
          last_mouse = Eigen::RowVector3f(viewer.current_mouse_x, viewer.core().viewport(3) - viewer.current_mouse_y, CP(0,2));
        }
        return true;
      case 'K': case 'k':
        selection.mode = selection.RECTANGULAR_MARQUEE;
        selection.clear();
        return true;
      case 'S': case 's':
        {
          auto t = std::time(nullptr);
          auto tm = *std::localtime(&t);
          std::ostringstream oss;
          oss << std::put_time(&tm, "%d-%m-%Y %H-%M-%S");
          screenshot(viewer, oss.str()+".png", 1);
        }
        return true;
      case 'T': case 't':
        if (mode == 0)
        {
          viewer.data().set_texture(R,G,B,A);
        } else {
          Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R,G,B,A;
          igl::png::readPNG(argc > 3 ? argv[3] : "../textures/texture.png", R,G,B,A);
          viewer.data().set_texture(R,G,B,A);
        }
        return true;
      case 'R': case 'r':
        // reset mesh
        U = V; transformed_dual(V, U, F, Adata, Vout, Fout); refresh_mesh(); return true;
      case 'F': case 'f':
        // filter
        selected_mask = selected_mask*constraints_mask;
        selected_ids = igl::slice_mask(indices, selected_mask.cast<bool>(), 1);
        update_points();
        return true;
      case 'P': case 'p':
        view_all_points = !view_all_points;
        update_points();
        return true;
      case 'X': case 'x':
        // reset constrainted positions of selected
        for (size_t i = 0; i < selected_mask.size(); i++) {
          if (selected_mask(i))
            all_constraints.row(i) = V.row(i);
        }
        update_constraints();
        return true;
      case 'E': case 'e':
        show_f = !show_f;
        refresh_mesh();
        return true;
      case 'Q': case 'q':
        std::vector<Eigen::VectorXd> points;
        std::vector<std::array<int, 2>> edges;
        Eigen::MatrixXd P1, P2;
        P1.resize(0,3);
        P2.resize(0,3);
        viewer.data().set_edges(P1, P2.cast<int>(), CM.row(4));
        intrinsicEdges(data.intTri, U, points, edges, P1, P2);
        viewer.data().add_edges(P1, P2, CM.row(4));
        return true;
    }
    return false;
  };

  viewer.plugins.push_back(&plugin);
  plugin.widgets.push_back((igl::opengl::glfw::imgui::ImGuiWidget*) &selection);
  selection.mode = selection.OFF;

  std::cout<<R"( igl::opengl::glfw::Viewer usage:
  I,i     Toggle invert normals
  L       Toggle wireframe
  O,o     Toggle orthographic/perspective projection
  Z       Snap to canonical view
  [,]     Toggle between rotation control types (trackball, two-axis
          valuator with fixed up, 2D mode with no rotation))
  <,>     Toggle between models
  ;       Toggle vertex labels
  :       Toggle face labels

ARAP Usage:
  [space]  Run one ARAP Step (hold for animation)
  1,2,3    Change ARAP Mode 1="iARAP", 2="OG ARAP", 3="Spokes+Rims ARAP"
  R,r      Reset mesh position

  H,h      Toggle whether to take visibility into account for selection
  F,f      Remove non constrained points from selection

  C,c      Add selection to constraints
  U,u      Remove selection from constraints

  X,x      Reset constrain position of selected vertices
  G,g      Move selection
  P,p      View unconstrained point positions

  S,s      Make screenshot
)";

  viewer.data().clear();
  viewer.data().set_mesh(V,F);
  init_viewer_data();
  viewer.core().is_animating = true;
  viewer.core().background_color.head(3) = CM.row(0).head(3).cast<float>();
  // update_points()
  viewer.launch();
}
