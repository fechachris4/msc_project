#include "render/Viewer.h"

#include <stdexcept>

#include <GLFW/glfw3.h>

namespace srl::render {
namespace {

constexpr int kMaxSceneGeoms = 4000;

}  // namespace

Viewer::Viewer(mjModel* model, mjData* data, const char* title)
    : model_(model), data_(data) {
  if (!glfwInit()) throw std::runtime_error("glfwInit failed");
  glfwWindowHint(GLFW_SAMPLES, 4);
  window_ = glfwCreateWindow(1200, 900, title, nullptr, nullptr);
  if (window_ == nullptr) {
    glfwTerminate();
    throw std::runtime_error("could not create a GLFW window");
  }
  glfwMakeContextCurrent(window_);
  glfwSwapInterval(1);

  mjv_defaultCamera(&camera_);
  mjv_defaultOption(&option_);
  mjv_defaultPerturb(&perturb_);
  mjv_defaultScene(&scene_);
  mjr_defaultContext(&context_);

  mjv_makeScene(model_, &scene_, kMaxSceneGeoms);
  mjr_makeContext(model_, &context_, mjFONTSCALE_150);

  // A default view that frames the torso and both arms.
  camera_.type = mjCAMERA_FREE;
  camera_.lookat[0] = 0.0;
  camera_.lookat[1] = 0.0;
  camera_.lookat[2] = 1.1;
  camera_.distance = 3.0;
  camera_.azimuth = 135.0;
  camera_.elevation = -15.0;

  glfwSetWindowUserPointer(window_, this);
  glfwSetMouseButtonCallback(window_, &Viewer::MouseButtonCallback);
  glfwSetCursorPosCallback(window_, &Viewer::CursorPositionCallback);
  glfwSetScrollCallback(window_, &Viewer::ScrollCallback);
  glfwSetKeyCallback(window_, &Viewer::KeyCallback);
}

Viewer::~Viewer() {
  mjr_freeContext(&context_);
  mjv_freeScene(&scene_);
  if (window_ != nullptr) glfwDestroyWindow(window_);
  glfwTerminate();
}

bool Viewer::IsRunning() const {
  return window_ != nullptr && !glfwWindowShouldClose(window_);
}

void Viewer::UpdateScene() {
  mjv_updateScene(model_, data_, &option_, &perturb_, &camera_, mjCAT_ALL,
                  &scene_);
}

void Viewer::Render() {
  mjrRect viewport{0, 0, 0, 0};
  glfwGetFramebufferSize(window_, &viewport.width, &viewport.height);
  mjr_render(viewport, &scene_, &context_);
  glfwSwapBuffers(window_);
  glfwPollEvents();
}

void Viewer::MouseButtonCallback(GLFWwindow* window, int, int, int) {
  auto* viewer = static_cast<Viewer*>(glfwGetWindowUserPointer(window));
  viewer->button_left_ =
      glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
  viewer->button_middle_ =
      glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
  viewer->button_right_ =
      glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
  glfwGetCursorPos(window, &viewer->last_x_, &viewer->last_y_);
}

void Viewer::CursorPositionCallback(GLFWwindow* window, double x, double y) {
  auto* viewer = static_cast<Viewer*>(glfwGetWindowUserPointer(window));
  if (!viewer->button_left_ && !viewer->button_middle_ &&
      !viewer->button_right_) {
    viewer->last_x_ = x;
    viewer->last_y_ = y;
    return;
  }
  int width = 0;
  int height = 0;
  glfwGetWindowSize(window, &width, &height);
  const double dx = x - viewer->last_x_;
  const double dy = y - viewer->last_y_;
  viewer->last_x_ = x;
  viewer->last_y_ = y;

  mjtMouse action = mjMOUSE_ZOOM;
  if (viewer->button_right_) {
    action = mjMOUSE_MOVE_V;
  } else if (viewer->button_left_) {
    action = mjMOUSE_ROTATE_V;
  }
  mjv_moveCamera(viewer->model_, action, dx / height, dy / height,
                 &viewer->scene_, &viewer->camera_);
}

void Viewer::ScrollCallback(GLFWwindow* window, double, double y_offset) {
  auto* viewer = static_cast<Viewer*>(glfwGetWindowUserPointer(window));
  mjv_moveCamera(viewer->model_, mjMOUSE_ZOOM, 0.0, -0.05 * y_offset,
                 &viewer->scene_, &viewer->camera_);
}

void Viewer::KeyCallback(GLFWwindow* window, int key, int, int action, int) {
  if (action == GLFW_PRESS && key == GLFW_KEY_ESCAPE) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  }
}

}  // namespace srl::render
