// GLFW + mjr passive viewer.
//
// mujoco.viewer.launch_passive is Python-only, so the port supplies the
// equivalent through MuJoCo's C rendering API: same scene, same camera
// defaults, same overlay geometry. Interaction affordances differ (documented
// deviation 4 in docs/03-cpp-design.md).
//
// Overlay geometry is appended to the main mjvScene after mjv_updateScene,
// which is the C-API counterpart of the Python viewer's user_scn.

#pragma once

#include <mujoco/mujoco.h>

struct GLFWwindow;

namespace srl::render {

class Viewer {
 public:
  Viewer(mjModel* model, mjData* data, const char* title,
         bool vertical_sync = true);
  ~Viewer();

  Viewer(const Viewer&) = delete;
  Viewer& operator=(const Viewer&) = delete;

  bool IsRunning() const;
  void SetTitle(const char* title);

  // Refresh the scene from the current model/data, then reset the overlay
  // cursor so callers can append their own geoms.
  void UpdateScene();

  mjvScene* scene() { return &scene_; }

  // Draw and present the frame, and pump input events.
  void Render();

 private:
  static void MouseButtonCallback(GLFWwindow* window, int button, int action,
                                  int mods);
  static void CursorPositionCallback(GLFWwindow* window, double x, double y);
  static void ScrollCallback(GLFWwindow* window, double x_offset,
                             double y_offset);
  static void KeyCallback(GLFWwindow* window, int key, int scancode, int action,
                          int mods);

  mjModel* model_{nullptr};
  mjData* data_{nullptr};
  GLFWwindow* window_{nullptr};
  mjvScene scene_{};
  mjvCamera camera_{};
  mjvOption option_{};
  mjvPerturb perturb_{};
  mjrContext context_{};

  bool button_left_{false};
  bool button_middle_{false};
  bool button_right_{false};
  double last_x_{0.0};
  double last_y_{0.0};
};

}  // namespace srl::render
