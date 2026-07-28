// Conservative sphere chains derived from the Kinova collision meshes.
//
// GENERATED from controller/link_spheres.py by tools/generate_link_spheres.py.
// Do not hand tune: rerun tools/derive_link_spheres.py on the Python side,
// then regenerate this header. The base, shoulder, and innermost upper-arm
// sphere are the intended wearer/robot mount interface; mount_exempt keeps
// that unavoidable overlap out of the avoidance constraints.

#pragma once

#include <array>
#include <string_view>

namespace srl::kinematics {

struct LinkSphere {
  std::string_view frame_name;
  std::string_view name;
  std::array<double, 3> center_frame_m;
  double radius_m;
  bool mount_exempt;
};

inline constexpr std::array<LinkSphere, 18> kLinkSpheres = {{
    {"base_link", "base_link_0",
     {0.0084971072214257842, -2.1802893685163001e-05, 0.036312064882628337},
     0.10047996262203035, true},
    {"base_link", "base_link_1",
     {-0.030127163827372329, -0.0024900667164666386, 0.1247565714346299},
     0.10047996262203035, true},
    {"shoulder_link", "shoulder_link_0",
     {-4.185197320968381e-05, -0.02121522635089907, -0.12934190890018354},
     0.076380412377460044, true},
    {"shoulder_link", "shoulder_link_1",
     {7.3352337945757741e-06, -0.0023756042250612699, -0.039541269074411245},
     0.076380412377460044, true},
    {"half_arm_1_link", "half_arm_1_link_0",
     {-4.5598245520185772e-06, -0.18365622251915714, -0.006949330633129731},
     0.068973402063425046, false},
    {"half_arm_1_link", "half_arm_1_link_1",
     {1.7148291342791447e-05, -0.091828731802096247, -0.017789462502748557},
     0.068973402063425046, false},
    {"half_arm_1_link", "half_arm_1_link_2",
     {3.8856407237601465e-05, -1.2410850353944403e-06, -0.028629594372367381},
     0.068973402063425046, true},
    {"half_arm_2_link", "half_arm_2_link_0",
     {-5.7020210132105909e-05, -0.018784624715124404, -0.21366163616926856},
     0.073981850229313861, false},
    {"half_arm_2_link", "half_arm_2_link_1",
     {0.00013318019806315992, -0.011992755578780125, -0.12770570351954591},
     0.073981850229313861, false},
    {"half_arm_2_link", "half_arm_2_link_2",
     {0.0003233806062584258, -0.0052008864424358484, -0.041749770869823219},
     0.073981850229313861, false},
    {"forearm_link", "forearm_link_0",
     {-3.5163269325690996e-05, -0.17941531638936109, -0.0090708071779034387},
     0.068777599801654016, false},
    {"forearm_link", "forearm_link_1",
     {-0.00011471094273642925, -0.089264947640208064, -0.019207405199549402},
     0.068777599801654016, false},
    {"forearm_link", "forearm_link_2",
     {-0.00019425861614716752, 0.00088542110894496151, -0.029344003221195365},
     0.068777599801654016, false},
    {"spherical_wrist_1_link", "spherical_wrist_1_link_0",
     {-0.00014923402205341372, -0.02483740728353772, -0.10366969047099109},
     0.063943500900099065, false},
    {"spherical_wrist_1_link", "spherical_wrist_1_link_1",
     {3.6482064777598603e-05, -0.0025287863499944083, -0.031175746637897711},
     0.063943500900099065, false},
    {"spherical_wrist_2_link", "spherical_wrist_2_link_0",
     {7.8481209720445257e-05, -0.085937330325062844, 0.00027936894079960868},
     0.060596027636264149, false},
    {"spherical_wrist_2_link", "spherical_wrist_2_link_1",
     {-0.00034408381231477807, -0.0069887874736374658, -0.026962071310574931},
     0.060596027636264149, false},
    {"bracelet_link", "bracelet_link_0",
     {0.0002207606478555784, -0.052302347390847569, -0.047660444828085541},
     0.10682368330155811, false},
}};

inline constexpr std::size_t kLinkSphereCount = kLinkSpheres.size();

// Constraint capacity for the safety filter: non-exempt spheres only.
inline constexpr std::size_t kMaxHumanConstraints = []{
  std::size_t count = 0;
  for (const auto& sphere : kLinkSpheres) {
    if (!sphere.mount_exempt) ++count;
  }
  return count;
}();

}  // namespace srl::kinematics
