#!/usr/bin/env python3
"""Emit cpp/src/kinematics/LinkSpheres.h from controller/link_spheres.py.

Run from the Python project root with its venv:

    .venv/bin/python cpp/tools/generate_link_spheres.py > \
        cpp/src/kinematics/LinkSpheres.h

Generating rather than transcribing keeps the 17-significant-digit sphere
centres exactly equal between the two implementations.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controller.link_spheres import LINK_SPHERES  # noqa: E402


def literal(value):
    return format(float(value), ".17g")


def main():
    print("""// Conservative sphere chains derived from the Kinova collision meshes.
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
""")
    print(f"inline constexpr std::array<LinkSphere, {len(LINK_SPHERES)}> "
          "kLinkSpheres = {{")
    for sphere in LINK_SPHERES:
        centre = sphere.center_frame_m
        print(f'    {{"{sphere.frame_name}", "{sphere.name}",')
        print(f"     {{{literal(centre[0])}, {literal(centre[1])}, "
              f"{literal(centre[2])}}},")
        print(f"     {literal(sphere.radius_m)}, "
              f'{"true" if sphere.mount_exempt else "false"}}},')
    print("}};")
    print()
    print("inline constexpr std::size_t kLinkSphereCount = "
          "kLinkSpheres.size();")
    print()
    print("// Constraint capacity for the safety filter: non-exempt spheres.")
    print("inline constexpr std::size_t kMaxHumanConstraints = []{")
    print("  std::size_t count = 0;")
    print("  for (const auto& sphere : kLinkSpheres) {")
    print("    if (!sphere.mount_exempt) ++count;")
    print("  }")
    print("  return count;")
    print("}();")
    print()
    print("}  // namespace srl::kinematics")


if __name__ == "__main__":
    main()
