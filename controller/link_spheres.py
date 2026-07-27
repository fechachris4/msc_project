"""Conservative sphere chains derived from the Kinova collision meshes.

The values below are generated from the collision geoms in
``sim/assets/kinova_gen3/gen3.xml``.  For each mesh, principal-axis bins are
covered by spheres whose radius includes both the maximum perpendicular mesh
extent and half of the bin length.  Therefore every collision-mesh vertex is
inside at least one sphere.

The base, shoulder, and innermost upper-arm sphere are the intentional
wearer/robot mount interface.  They remain represented for diagnostics and
mesh-coverage validation, but ``mount_exempt`` keeps that unavoidable
attachment overlap out of the avoidance constraints.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LinkSphere:
    frame_name: str
    name: str
    center_frame_m: tuple[float, float, float]
    radius_m: float
    mount_exempt: bool


# Generated from the contype-enabled MuJoCo meshes.  Do not hand tune these
# values: rerun tools/derive_link_spheres.py and re-run the mesh coverage test.
LINK_SPHERES = (
    # base_link: mesh vertices=1867, chain radius=0.10047996262203035 m
    LinkSphere(
        "base_link",
        "base_link_0",
        (0.008497107221425784, -2.1802893685163e-05, 0.03631206488262834),
        0.10047996262203035,
        True,
    ),
    LinkSphere(
        "base_link",
        "base_link_1",
        (-0.03012716382737233, -0.0024900667164666386, 0.1247565714346299),
        0.10047996262203035,
        True,
    ),
    # shoulder_link: mesh vertices=4793, chain radius=0.07638041237746004 m
    LinkSphere(
        "shoulder_link",
        "shoulder_link_0",
        (-4.185197320968381e-05, -0.02121522635089907, -0.12934190890018354),
        0.07638041237746004,
        True,
    ),
    LinkSphere(
        "shoulder_link",
        "shoulder_link_1",
        (7.335233794575774e-06, -0.00237560422506127, -0.039541269074411245),
        0.07638041237746004,
        True,
    ),
    # half_arm_1_link: mesh vertices=5156, chain radius=0.06897340206342505 m
    LinkSphere(
        "half_arm_1_link",
        "half_arm_1_link_0",
        (-4.559824552018577e-06, -0.18365622251915714, -0.006949330633129731),
        0.06897340206342505,
        False,
    ),
    LinkSphere(
        "half_arm_1_link",
        "half_arm_1_link_1",
        (1.7148291342791447e-05, -0.09182873180209625, -0.017789462502748557),
        0.06897340206342505,
        False,
    ),
    LinkSphere(
        "half_arm_1_link",
        "half_arm_1_link_2",
        (3.8856407237601465e-05, -1.2410850353944403e-06, -0.02862959437236738),
        0.06897340206342505,
        True,
    ),
    # half_arm_2_link: mesh vertices=4931, chain radius=0.07398185022931386 m
    LinkSphere(
        "half_arm_2_link",
        "half_arm_2_link_0",
        (-5.702021013210591e-05, -0.018784624715124404, -0.21366163616926856),
        0.07398185022931386,
        False,
    ),
    LinkSphere(
        "half_arm_2_link",
        "half_arm_2_link_1",
        (0.00013318019806315992, -0.011992755578780125, -0.1277057035195459),
        0.07398185022931386,
        False,
    ),
    LinkSphere(
        "half_arm_2_link",
        "half_arm_2_link_2",
        (0.0003233806062584258, -0.005200886442435848, -0.04174977086982322),
        0.07398185022931386,
        False,
    ),
    # forearm_link: mesh vertices=4593, chain radius=0.06877759980165402 m
    LinkSphere(
        "forearm_link",
        "forearm_link_0",
        (-3.5163269325690996e-05, -0.1794153163893611, -0.009070807177903439),
        0.06877759980165402,
        False,
    ),
    LinkSphere(
        "forearm_link",
        "forearm_link_1",
        (-0.00011471094273642925, -0.08926494764020806, -0.019207405199549402),
        0.06877759980165402,
        False,
    ),
    LinkSphere(
        "forearm_link",
        "forearm_link_2",
        (-0.00019425861614716752, 0.0008854211089449615, -0.029344003221195365),
        0.06877759980165402,
        False,
    ),
    # spherical_wrist_1_link: vertices=5494, radius=0.06394350090009906 m
    LinkSphere(
        "spherical_wrist_1_link",
        "spherical_wrist_1_link_0",
        (-0.00014923402205341372, -0.02483740728353772, -0.10366969047099109),
        0.06394350090009906,
        False,
    ),
    LinkSphere(
        "spherical_wrist_1_link",
        "spherical_wrist_1_link_1",
        (3.64820647775986e-05, -0.0025287863499944083, -0.03117574663789771),
        0.06394350090009906,
        False,
    ),
    # spherical_wrist_2_link: vertices=5173, radius=0.06059602763626415 m
    LinkSphere(
        "spherical_wrist_2_link",
        "spherical_wrist_2_link_0",
        (7.848120972044526e-05, -0.08593733032506284, 0.0002793689407996087),
        0.06059602763626415,
        False,
    ),
    LinkSphere(
        "spherical_wrist_2_link",
        "spherical_wrist_2_link_1",
        (-0.00034408381231477807, -0.006988787473637466, -0.02696207131057493),
        0.06059602763626415,
        False,
    ),
    # bracelet_link: mesh vertices=11637, chain radius=0.10682368330155811 m
    LinkSphere(
        "bracelet_link",
        "bracelet_link_0",
        (0.0002207606478555784, -0.05230234739084757, -0.04766044482808554),
        0.10682368330155811,
        False,
    ),
)
