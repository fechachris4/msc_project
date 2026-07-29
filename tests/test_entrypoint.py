import unittest


class EntrypointTest(unittest.TestCase):
    def test_import_does_not_start_the_viewer(self):
        import main

        self.assertTrue(callable(main.main))

    def test_argument_contract(self):
        import main

        self.assertEqual(main._parse_args([]), ("right", "left"))
        self.assertEqual(main._parse_args([], default_choice="left"), ("left",))
        self.assertEqual(
            main._parse_args(["right"], default_choice="left"),
            ("right",),
        )
        self.assertEqual(main._parse_args(["right"]), ("right",))
        with self.assertRaises(SystemExit):
            main._parse_args(["left", "tune"])
        with self.assertRaises(SystemExit):
            main._parse_args(["invalid"])

    def test_optional_live_trajectory_plot_flag(self):
        import main

        self.assertEqual(
            main._parse_options(["--trajectory-plot"]),
            (("right", "left"), True),
        )
        self.assertEqual(
            main._parse_options(
                ["left", "--trajectory-plot"],
                default_choice="right",
            ),
            (("left",), True),
        )
        self.assertEqual(
            main._parse_options([], default_choice="right"),
            (("right",), False),
        )
        with self.assertRaises(SystemExit):
            main._parse_options(
                ["--trajectory-plot", "--trajectory-plot"]
            )

    def test_different_arm_sources_are_composed_not_overwritten(self):
        import main
        from controller import desired_pos
        from controller.state import FramedTarget, Pose, TargetFrame, Twist
        from controller.trajectory import StaticTargetSource
        import numpy as np

        def source_at(position):
            return StaticTargetSource(
                FramedTarget(
                    TargetFrame.WORLD,
                    Pose(np.asarray(position, dtype=float), np.eye(3)),
                    Twist.zero(),
                )
            )

        right_plan = source_at((0.1, -0.2, 1.0))
        left_trajectory = source_at((0.6, 0.3, 1.2))
        composed = main._compose_target_source(
            desired_pos.configured_targets(),
            {"right": right_plan, "left": left_trajectory},
        )
        sampled = composed.sample(1.0)

        np.testing.assert_array_equal(
            sampled.right.pose.position_m,
            right_plan.target.pose.position_m,
        )
        np.testing.assert_array_equal(
            sampled.left.pose.position_m,
            left_trajectory.target.pose.position_m,
        )

    def test_scene_path_is_absolute(self):
        from sim import world

        self.assertTrue(world.SCENE_PATH.is_absolute())
        self.assertTrue(world.SCENE_PATH.is_file())


if __name__ == "__main__":
    unittest.main()
