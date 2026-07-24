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

    def test_scene_path_is_absolute(self):
        from sim import world

        self.assertTrue(world.SCENE_PATH.is_absolute())
        self.assertTrue(world.SCENE_PATH.is_file())


if __name__ == "__main__":
    unittest.main()
