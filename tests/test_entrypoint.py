import unittest


class EntrypointTest(unittest.TestCase):
    def test_import_does_not_start_the_viewer(self):
        import main

        self.assertTrue(callable(main.main))

    def test_argument_contract(self):
        import main

        self.assertEqual(main._parse_args([]), ("right", "left"))
        self.assertEqual(main._parse_args(["right"]), ("right",))
        with self.assertRaises(SystemExit):
            main._parse_args(["left", "tune"])
        with self.assertRaises(SystemExit):
            main._parse_args(["invalid"])

    def test_scene_path_is_absolute(self):
        from sim import world

        self.assertTrue(world.SCENE_PATH.is_absolute())
        self.assertTrue(world.SCENE_PATH.is_file())


if __name__ == "__main__":
    unittest.main()
