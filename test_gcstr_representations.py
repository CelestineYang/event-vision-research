import unittest

import numpy as np

from gcstr_representations import GCSTRConfig, VIEW_NAMES, build_gcstr_views


class GCSTRRepresentationTests(unittest.TestCase):
    def setUp(self):
        self.config = GCSTRConfig(
            sensor_width=4, sensor_height=3, image_size=4, time_bins=4
        )
        self.t = np.array([0, 25, 50, 75, 100], dtype=np.uint64)
        self.x = np.array([0, 1, 2, 3, 3], dtype=np.uint16)
        self.y = np.array([0, 1, 2, 1, 2], dtype=np.uint16)
        self.p = np.array([False, True, False, True, False])

    def test_all_views_have_expected_shapes_and_ranges(self):
        views = build_gcstr_views(
            self.t, self.x, self.y, self.p, config=self.config
        )
        self.assertEqual(set(views), set(VIEW_NAMES))
        for image in views.values():
            self.assertEqual(image.shape, (3, 4, 4))
            self.assertEqual(image.dtype, np.float32)
            self.assertTrue(np.isfinite(image).all())
            self.assertGreaterEqual(float(image.min()), 0.0)
            self.assertLessEqual(float(image.max()), 1.0)

    def test_polarity_channels_and_zero_channel(self):
        views = build_gcstr_views(
            self.t, self.x, self.y, self.p,
            views=("xt_p", "ty_p"), config=self.config,
        )
        self.assertGreater(float(views["xt_p"][0].sum()), 0.0)
        self.assertGreater(float(views["xt_p"][1].sum()), 0.0)
        self.assertTrue(np.array_equal(views["xt_p"][2], np.zeros((4, 4))))
        self.assertTrue(np.array_equal(views["ty_p"][2], np.zeros((4, 4))))

    def test_cstr2_replaces_only_density_with_zero(self):
        views = build_gcstr_views(
            self.t, self.x, self.y, self.p,
            views=("cstr2", "cstr3"), config=self.config,
        )
        np.testing.assert_array_equal(views["cstr2"][0], views["cstr3"][0])
        np.testing.assert_array_equal(views["cstr2"][2], views["cstr3"][2])
        self.assertEqual(np.count_nonzero(views["cstr2"][1]), 0)
        self.assertGreater(np.count_nonzero(views["cstr3"][1]), 0)

    def test_cstr2_single_event_polarity_fallback(self):
        view = build_gcstr_views(
            np.array([7], dtype=np.uint64),
            np.array([1]),
            np.array([2]),
            np.array([True]),
            views=("cstr2",), config=self.config,
        )["cstr2"]
        np.testing.assert_array_equal(view[0], np.full((4, 4), 0.5))
        np.testing.assert_array_equal(view[1], np.zeros((4, 4)))
        np.testing.assert_array_equal(view[2], np.full((4, 4), 0.5))

    def test_last_timestamp_is_retained(self):
        view = build_gcstr_views(
            self.t, self.x, self.y, self.p,
            views=("xt_p",), config=self.config,
        )["xt_p"]
        self.assertGreater(float(view[1, 3, 3]), 0.0)

    def test_xt_variant_channels_match_density_and_mean_y_definitions(self):
        t = np.array([0, 10, 20, 30], dtype=np.uint64)
        x = np.array([0, 0, 1, 1], dtype=np.uint16)
        y = np.array([0, 2, 1, 2], dtype=np.uint16)
        p = np.array([True, True, False, False])
        views = build_gcstr_views(
            t, x, y, p,
            views=("xt_p", "xt_p_b", "xt_b_p", "xt_mean_p"),
            config=self.config,
        )

        # XT_p^b puts the binary density before the two polarized mean-y maps.
        np.testing.assert_array_equal(
            views["xt_p_b"][0], views["xt_p"].sum(axis=0)
        )
        np.testing.assert_array_equal(
            views["xt_b_p"][:2], views["xt_p"][:2]
        )
        np.testing.assert_array_equal(
            views["xt_p_b"][1:], views["xt_mean_p"][:2]
        )
        self.assertTrue(np.count_nonzero(views["xt_mean_p"][2]) == 0)

        # Polarized means use each polarity's own time span, not the full span.
        self.assertAlmostEqual(float(views["xt_mean_p"][0, 0, 3]), 1.0)
        self.assertAlmostEqual(float(views["xt_mean_p"][1, 1, 0]), 0.5)
        self.assertAlmostEqual(float(views["xt_mean_p"][1, 1, 3]), 1.0)
        # The unpolarized mean uses the full span: t=10 and t=20 map to bins 1,2.
        self.assertAlmostEqual(float(views["xt_b_p"][2, 0, 1]), 1.0)
        self.assertAlmostEqual(float(views["xt_b_p"][2, 1, 2]), 0.5)

    def test_xt_mean_y_single_or_missing_polarity_matches_official_fallback(self):
        views = build_gcstr_views(
            np.array([7], dtype=np.uint64),
            np.array([1]),
            np.array([2]),
            np.array([True]),
            views=("xt_p_b", "xt_b_p", "xt_mean_p"),
            config=self.config,
        )
        for image in views.values():
            self.assertTrue(np.isfinite(image).all())
        np.testing.assert_array_equal(views["xt_mean_p"][0], np.full((4, 4), 0.5))
        np.testing.assert_array_equal(views["xt_mean_p"][1], np.full((4, 4), 0.5))
        np.testing.assert_array_equal(views["xt_b_p"][2], np.full((4, 4), 0.5))

    def test_ty_variant_channels_match_density_and_mean_x_definitions(self):
        config = GCSTRConfig(
            sensor_width=4, sensor_height=4, image_size=4, time_bins=4
        )
        t = np.array([0, 10, 20, 30], dtype=np.uint64)
        x = np.array([0, 2, 1, 2], dtype=np.uint16)
        y = np.array([0, 0, 1, 1], dtype=np.uint16)
        p = np.array([True, True, False, False])
        views = build_gcstr_views(
            t, x, y, p,
            views=("ty_p", "ty_p_b", "ty_b_p", "ty_mean_p"),
            config=config,
        )

        np.testing.assert_array_equal(
            views["ty_p_b"][0], views["ty_p"].sum(axis=0)
        )
        np.testing.assert_array_equal(
            views["ty_b_p"][:2], views["ty_p"][:2]
        )
        np.testing.assert_array_equal(
            views["ty_p_b"][1:], views["ty_mean_p"][:2]
        )
        self.assertEqual(np.count_nonzero(views["ty_mean_p"][2]), 0)

        # Each polarized mean-x uses its own time range and maximum value.
        self.assertAlmostEqual(float(views["ty_mean_p"][0, 3, 0]), 1.0)
        self.assertAlmostEqual(float(views["ty_mean_p"][1, 0, 1]), 0.5)
        self.assertAlmostEqual(float(views["ty_mean_p"][1, 3, 1]), 1.0)
        # The binary mean-x uses the full event stream's time range.
        self.assertAlmostEqual(float(views["ty_b_p"][2, 1, 0]), 1.0)
        self.assertAlmostEqual(float(views["ty_b_p"][2, 2, 1]), 0.5)

    def test_ty_mean_x_single_or_missing_polarity_matches_official_fallback(self):
        views = build_gcstr_views(
            np.array([7], dtype=np.uint64),
            np.array([1]),
            np.array([2]),
            np.array([True]),
            views=("ty_p_b", "ty_b_p", "ty_mean_p"),
            config=self.config,
        )
        for image in views.values():
            self.assertTrue(np.isfinite(image).all())
        np.testing.assert_array_equal(views["ty_mean_p"][0], np.full((4, 4), 0.5))
        np.testing.assert_array_equal(views["ty_mean_p"][1], np.full((4, 4), 0.5))
        np.testing.assert_array_equal(views["ty_b_p"][2], np.full((4, 4), 0.5))

    def test_rejects_invalid_or_unsorted_events(self):
        with self.assertRaisesRegex(ValueError, "monotonic"):
            build_gcstr_views(
                self.t[::-1], self.x, self.y, self.p, config=self.config
            )
        bad_x = self.x.copy()
        bad_x[0] = 4
        with self.assertRaisesRegex(ValueError, "sensor"):
            build_gcstr_views(
                self.t, bad_x, self.y, self.p, config=self.config
            )


if __name__ == "__main__":
    unittest.main()
