from __future__ import annotations

import unittest

from PIL import Image, ImageDraw

from risk_regions import detect_candidate_regions


class RiskRegionDetectorTest(unittest.TestCase):
    def test_flat_frame_does_not_trigger_ocr(self) -> None:
        image = Image.new("RGB", (640, 360), "gray")
        self.assertEqual([], detect_candidate_regions(image))

    def test_bottom_subtitle_is_not_a_risk_candidate(self) -> None:
        image = Image.new("RGB", (640, 360), "gray")
        draw = ImageDraw.Draw(image)
        for offset in range(10):
            left = 170 + offset * 26
            draw.rectangle((left, 280, left + 14, 310), fill="white")
            draw.rectangle((left + 4, 286, left + 10, 304), fill="black")
        regions = detect_candidate_regions(image)
        self.assertEqual([], regions)

    def test_central_scene_text_is_not_a_risk_candidate(self) -> None:
        image = Image.new("RGB", (640, 360), "gray")
        draw = ImageDraw.Draw(image)
        for offset in range(8):
            left = 210 + offset * 26
            draw.rectangle((left, 150, left + 14, 180), fill="white")
            draw.rectangle((left + 4, 156, left + 10, 174), fill="black")
        self.assertEqual([], detect_candidate_regions(image))

    def test_corner_overlay_becomes_prefilter_candidate(self) -> None:
        image = Image.new("RGB", (640, 360), "gray")
        draw = ImageDraw.Draw(image)
        for offset in range(5):
            left = 12 + offset * 20
            draw.rectangle((left, 15, left + 12, 39), fill="white")
            draw.rectangle((left + 3, 19, left + 9, 35), fill="black")
        regions = detect_candidate_regions(image)
        self.assertTrue(regions)
        self.assertTrue(any(region["reason"] == "overlay_text_geometry" for region in regions))


if __name__ == "__main__":
    unittest.main()
