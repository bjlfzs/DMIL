import unittest

from training_schedule import phase_for_epoch, validate_schedule


class TrainingScheduleTests(unittest.TestCase):
    def test_paper_schedule_boundaries(self):
        self.assertEqual(phase_for_epoch(1, 10, 5), 1)
        self.assertEqual(phase_for_epoch(10, 10, 5), 1)
        self.assertEqual(phase_for_epoch(11, 10, 5), 2)
        self.assertEqual(phase_for_epoch(15, 10, 5), 2)
        self.assertEqual(phase_for_epoch(16, 10, 5), 3)

    def test_zero_stage2_stops_after_stage1(self):
        self.assertEqual(phase_for_epoch(10, 10, 0), 1)
        self.assertIsNone(phase_for_epoch(11, 10, 0))

    def test_invalid_schedules_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_schedule(0, 5, 70)
        with self.assertRaises(ValueError):
            validate_schedule(10, -1, 70)
        with self.assertRaises(ValueError):
            validate_schedule(10, 5, 14)


if __name__ == "__main__":
    unittest.main()
