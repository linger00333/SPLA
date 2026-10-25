"""Verify the primary gate cannot be opened by a secondary result."""
import unittest
from finalize_report import primary_gate, interval_text

def scores(p, p_lower, secondary):
    return {'runs': {'A0': {'SA': .8}, 'P': {'SA': .8+p}, 'S': {'SA': .8+secondary}},
            'vs_baseline': {'P-A0': {'SA': {'ci95': [p_lower, p+.01]}},
                            'S-A0': {'SA': {'ci95': [.004, .02]}}}}

class ReportGate(unittest.TestCase):
    def test_secondary_success_does_not_open_gate(self):
        self.assertFalse(primary_gate(scores(-.004, -.01, .01))['passed'])
    def test_positive_point_without_positive_interval_fails(self):
        self.assertFalse(primary_gate(scores(.004, -.001, .01))['passed'])
    def test_positive_interval_below_minimum_effect_fails(self):
        self.assertFalse(primary_gate(scores(.002, .001, .01))['passed'])
    def test_primary_both_conditions_pass(self):
        self.assertTrue(primary_gate(scores(.004, .001, 0))['passed'])
    def test_undefined_metric_is_not_formatted_as_zero(self):
        self.assertEqual(interval_text(None, None), 'null (undefined)')

if __name__ == '__main__': unittest.main()
