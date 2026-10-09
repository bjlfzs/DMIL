import math
import unittest

try:
    import torch
except ImportError:  # pragma: no cover - exercised in minimal environments
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DiagonalGaussianKLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from models.multimodal import diagonal_gaussian_kl

        cls.kl = staticmethod(diagonal_gaussian_kl)

    def test_identical_distributions_have_zero_kl(self):
        mean = torch.tensor([[1.0, -2.0]])
        log_var = torch.tensor([[math.log(2.0), math.log(0.5)]])
        actual = self.kl(mean, log_var, mean, log_var)
        self.assertTrue(torch.allclose(actual, torch.tensor(0.0), atol=1e-7))

    def test_standard_normal_closed_form(self):
        mean = torch.tensor([[1.0, 2.0]])
        log_var = torch.log(torch.tensor([[1.0, 4.0]]))
        expected_terms = 0.5 * (
            torch.exp(log_var) + mean.pow(2) - 1 - log_var
        )
        actual = self.kl(mean, log_var)
        self.assertTrue(torch.allclose(actual, expected_terms.mean(), atol=1e-7))

    def test_non_unit_reference_variance_scales_mean_term(self):
        q_mean = torch.tensor([[1.0]])
        q_log_var = torch.tensor([[math.log(2.0)]])
        p_mean = torch.tensor([[-1.0]])
        p_log_var = torch.tensor([[math.log(4.0)]])
        expected = 0.5 * (math.log(2.0) + 0.5)
        actual = self.kl(q_mean, q_log_var, p_mean, p_log_var)
        self.assertAlmostEqual(actual.item(), expected, places=6)


if __name__ == "__main__":
    unittest.main()
