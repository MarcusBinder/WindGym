# tests/test_generate_layouts.py
import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch

# Import the functions to be tested
from WindGym.utils.generate_layouts import (
    generate_square_grid,
    generate_circle,
    generate_cirular_farm,
    generate_staggered_grid,
    plot_farm,
)

# Import a standard turbine for testing, as used in other tests
from py_wake.examples.data.hornsrev1 import V80


@pytest.fixture
def dummy_turbine():
    """Provides a standard V80 turbine instance for tests."""
    return V80()


class TestGenerateSquareGrid:
    """Tests for the generate_square_grid function."""

    def test_basic_grid(self, dummy_turbine):
        """Test a simple 2x2 square grid generation."""
        nx, ny = 2, 2
        x_dist, y_dist = 5, 5
        D = dummy_turbine.diameter()

        x_pos, y_pos = generate_square_grid(dummy_turbine, nx, ny, x_dist, y_dist)

        assert isinstance(x_pos, np.ndarray), "x_pos should be a numpy array"
        assert isinstance(y_pos, np.ndarray), "y_pos should be a numpy array"
        assert x_pos.shape == (
            nx * ny,
        ), f"Expected shape ({nx * ny},), but got {x_pos.shape}"
        assert y_pos.shape == (
            nx * ny,
        ), f"Expected shape ({nx * ny},), but got {y_pos.shape}"

        expected_x = np.array([0, D * x_dist, 0, D * x_dist])
        expected_y = np.array([0, 0, D * y_dist, D * y_dist])

        np.testing.assert_allclose(x_pos, expected_x)
        np.testing.assert_allclose(y_pos, expected_y)

    def test_single_turbine_grid(self, dummy_turbine):
        """Test the edge case of a 1x1 grid."""
        nx, ny = 1, 1
        x_dist, y_dist = 5, 5

        x_pos, y_pos = generate_square_grid(dummy_turbine, nx, ny, x_dist, y_dist)

        assert x_pos.shape == (1,), "A 1x1 grid should have one x-coordinate"
        assert y_pos.shape == (1,), "A 1x1 grid should have one y-coordinate"
        assert x_pos[0] == 0, "The single x-coordinate should be 0"
        assert y_pos[0] == 0, "The single y-coordinate should be 0"

    def test_linear_grid(self, dummy_turbine):
        """Test a 3x1 linear grid."""
        nx, ny = 3, 1
        x_dist, y_dist = 6, 4
        D = dummy_turbine.diameter()

        x_pos, y_pos = generate_square_grid(dummy_turbine, nx, ny, x_dist, y_dist)

        assert x_pos.shape == (3,), "A 3x1 grid should have three x-coordinates"
        assert y_pos.shape == (3,), "A 3x1 grid should have three y-coordinates"

        expected_x = np.array([0, D * x_dist, D * x_dist * 2])
        expected_y = np.array([0, 0, 0])

        np.testing.assert_allclose(x_pos, expected_x)
        np.testing.assert_allclose(y_pos, expected_y)


class TestGenerateCircle:
    """Tests for the generate_circle function."""

    def test_basic_circle(self):
        """Test generating 4 points on a circle of radius 10."""
        n, r = 4, 10
        x, y = generate_circle(n, r)

        assert x.shape == (n,), f"Expected {n} x-coordinates, got {x.shape[0]}"
        assert y.shape == (n,), f"Expected {n} y-coordinates, got {y.shape[0]}"

        # Check if points lie on the circle
        np.testing.assert_allclose(np.sqrt(x**2 + y**2), r)

        # Check coordinates for 4 points (0, 90, 180, 270 degrees)
        expected_x = np.array([10, 0, -10, 0])
        expected_y = np.array([0, 10, 0, -10])
        np.testing.assert_allclose(x, expected_x, atol=1e-10)
        np.testing.assert_allclose(y, expected_y, atol=1e-10)

    def test_circle_with_offset(self):
        """Test generating points with an angle offset."""
        n, r, offset = 4, 10, 45
        x, y = generate_circle(n, r, angle_offset=offset)

        # Check if points still lie on the circle
        np.testing.assert_allclose(np.sqrt(x**2 + y**2), r)

        # Check coordinates for 4 points with 45-degree offset
        val = 10 / np.sqrt(2)
        expected_x = np.array([val, -val, -val, val])
        expected_y = np.array([val, val, -val, -val])
        np.testing.assert_allclose(x, expected_x, atol=1e-10)
        np.testing.assert_allclose(y, expected_y, atol=1e-10)

    def test_single_point_circle(self):
        """Test the edge case of a circle with one point."""
        n, r = 1, 5
        x, y = generate_circle(n, r)

        assert x.shape == (1,)
        assert y.shape == (1,)
        assert x[0] == 5
        assert y[0] == 0


class TestGenerateCircularFarm:
    """Tests for the generate_cirular_farm function."""

    def test_simple_circular_farm(self, dummy_turbine):
        """Test a farm with two concentric circles."""
        n_list = [1, 6]  # One turbine at the center, 6 in a circle
        r_dist = 5
        D = dummy_turbine.diameter()
        x, y = generate_cirular_farm(n_list, dummy_turbine, r_dist)

        assert x.shape == (sum(n_list),), "Incorrect number of total turbines"
        assert y.shape == (sum(n_list),), "Incorrect number of total turbines"

        # Check center turbine
        assert x[0] == 0
        assert y[0] == 0

        # Check outer circle
        outer_x, outer_y = x[1:], y[1:]
        expected_radius = 1 * r_dist * D  # Second circle (index 1)
        np.testing.assert_allclose(np.sqrt(outer_x**2 + outer_y**2), expected_radius)

    def test_circular_farm_with_offsets(self, dummy_turbine):
        """Test a circular farm with angle offsets."""
        n_list = [4, 8]
        r_dist = 4
        angle_offsets = [0, 22.5]
        x, y = generate_cirular_farm(
            n_list, dummy_turbine, r_dist, angle_offset_list=angle_offsets
        )

        assert x.shape == (sum(n_list),)

        # Check first circle (no offset)
        first_circle_x, first_circle_y = x[:4], y[:4]
        np.testing.assert_allclose(first_circle_x[1], 0, atol=1e-10)

        # Check second circle (with offset)
        second_circle_x, second_circle_y = x[4:], y[4:]
        # A point should not be at (0, r) for the second circle
        assert not np.isclose(second_circle_x[2], 0)


class TestGenerateStaggeredGrid:
    """Tests for the generate_staggered_grid function."""

    def test_staggered_vs_square(self, dummy_turbine):
        """
        Test that with no offsets, generate_staggered_grid produces the same
        set of points as generate_square_grid, ignoring order.
        """
        nx, ny = 2, 2
        x_dist, y_dist = 5, 5
        x_sq, y_sq = generate_square_grid(dummy_turbine, nx, ny, x_dist, y_dist)
        x_st, y_st = generate_staggered_grid(dummy_turbine, nx, ny, x_dist, y_dist)

        # Sort coordinate pairs to make the comparison order-independent
        # A stable sort is needed, so we sort by y then by x.
        sq_coords = np.array(sorted(zip(x_sq, y_sq), key=lambda p: (p[0], p[1])))
        st_coords = np.array(sorted(zip(x_st, y_st), key=lambda p: (p[0], p[1])))

        np.testing.assert_array_equal(sq_coords, st_coords)

    def test_y_stagger_offset(self, dummy_turbine):
        """Test a grid with vertical staggering."""
        nx, ny = 2, 2
        x_dist, y_dist = 5, 5
        D = dummy_turbine.diameter()
        # FIX: Provide offset in diameters, not meters.
        y_stagger = [0, y_dist / 2]  # Stagger second column by half yDist

        x_pos, y_pos = generate_staggered_grid(
            dummy_turbine, nx, ny, x_dist, y_dist, y_stagger_offset=y_stagger
        )

        # The expected layout is:
        # Col 0: (0, 0), (0, 400)
        # Col 1: (400, 200), (400, 600)
        # The function generates column by column.
        expected_x = np.array([0, 0, D * x_dist, D * x_dist])
        expected_y = np.array(
            [
                0,  # (i=0, j=0) y=0*400+0*80 = 0
                D * y_dist,  # (i=0, j=1) y=1*400+0*80 = 400
                (y_dist / 2) * D,  # (i=1, j=0) y=0*400+2.5*80 = 200
                D * y_dist + (y_dist / 2) * D,  # (i=1, j=1) y=1*400+2.5*80 = 600
            ]
        )

        np.testing.assert_allclose(x_pos, expected_x)
        np.testing.assert_allclose(y_pos, expected_y)


class TestPlotFarm:
    """Tests for the plot_farm utility function."""

    @patch("matplotlib.pyplot.show")
    def test_plot_farm_runs_without_error(self, mock_show, dummy_turbine):
        """Check if plot_farm can be called without raising an exception."""
        x, y = generate_square_grid(dummy_turbine, 3, 3, 5, 5)

        try:
            # Test with turbine object
            plot_farm(x, y, turbine=dummy_turbine)
            plt.close()

            # Test with diameter D
            plot_farm(x, y, D=dummy_turbine.diameter())
            plt.close()

            # Test with no turbine or D
            plot_farm(x, y)
            plt.close()

        except Exception as e:
            pytest.fail(f"plot_farm raised an exception: {e}")

        # Assert that plt.show() was called, indicating the function ran to completion.
        # This is a basic check. We won't check the plot's content.
        assert mock_show.called, "plt.show() was not called by plot_farm"
