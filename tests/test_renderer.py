"""
Comprehensive test suite for WindFarmRenderer.

This test file covers all rendering scenarios including:
- Different render modes (rgb_array, human, None)
- Error handling and edge cases
- Probe rendering (WS, TI, and unknown probe types)
- Farm and frame plotting methods
- Initialization scenarios
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from WindGym.core.renderer import WindFarmRenderer
from WindGym.core.wind_probe import WindProbe
from WindGym import WindFarmEnv
from WindGym.utils.generate_layouts import generate_square_grid
from py_wake.examples.data.hornsrev1 import V80
from dynamiks.sites.turbulence_fields import MannTurbulenceField


@pytest.fixture
def turbine():
    """Provides a standard wind turbine configuration for testing"""
    return V80()


@pytest.fixture
def mock_fs():
    """
    Creates a mock flow simulation (fs) object.
    Configured to return predictable wind speed and turbulence values.
    """
    fs = MagicMock()

    # Mock windTurbines with positions
    fs.windTurbines = MagicMock()
    fs.windTurbines.positions_xyz = np.array([[0, 500, 1000], [0, 0, 0], [90, 90, 90]])
    fs.windTurbines.types = [0, 0, 0]
    fs.windTurbines.yaw = np.array([0, 0, 0])

    # Mock yaw_tilt method
    fs.windTurbines.yaw_tilt = MagicMock(return_value=(np.array([0, 0, 0]), np.array([0, 0, 0])))

    # Mock wind_direction and time
    fs.wind_direction = 270
    fs.time = 42.5

    # Mock get_windspeed to return an xarray-like object
    mock_uvw = MagicMock()
    mock_uvw.x.values = np.linspace(0, 100, 10)
    mock_uvw.y.values = np.linspace(0, 100, 10)
    mock_uvw.__getitem__ = MagicMock(return_value=MagicMock(T=np.random.rand(10, 10) * 10 + 5))
    fs.get_windspeed = MagicMock(return_value=mock_uvw)

    # Mock get_turbulence_intensity
    fs.get_turbulence_intensity = MagicMock(return_value=0.15)

    return fs


@pytest.fixture
def mock_turbine():
    """Creates a mock turbine object"""
    turbine = MagicMock()
    turbine.hub_height = MagicMock(return_value=90)
    return turbine


@pytest.fixture
def mock_probes_ws_ti_unknown(mock_fs):
    """Creates a list of mock probes with different types: WS, TI, and unknown"""
    probes = []

    # WS probe
    ws_probe = MagicMock(spec=WindProbe)
    ws_probe.position = (100, 50, 90)
    ws_probe.probe_type = "ws"
    ws_probe.read = MagicMock(return_value=8.5)
    ws_probe.get_inflow_angle_to_turbine = MagicMock(return_value=np.pi / 4)
    probes.append(ws_probe)

    # TI probe
    ti_probe = MagicMock(spec=WindProbe)
    ti_probe.position = (200, 100, 90)
    ti_probe.probe_type = "ti"
    ti_probe.read = MagicMock(return_value=0.12)
    ti_probe.get_inflow_angle_to_turbine = MagicMock(return_value=np.pi / 6)
    probes.append(ti_probe)

    # Unknown probe type
    unknown_probe = MagicMock(spec=WindProbe)
    unknown_probe.position = (300, 150, 90)
    unknown_probe.probe_type = "unknown"
    unknown_probe.read = MagicMock(return_value=999)
    unknown_probe.get_inflow_angle_to_turbine = MagicMock(return_value=0)
    probes.append(unknown_probe)

    return probes


class TestRendererInitialization:
    """Test renderer initialization and basic setup"""

    def test_init_none_render_mode(self):
        """Test initialization with None render_mode"""
        renderer = WindFarmRenderer(render_mode=None)
        assert renderer.render_mode is None
        assert renderer.figure is None
        assert renderer.ax is None
        assert renderer.view is None

    def test_init_rgb_array_mode(self):
        """Test initialization with rgb_array render_mode"""
        renderer = WindFarmRenderer(render_mode="rgb_array")
        assert renderer.render_mode == "rgb_array"

    def test_init_human_mode(self):
        """Test initialization with human render_mode"""
        renderer = WindFarmRenderer(render_mode="human")
        assert renderer.render_mode == "human"

    def test_init_render(self, mock_fs, mock_turbine):
        """Test init_render method"""
        renderer = WindFarmRenderer(render_mode="rgb_array")
        renderer.init_render(mock_fs, mock_turbine)

        assert renderer.view is not None
        assert renderer.a is not None
        assert renderer.b is not None
        assert renderer.figure is not None
        assert renderer.ax is not None


class TestRenderMethod:
    """Test the main render() method routing"""

    def test_render_none_mode(self, mock_fs):
        """Test render with None mode returns None"""
        renderer = WindFarmRenderer(render_mode=None)
        result = renderer.render(mock_fs)
        assert result is None

    def test_render_rgb_array_mode(self, mock_fs, mock_turbine):
        """Test render with rgb_array mode returns frame"""
        renderer = WindFarmRenderer(render_mode="rgb_array")
        renderer.init_render(mock_fs, mock_turbine)

        with patch.object(renderer, '_render_frame', return_value=np.zeros((100, 100, 3))):
            result = renderer.render(mock_fs)
            assert result is not None
            assert isinstance(result, np.ndarray)

    def test_render_human_mode_not_initialized(self, mock_fs):
        """Test line 87: render with human mode raises RuntimeError when view is None"""
        renderer = WindFarmRenderer(render_mode="human")

        with pytest.raises(RuntimeError, match="Renderer not initialized. Call init_render\\(\\) first."):
            renderer.render(mock_fs)

    @patch('matplotlib.pyplot.imshow')
    @patch('matplotlib.pyplot.axis')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.pause')
    def test_render_human_mode_initialized(self, mock_pause, mock_show, mock_title,
                                          mock_axis, mock_imshow, mock_fs, mock_turbine):
        """Test render with human mode when initialized"""
        renderer = WindFarmRenderer(render_mode="human")
        renderer.init_render(mock_fs, mock_turbine)

        with patch.object(renderer, '_render_frame_for_human',
                         return_value=np.zeros((100, 100, 3))):
            result = renderer.render(mock_fs)
            assert result is None
            mock_imshow.assert_called_once()
            mock_show.assert_called_once()


class TestRenderFrameForHuman:
    """Test _render_frame_for_human method and probe rendering (lines 160-217)"""

    def test_render_frame_for_human_basic(self, mock_fs, mock_turbine):
        """Test basic _render_frame_for_human without probes"""
        renderer = WindFarmRenderer(render_mode="human")
        renderer.init_render(mock_fs, mock_turbine)

        frame = renderer._render_frame_for_human(mock_fs)

        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3  # RGB

    def test_render_frame_for_human_with_baseline(self, mock_fs, mock_turbine):
        """Test _render_frame_for_human with baseline"""
        renderer = WindFarmRenderer(render_mode="human")
        renderer.init_render(mock_fs, mock_turbine)

        mock_fs_baseline = MagicMock()
        mock_fs_baseline.windTurbines = mock_fs.windTurbines
        mock_fs_baseline.wind_direction = 270
        mock_fs_baseline.time = 10.0
        mock_fs_baseline.get_windspeed = mock_fs.get_windspeed

        frame = renderer._render_frame_for_human(mock_fs, mock_fs_baseline, baseline=True)

        assert isinstance(frame, np.ndarray)

    def test_render_frame_for_human_view_none_with_turbine(self, mock_fs, mock_turbine):
        """Test line 133: _render_frame_for_human initializes render when view is None"""
        renderer = WindFarmRenderer(render_mode="human")
        # view is None initially
        assert renderer.view is None

        frame = renderer._render_frame_for_human(mock_fs, turbine=mock_turbine)

        # After rendering, view should be initialized
        assert renderer.view is not None
        assert isinstance(frame, np.ndarray)

    def test_render_frame_for_human_with_ws_probe(self, mock_fs, mock_turbine):
        """Test lines 165-169: _render_frame_for_human with WS probe"""
        renderer = WindFarmRenderer(render_mode="human")
        renderer.init_render(mock_fs, mock_turbine)

        # Create WS probe
        ws_probe = MagicMock(spec=WindProbe)
        ws_probe.position = (100, 50, 90)
        ws_probe.probe_type = "WS"
        ws_probe.read = MagicMock(return_value=8.5)
        ws_probe.get_inflow_angle_to_turbine = MagicMock(return_value=np.pi / 4)

        frame = renderer._render_frame_for_human(mock_fs, probes=[ws_probe])

        assert isinstance(frame, np.ndarray)
        ws_probe.read.assert_called()
        ws_probe.get_inflow_angle_to_turbine.assert_called()

    def test_render_frame_for_human_with_ti_probe(self, mock_fs, mock_turbine):
        """Test lines 170-174: _render_frame_for_human with TI probe"""
        renderer = WindFarmRenderer(render_mode="human")
        renderer.init_render(mock_fs, mock_turbine)

        # Create TI probe
        ti_probe = MagicMock(spec=WindProbe)
        ti_probe.position = (200, 100, 90)
        ti_probe.probe_type = "TI"
        ti_probe.read = MagicMock(return_value=0.12)
        ti_probe.get_inflow_angle_to_turbine = MagicMock(return_value=np.pi / 6)

        frame = renderer._render_frame_for_human(mock_fs, probes=[ti_probe])

        assert isinstance(frame, np.ndarray)
        ti_probe.read.assert_called()

    def test_render_frame_for_human_with_unknown_probe(self, mock_fs, mock_turbine):
        """Test lines 175-178: _render_frame_for_human with unknown probe type"""
        renderer = WindFarmRenderer(render_mode="human")
        renderer.init_render(mock_fs, mock_turbine)

        # Create unknown probe
        unknown_probe = MagicMock(spec=WindProbe)
        unknown_probe.position = (300, 150, 90)
        unknown_probe.probe_type = "UNKNOWN"
        unknown_probe.read = MagicMock(return_value=999)
        unknown_probe.get_inflow_angle_to_turbine = MagicMock(return_value=0)

        frame = renderer._render_frame_for_human(mock_fs, probes=[unknown_probe])

        assert isinstance(frame, np.ndarray)

    def test_render_frame_for_human_with_multiple_probes(self, mock_fs, mock_turbine,
                                                          mock_probes_ws_ti_unknown):
        """Test lines 160-217: _render_frame_for_human with all probe types"""
        renderer = WindFarmRenderer(render_mode="human")
        renderer.init_render(mock_fs, mock_turbine)

        frame = renderer._render_frame_for_human(mock_fs, probes=mock_probes_ws_ti_unknown)

        assert isinstance(frame, np.ndarray)
        # Verify all probes were processed
        for probe in mock_probes_ws_ti_unknown:
            probe.read.assert_called()
            probe.get_inflow_angle_to_turbine.assert_called()


class TestRenderFrame:
    """Test _render_frame method (lines 255-260)"""

    def test_render_frame_view_none_no_turbine(self, mock_fs):
        """Test lines 255-259: _render_frame raises RuntimeError when view and turbine are None"""
        renderer = WindFarmRenderer(render_mode="rgb_array")

        with pytest.raises(RuntimeError, match="Renderer not initialized and turbine not provided"):
            renderer._render_frame(mock_fs, turbine=None)

    def test_render_frame_view_none_with_turbine(self, mock_fs, mock_turbine):
        """Test line 260: _render_frame initializes render when view is None but turbine provided"""
        renderer = WindFarmRenderer(render_mode="rgb_array")
        assert renderer.view is None

        frame = renderer._render_frame(mock_fs, turbine=mock_turbine)

        # After rendering, view should be initialized
        assert renderer.view is not None
        assert isinstance(frame, np.ndarray)

    def test_render_frame_with_ws(self, mock_fs, mock_turbine):
        """Test _render_frame with wind speed parameter"""
        renderer = WindFarmRenderer(render_mode="rgb_array")
        renderer.init_render(mock_fs, mock_turbine)

        frame = renderer._render_frame(mock_fs, turbine=mock_turbine, ws=10)

        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3  # RGB

    def test_render_frame_baseline(self, mock_fs, mock_turbine):
        """Test _render_frame with baseline parameter"""
        renderer = WindFarmRenderer(render_mode="rgb_array")
        renderer.init_render(mock_fs, mock_turbine)

        mock_fs_baseline = MagicMock()
        mock_fs_baseline.windTurbines = mock_fs.windTurbines
        mock_fs_baseline.wind_direction = 270
        mock_fs_baseline.time = 10.0
        mock_fs_baseline.get_windspeed = mock_fs.get_windspeed

        frame = renderer._render_frame(mock_fs, fs_baseline=mock_fs_baseline,
                                       baseline=True, turbine=mock_turbine)

        assert isinstance(frame, np.ndarray)


class TestPlotFarm:
    """Test plot_farm method (lines 328-330)"""

    def test_plot_farm_without_turbine(self, mock_fs):
        """Test plot_farm without turbine (skips init_render)"""
        renderer = WindFarmRenderer(render_mode="rgb_array")

        with patch.object(renderer, '_render_farm') as mock_render_farm:
            renderer.plot_farm(mock_fs, turbine=None)
            mock_render_farm.assert_called_once()

    def test_plot_farm_with_turbine(self, mock_fs, mock_turbine):
        """Test lines 328-330: plot_farm with turbine initializes and renders"""
        renderer = WindFarmRenderer(render_mode="rgb_array")

        with patch.object(renderer, 'init_render') as mock_init, \
             patch.object(renderer, '_render_farm') as mock_render_farm:

            renderer.plot_farm(mock_fs, turbine=mock_turbine)

            mock_init.assert_called_once_with(mock_fs, mock_turbine)
            mock_render_farm.assert_called_once()

    def test_plot_farm_with_baseline(self, mock_fs, mock_turbine):
        """Test plot_farm with baseline parameter"""
        renderer = WindFarmRenderer(render_mode="rgb_array")
        mock_fs_baseline = MagicMock()

        with patch.object(renderer, 'init_render'), \
             patch.object(renderer, '_render_farm') as mock_render_farm:

            renderer.plot_farm(mock_fs, fs_baseline=mock_fs_baseline,
                              turbine=mock_turbine, baseline=True)

            mock_render_farm.assert_called_once_with(mock_fs, mock_fs_baseline, True)


class TestRenderFarm:
    """Test _render_farm method (lines 341-365)"""

    @patch('matplotlib.pyplot.ion')
    @patch('matplotlib.pyplot.gca')
    @patch('matplotlib.pyplot.pcolormesh')
    @patch('matplotlib.pyplot.gcf')
    @patch('IPython.display.display')
    @patch('IPython.display.clear_output')
    def test_render_farm_basic(self, mock_clear, mock_display, mock_gcf,
                               mock_pcolormesh, mock_gca, mock_ion,
                               mock_fs, mock_turbine):
        """Test lines 341-365: _render_farm renders farm for IPython"""
        renderer = WindFarmRenderer(render_mode="rgb_array")
        renderer.init_render(mock_fs, mock_turbine)

        mock_ax = MagicMock()
        mock_gca.return_value = mock_ax
        mock_fig = MagicMock()
        mock_gcf.return_value = mock_fig

        renderer._render_farm(mock_fs)

        # Verify IPython display calls
        mock_ion.assert_called_once()
        mock_display.assert_called_once()
        mock_clear.assert_called_once_with(wait=True)

        # Verify plotting calls
        mock_pcolormesh.assert_called_once()
        assert mock_ax.set_title.called
        assert mock_ax.set_aspect.called

    @patch('matplotlib.pyplot.ion')
    @patch('matplotlib.pyplot.gca')
    @patch('matplotlib.pyplot.pcolormesh')
    @patch('matplotlib.pyplot.gcf')
    @patch('IPython.display.display')
    @patch('IPython.display.clear_output')
    def test_render_farm_with_baseline(self, mock_clear, mock_display, mock_gcf,
                                       mock_pcolormesh, mock_gca, mock_ion,
                                       mock_fs, mock_turbine):
        """Test _render_farm with baseline parameter"""
        renderer = WindFarmRenderer(render_mode="rgb_array")
        renderer.init_render(mock_fs, mock_turbine)

        mock_ax = MagicMock()
        mock_gca.return_value = mock_ax
        mock_fig = MagicMock()
        mock_gcf.return_value = mock_fig

        mock_fs_baseline = MagicMock()
        mock_fs_baseline.windTurbines = mock_fs.windTurbines
        mock_fs_baseline.wind_direction = 270
        mock_fs_baseline.time = 10.0
        mock_fs_baseline.get_windspeed = mock_fs.get_windspeed

        renderer._render_farm(mock_fs, mock_fs_baseline, baseline=True)

        mock_display.assert_called_once()
        mock_clear.assert_called_once()


class TestPlotFrame:
    """Test plot_frame method"""

    def test_plot_frame_without_turbine(self, mock_fs, mock_turbine):
        """Test plot_frame without turbine parameter"""
        renderer = WindFarmRenderer(render_mode="rgb_array")
        renderer.init_render(mock_fs, mock_turbine)

        with patch.object(renderer, '_render_frame') as mock_render_frame:
            renderer.plot_frame(mock_fs)
            mock_render_frame.assert_called_once()

    def test_plot_frame_with_turbine(self, mock_fs, mock_turbine):
        """Test plot_frame with turbine parameter"""
        renderer = WindFarmRenderer(render_mode="rgb_array")

        with patch.object(renderer, 'init_render') as mock_init, \
             patch.object(renderer, '_render_frame') as mock_render_frame:

            renderer.plot_frame(mock_fs, turbine=mock_turbine)

            mock_init.assert_called_once()
            mock_render_frame.assert_called_once()


class TestClose:
    """Test close method"""

    def test_close(self, mock_fs, mock_turbine):
        """Test close cleans up matplotlib resources"""
        renderer = WindFarmRenderer(render_mode="rgb_array")
        renderer.init_render(mock_fs, mock_turbine)

        # Verify objects exist
        assert renderer.figure is not None
        assert renderer.ax is not None
        assert renderer.view is not None

        renderer.close()

        # Verify cleanup
        assert renderer.figure is None
        assert renderer.ax is None
        assert renderer.view is None


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_multiple_renders(self, mock_fs, mock_turbine):
        """Test multiple render calls work correctly"""
        renderer = WindFarmRenderer(render_mode="rgb_array")
        renderer.init_render(mock_fs, mock_turbine)

        frame1 = renderer._render_frame(mock_fs, turbine=mock_turbine)
        frame2 = renderer._render_frame(mock_fs, turbine=mock_turbine)

        assert isinstance(frame1, np.ndarray)
        assert isinstance(frame2, np.ndarray)

    def test_close_and_reopen(self, mock_fs, mock_turbine):
        """Test closing and reinitializing renderer"""
        renderer = WindFarmRenderer(render_mode="rgb_array")
        renderer.init_render(mock_fs, mock_turbine)
        renderer.close()

        # Reinitialize after close
        renderer.init_render(mock_fs, mock_turbine)
        frame = renderer._render_frame(mock_fs, turbine=mock_turbine)

        assert isinstance(frame, np.ndarray)
