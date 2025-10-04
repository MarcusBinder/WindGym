# Evaluation Framework

Evaluating the performance of wind farm control agents is crucial for understanding their effectiveness and comparing different control strategies. WindGym provides a flexible and comprehensive evaluation framework, centered around the `AgentEval` and `Coliseum` classes.

## 1. Objectives of Evaluation

Effective evaluation in WindGym aims to:

- **Quantify Performance**: Measure how well an agent achieves its objectives (e.g., maximizing power, minimizing loads).
- **Compare Agents**: Benchmark the performance of different agents against each other and against baseline controllers.
- **Assess Robustness**: Determine how an agent performs under varying wind conditions, turbulence levels, and farm configurations.
- **Identify Strengths and Weaknesses**: Pinpoint scenarios where an agent excels or struggles.
- **Track Progress**: Monitor the learning progress of reinforcement learning agents over time.

## 2. The AgentEval Class (Legacy)

The `AgentEval` class (and its associated `eval_single_fast` function) provides a way to evaluate a single agent under specific, fixed wind conditions.

- **Fixed Conditions**: You can explicitly set parameters like wind speed, wind direction, and turbulence intensity for an evaluation run.
- **Time Series Data**: It records detailed time series data for power output, yaw angles, and wind speeds for both the agent-controlled farm and a baseline farm.
- **Basic Plotting**: It includes basic plotting functionalities to visualize the flow field and time series results.
- **Xarray Dataset Output**: Results are stored in an `xarray.Dataset`, a powerful data structure for multi-dimensional data, facilitating post-processing and analysis.

While `AgentEval` is functional, the `Coliseum` class offers a more advanced and scalable approach for multi-agent comparisons and broader evaluation scenarios.

## 3. The Coliseum Class (Recommended)

The `Coliseum` class, located in `WindGym/utils/evaluate_PPO.py`, is the recommended and most powerful tool for evaluating multiple agents in WindGym. It provides a robust framework for systematic performance assessment.

### Key Features of Coliseum:

- **Multi-Agent Comparison**: Easily compare the performance of several different agents simultaneously.
- **Flexible Environment Factory**: It takes an "environment factory" (a function that returns a new environment instance) allowing it to create fresh environments for each evaluation run, ensuring isolation and reproducibility.

### Time Series Evaluation (`run_time_series_evaluation`):

- Designed for realistic, long-term performance assessment.
- Utilizes the `sample_site` feature of `WindFarmEnv` to draw wind conditions from realistic statistical distributions (e.g., Weibull for wind speed, frequency for wind direction).
- Runs multiple episodes (e.g., 10, 50, 100) and aggregates results to provide average performance and standard deviations.
- Can save detailed episode histories for in-depth analysis.

### Wind Grid Evaluation (`run_wind_grid_evaluation`):

- Systematically evaluates agents across a predefined grid of fixed wind conditions (e.g., varying wind speeds, directions, and turbulence intensities).
- Generates an `xarray.Dataset` containing the mean reward for each agent at each point in the wind grid.
- Ideal for creating performance maps and identifying optimal operating points or challenging conditions for agents.

### Comprehensive Plotting:

Includes built-in methods for visualizing:
- Mean cumulative rewards over time for multiple agents.
- Summary bar charts comparing average performance across agents.
- Heatmaps or line plots of agent performance across the wind condition grid.

### Detailed Statistics:

Provides summary statistics (mean, std, min, max, median) of agent performance across all evaluation episodes.

### Workflow with Coliseum:

1. **Define Agent(s)**: Instantiate the `BaseAgent` subclasses you want to evaluate (e.g., `PyWakeAgent`, `RandomAgent`, your custom RL agent).
2. **Create Environment Factory**: Define a Python function (or lambda) that returns a new instance of your `WindFarmEnv` (or `FarmEval`) with the desired base parameters. This ensures each evaluation run starts with a clean environment.
3. **Initialize Coliseum**: Pass your environment factory and a dictionary of agents (or a list of agents with labels) to the `Coliseum` constructor.
4. **Run Evaluations**: Call `run_time_series_evaluation` for stochastic, long-term tests, or `run_wind_grid_evaluation` for systematic fixed-condition analysis.
5. **Analyze Results**: Use Coliseum's plotting methods or access the returned `pandas.DataFrame` or `xarray.Dataset` for further custom analysis and visualization.

## 4. Considerations for Evaluation

- **Baseline Selection**: Always compare your agents against a sensible baseline (e.g., a `PyWakeAgent`, a simple `GreedyAgent`, or a traditional control strategy).
- **Wind Condition Representativeness**: Choose evaluation wind conditions (fixed points or a `sample_site` for stochastic runs) that are representative of the real-world scenarios your agent will encounter.
- **Episode Length**: Ensure evaluation episodes are long enough for wake effects to develop and for agents to demonstrate stable behavior.
- **Reproducibility**: Use fixed random seeds for consistent evaluation runs.
- **Metrics**: Consider a variety of metrics beyond just total power, such as power efficiency, yaw activity, and potentially load indicators if using high-fidelity models.

By leveraging the `Coliseum` framework, you can rigorously test and validate your WindGym agents, ensuring their robustness and effectiveness in diverse wind farm scenarios.
