# Agents

In WindGym, an **agent** is the intelligent entity responsible for making decisions within the wind farm environment. These decisions typically involve adjusting the yaw angles of wind turbines to achieve specific objectives, such as maximizing power production or minimizing structural loads.

WindGym provides a modular framework for developing and integrating various types of agents, from simple rule-based controllers to sophisticated reinforcement learning policies.

---

## The `BaseAgent`

All agents in WindGym are built upon the `BaseAgent` class. This foundational class defines a common interface that all agents must adhere to, primarily requiring a `predict()` method. This ensures consistency and allows different agents to be seamlessly interchanged within the simulation and evaluation frameworks.

The `BaseAgent` also includes utility functions for scaling yaw angles to the environment's expected action space (typically between -1 and 1) and inversely unscaling them back to degrees.

---

## Types of Agents

WindGym comes with several pre-built agent implementations that serve as examples, baselines, or simple controllers for various testing and comparison purposes.

### 1. `PyWakeAgent`

The `PyWakeAgent` is a powerful baseline agent that leverages the **PyWake** library's advanced wind farm modeling and optimization capabilities.

* **Behavior**: For a given set of static wind conditions (wind speed, direction, turbulence intensity), the `PyWakeAgent` calculates the theoretically optimal yaw angles for all turbines in the farm that would maximize the total power production according to its internal wake models. It then applies these optimized, static yaw angles throughout the simulation.
* **Use Case**: Ideal as a high-performing benchmark to compare against dynamic, learning-based agents. It represents a "perfect" steady-state controller.

### 2. `GreedyAgent`

The `GreedyAgent` is a simple, rule-based controller.

* **Behavior**: This agent attempts to align each turbine's yaw angle with the perceived wind direction. It can operate based on either the *local* wind direction at each turbine (influenced by wakes) or the *global* incoming wind direction. It tries to move towards a zero yaw offset (aligned with the wind) or a fixed, global setpoint.
* **Use Case**: Useful for quick sanity checks, demonstrating basic control logic, or as a very simple baseline against which more complex agents can show improvement.

### 3. `RandomAgent`

As its name suggests, the `RandomAgent` takes completely random actions within the permissible action space of the environment.

* **Behavior**: In each step, the `RandomAgent` generates a random set of yaw adjustments (or target yaw angles, depending on the environment's `ActionMethod`) for all turbines.
* **Use Case**: Primarily used for testing the environment's stability, ensuring that it can handle arbitrary inputs without crashing, and establishing a "worst-case" performance baseline. Any agent should ideally perform better than a `RandomAgent`.

### 4. `ConstantAgent`

The `ConstantAgent` applies a predefined, fixed set of yaw angles to all turbines throughout the simulation.

* **Behavior**: This agent is initialized with a specific array of yaw angles (e.g., all zeros, or a fixed offset for specific turbines), and it maintains these angles without change.
* **Use Case**: Useful for evaluating the wind farm's performance under static yaw settings, verifying specific design conditions, or as a simple non-reactive baseline.

---

## Agent-Environment Interaction

Regardless of their internal logic, all agents interact with the WindGym environment through the standard Gymnasium `step()` method. They receive an observation (sensor data from the wind farm), process it with their `predict()` method to determine an action, and send that action back to the environment. The environment then applies the action, simulates the next timestep, and returns a new observation, reward, and episode status.

When developing custom agents, you will typically create a new Python class that inherits from `BaseAgent` and implements your desired control or learning algorithm within its `predict()` method.
