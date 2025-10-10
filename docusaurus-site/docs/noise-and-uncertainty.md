# Noise, Uncertainty, and Advanced Agents

Real-world wind farm control systems must operate with imperfect information from noisy sensors. WindGym models this challenge through a modular system designed to introduce and manage measurement uncertainty. This allows for the development and testing of more robust, real-world-ready agents.

---

## 1. The Measurement & Noise System

The system is composed of two key components: the `MeasurementManager` and the `NoisyWindFarmEnv` wrapper.

### The `MeasurementManager`

The `MeasurementManager` is the brain behind the noise system. It is initialized with a clean environment instance and builds a detailed specification (`MeasurementSpec`) for every variable in the observation space. Its primary roles are:
-   **Defining Noise Models**: It allows you to set a `NoiseModel` (e.g., `WhiteNoiseModel`, `EpisodicBiasNoiseModel`, or a `HybridNoiseModel`) that defines the statistical properties of the noise in physical units (e.g., a standard deviation of 2.0 degrees for wind direction).
-   **Applying Noise**: At each step, it takes the clean observation from the base environment and applies the defined noise model to generate a noisy observation.

### The `NoisyWindFarmEnv` Wrapper

This is a Gymnasium `Wrapper` that orchestrates the noise application. Its workflow is:
1.  It contains a clean, underlying `WindFarmEnv` instance.
2.  On `reset()` or `step()`, it first calls the underlying environment to get a "ground truth" clean observation.
3.  It passes this clean observation to the `MeasurementManager`.
4.  The `MeasurementManager` applies the configured noise and returns a noisy observation.
5.  The wrapper then passes this noisy observation to the agent.
6.  Crucially, the wrapper adds both the `clean_obs` and detailed `noise_info` (including any biases applied) to the `info` dictionary, making it possible to analyze the difference between ground truth and the agent's perception.

---

## 2. The Nature of Yaw in Simulation

It's critical to understand how "yaw" is defined within the DYNAMIKS simulation backend versus how it might be measured in the real world.

-   **Simulation State (Ground Truth):** In DYNAMIKS, a turbine's orientation is defined by its **yaw offset** relative to the **true, average wind direction**. A yaw offset of 0° means the turbine is perfectly aligned with the true average incoming flow.

-   **Agent's Perception:** Any agent designed for deployment should not have access to this perfect ground truth direction information. This requires introducing synthetic errors when performing a desired action and when recording measured information.

---

## 3. How Measurement Errors Impact Control

Because the simulation's yaw is relative to the *true* wind, a noisy global wind direction measurement does not directly affect the physics. However, it critically affects the agent's *decision-making process*.

-   **`ActionMethod: "yaw"`**: The agent's action is a **change** in yaw offset (e.g., `+1°`). The execution of this action is unaffected by wind direction error. However, the agent's policy (especially a model-based one like `PyWakeAgent`) uses the sensed wind direction to calculate the *optimal* yaw offset. 

-   **`ActionMethod: "wind"`**: The agent's action is a **target** yaw offset. The environment controller then works to move the turbine to this target.

---

## 4. How Measurement Errors Impact the Recorded Measurements

Several measurements are available during wind farm simulation. The turbine- and farm-level measured direction is obviously corrupted by uncertianty in direction. 

WindGym currently only records the yaw position relative to the true wind direction. So, when there is uncertainty in wind direction, the yaw positions should also be confounded. 

---

## 5. PyWake Agent under uncertainty

The `NoisyPyWakeAgent` is specifically designed to handle this challenge. It does not trust any single measurement. Instead, it:
1.  Receives the full, noisy observation vector.
2.  Identifies all measurements related to wind speed and wind direction (both turbine-level and farm-level).
3.  Un-scales and **averages** these values to create a single, more robust estimate of the true wind conditions.
4.  Uses this **estimated** wind condition to run its internal PyWake optimization to determine the best yaw angles to command.


-   **`ActionMethod: "wind"`**: The agent's action is a **target** yaw offset relative to the sensed wind direction. The environment controller then works to move the turbine to this target. Because of the explicit dependancy on the sensed wind direction, the noisy pywake agent uses the sensed wind direction when determining the appropriate offset to make, leading to potentially large missalignments with the incoming flow.

-   **`ActionMethod: "yaw"`**: Since the agent is meant to predict changes in the yaw postion, we assume this is relative to the turbine controller, which generally is assumed to not be vulnerable to errors in wind direction

