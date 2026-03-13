# Model Evaluation and Validation

## 1. Hyperparameter Tuning

The model's behavior is governed by several hyperparameters, which were selected through a combination of literature-based calibration, domain knowledge, and sensitivity analysis.

**Particle Simulation Parameters**

- **Wind drag coefficient (α)**: Controls the degree to which surface wind influences plastic drift. Oceanographic literature (Maximenko et al., 2012; van Sebille et al., 2020) reports values between 0.01 and 0.03 for floating debris. The selected value α = 0.02 falls within this range and provides a balanced representation of wind-driven transport relative to ocean currents.
- **Diffusion coefficient (K)**: Represents sub-grid-scale turbulent mixing. K = 5 m²/s aligns with surface-layer diffusivities used in Lagrangian ocean models (Okubo, 1971; Lumpkin et al., 2017). Higher values increase particle dispersion; lower values produce more coherent patch structures.
- **Timestep size**: A 6-hour timestep aligns with HYCOM's temporal resolution and maintains numerical stability by keeping the Courant number below unity.
- **Particle count**: 500 particles provide sufficient spatial coverage of the river emission region while remaining computationally tractable for iterative optimization.

**Density Estimation Parameters**

- **Grid resolution**: 0.25° (~28 km at the equator) matches typical ocean model resolutions and captures major accumulation zones without excessive noise.
- **Kernel bandwidth**: The histogram-based density estimation does not use a kernel; the grid resolution effectively defines the spatial smoothing scale.

**Optimization Parameters**

- **Vessel speed**: 5° per day (~550 km/day) approximates typical cleanup vessel cruising speeds.
- **Sweep width**: 0.1° (~11 km) approximates the effective collection width of towed cleanup systems.
- **Fuel penalty weight (λ)**: λ = 0.01 balances plastic collection against fuel cost. This value was tuned so that routes favor high-density patches without excessive detours.

Parameter ranges were determined from published studies and physical constraints. Final values were selected through sensitivity analysis, varying one parameter at a time and assessing the impact on plastic transport patterns and optimization results.

---

## 2. Validation Techniques

**Drift Model Validation**

- **Trajectory consistency**: Particle trajectories were compared against known Pacific circulation patterns (e.g., Kuroshio Current, North Pacific Gyre). Particles originating from Asian river mouths were expected to drift toward the subtropical gyre—a pattern the model reproduces.
- **Accumulation zone alignment**: Simulated density fields were qualitatively compared with observed plastic accumulation zones (e.g., North Pacific Garbage Patch). The model produces elevated density in the subtropical gyre region.
- **Stability checks**: Repeated runs with different random seeds were used to assess trajectory variance. While diffusion introduces stochasticity, ensemble means remain stable.
- **Parameter sensitivity**: Drift patterns were tested across α ∈ [0.01, 0.03] and K ∈ [1, 10] m²/s to ensure results remain physically plausible within the literature range.

**Optimization Model Validation**

- **Baseline comparison**: Optimized routes were compared against a random routing baseline (randomly selected waypoints within range) and a greedy nearest-patch strategy. The greedy optimization consistently outperforms these baselines.
- **Multi-run evaluation**: Performance metrics were averaged over multiple simulation runs to account for stochasticity in particle positions and density fields.
- **Sector partitioning**: For multi-vessel routing, sector-based partitioning was validated by ensuring vessels do not overlap and that total plastic collected scales with fleet size.

These validation approaches are appropriate for a stochastic environmental simulation because they emphasize ensemble behavior, consistency with physical expectations, and relative performance against simple baselines rather than exact point predictions.

---

## 3. Performance Metrics

**Plastic Recovery Efficiency**

$$\text{Recovery Efficiency} = \frac{\text{plastic\_collected}}{\text{total\_available\_plastic}}$$

Measures the fraction of simulated plastic that is collected. Total available plastic is the sum of density over all grid cells within the mission region.

**Fuel Efficiency**

$$\text{Fuel Efficiency} = \frac{\text{plastic\_collected}}{\text{fuel\_consumed}}$$

Plastic collected per unit fuel (in distance-degrees). Higher values indicate better fuel utilization.

**Route Efficiency**

$$\text{Route Efficiency} = \frac{\text{plastic\_collected}}{\text{distance\_traveled}}$$

Plastic collected per unit distance traveled. Reflects how effectively routes target high-density areas.

**Coverage**

$$\text{Coverage} = \frac{\text{number of high-density cells visited}}{\text{total high-density cells}} \times 100\%$$

High-density cells are defined as those above a threshold (e.g., 90th percentile of the density distribution). Coverage indicates what fraction of the plastic-rich area is visited.

**Simulation Stability**

Variance of plastic collected across repeated runs with different random seeds. Lower variance indicates more stable and predictable performance.

**Tradeoff Analysis**

The objective function maximizes plastic_collected − λ × fuel_cost. Varying λ traces the Pareto frontier between plastic collected and fuel cost. Small λ favors maximum collection; large λ favors fuel savings. The chosen λ = 0.01 produces routes that prioritize high-density patches while avoiding excessive detours.

---

## 4. Interpretation of Results

**Plastic Transport Patterns**

The simulation shows plastic from Asian river sources (Meijer et al., 2021) drifting into the western North Pacific, with accumulation in the subtropical gyre region (approximately 115–145°E, 15–35°N). Wind drift and diffusion spread particles and create patchy density fields rather than narrow filaments. This aligns with observed accumulation zones and Lagrangian studies of marine debris.

**Optimal Cleanup Strategies**

- **Port placement**: Locating the port within or near the particle concentration region (e.g., 125°E, 22°N) substantially improves collection compared to a port far from the patches.
- **Target selection**: The greedy algorithm prioritizes high-density cells within daily range. Routes tend to form loops through dense patches rather than long linear transits.
- **Multi-vessel coordination**: Sector-based partitioning reduces overlap and increases total collection, but performance depends on how plastic is distributed across sectors.

**System Sensitivity**

- **Wind coefficient (α)**: Higher α increases westward drift and shifts accumulation zones. Routes must adapt to these shifts.
- **Diffusion (K)**: Higher K disperses plastic more, reducing peak densities and making routes less concentrated.
- **Decay rate**: The plastic decay term (0.001 per timestep) gradually reduces available plastic; its effect is modest over 90-day missions but relevant for longer horizons.

**Limitations**

- **Transport uncertainty**: Ocean currents and wind are approximated; real conditions are more variable.
- **Missing environmental factors**: Stokes drift, wave effects, vertical mixing, and biofouling are not included.
- **Simplified vessel dynamics**: Fuel consumption is approximated; real operations involve weather, maintenance, and crew constraints.
- **Static density assumption**: Routing uses a single snapshot; in practice, density evolves during the mission.

**Potential Improvements**

- Incorporate real wind data (e.g., ERA5) instead of synthetic fields.
- Improve plastic degradation and sinking models.
- Implement multi-vessel coordination (e.g., vehicle routing problem formulations) to reduce overlap and improve coverage.
- Use rolling-horizon optimization that updates routes as new density forecasts become available.
