# Learning Socially Compliant Navigation Policies from Human Demonstrations via Large Language Models

**Authors:**  
Heyang Huang · heyang.huang@colorado.edu  
Jackson Daggett · jackson.daggett@colorado.edu  

---

## Problem Statement & HRI Motivation
Service and delivery robots are increasingly deployed in human-populated environments, where socially compliant navigation is essential for safety and human trust. Unlike traditional obstacle avoidance, social navigation requires robots to infer and respect implicit human norms, such as yielding at intersections, maintaining personal space, and avoiding blocking human paths. Designing explicit reward functions to capture these nuanced, context-dependent behaviors is difficult. Learning from Demonstrations (LfD) offers an alternative by directly learning policies from human trajectories, yet purely end-to-end imitation learning often lacks interpretability and adaptability. 

To address these challenges, our project integrates Behavior Cloning (BC) with a Vision-Language Model (VLM) for socially compliant navigation. The BC component captures low-level reactive behaviors from human demonstrations, while the VLM provides high-level semantic reasoning and interpretable action generation. We will validate the proposed framework first in a Mujoco simulation—chosen for its realistic physics and fine-grained control of humanoid environments—and later deploy it on the Unitree Go1 quadruped robot for real-world testing. The VLM (LLaVA) is deployed locally via Ollama, ensuring low-latency inference and offline reproducibility. This research aims to advance interpretable and socially aware navigation in embodied AI.

---

## Related Work
Imitation learning has been widely used in navigation. The **SCAND dataset (Karnan et al., 2022)** provides large-scale teleoperated trajectories with rich multi-modal data and social interaction tags, enabling policies that reflect human norms. However, such end-to-end models often lack interpretability. **NavCon (Harel et al., 2023)** and **Code-as-Policies (Liang et al., 2023)** demonstrate modular designs where large language models generate interpretable code that invokes pre-defined APIs. **PaLM-E (Driess et al., 2023)** and **NaVILA (Cheng et al., 2024)** showcase multimodal reasoning for embodied control but without grounding in real human demonstrations. Our work builds upon these ideas by grounding VLM-driven planning within a BC-trained behavioral foundation, achieving both semantic interpretability and data-grounded social compliance.

---

## System Plan

### Simulator & Robot
We will use **Mujoco** as the primary simulator, given its high-fidelity dynamics, humanoid model library, and real-time **ROS2** compatibility. **Webots** may serve as an auxiliary visualization platform. The final real-world deployment will use the **Unitree Go1 quadruped robot**.

### Dataset
We will employ the **SCAND dataset**, which includes multimodal sensory streams and annotations of social interactions, for training and evaluating behavior cloning policies.

### Integration Pipeline
Our framework follows a three-layer architecture:
1. **Low-Level Controller (BC):**  
   Trained on SCAND trajectories to learn primitives such as `yield_to_human()`, `maintain_distance()`, and `follow_path()`.  
2. **API Layer:**  
   Exposes these learned primitives as callable functions through a ROS2 interface.  
3. **High-Level Reasoner (VLM):**  
   A locally hosted LLaVA model (via Ollama) observes visual inputs and contextual prompts, then generates structured action plans composed of API calls. This ensures semantic reasoning while maintaining behavioral grounding.

### Human Simulation Setup
To simulate social contexts, we will employ humanoid models in Mujoco to represent static or semi-dynamic pedestrians:
- **Stage 1 – Static Human Benchmark:** humanoids remain stationary in varied poses and positions to test proxemic compliance. 
- **Stage 2 – Semi-Dynamic Social Scenarios:** selected humanoids follow pre-defined slow trajectories to emulate natural movement. 

This setup allows scalable and reproducible evaluation of social navigation performance without requiring fully interactive agents.

### Metrics
We will evaluate across four main dimensions: 
- **Safety:** collision rate, minimum distance to obstacles. 
- **Social Compliance:** adherence to personal space and yielding norms. 
- **Human-Likeness:** trajectory similarity to human demonstrations. 
- **Interpretability:** clarity and semantic coherence of VLM-generated explanations.

---

## Evaluation Plan

Our evaluation includes both **quantitative** and **qualitative** analyses following recent social navigation frameworks *(Singamaneni et al., 2024)*.

### Quantitative Evaluation
In Mujoco, the robot will navigate through human-populated environments with varying densities. Metrics such as minimum interpersonal distance, path smoothness, and time-to-goal will be computed. Comparison against SCAND trajectories will assess human-likeness.

### Qualitative Evaluation
We will assess the robot’s behavior through proxemics and social acceptance principles *(Li et al., 2019)*. For each scenario, the VLM’s natural language explanations will be analyzed to determine if they align with social norms (e.g., “yielding to a person ahead” or “keeping respectful distance”).

### Human Models & Behavioral Modes
1. **Fixed-Trajectory Agents:** Replay recorded human paths for baseline evaluation. 
2. **Reactive Agents (future work):** Introduce simple rule-based or ORCA-based pedestrians to study two-way social dynamics. 

This two-tiered evaluation allows reproducible benchmarking and gradual increase of social realism.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-------------|
| VLM outputs may be noisy or unsafe | Constrain output via API calls; enforce safety filters at the BC level. |
| Domain gap between SCAND and MuJoCo | Collect additional small-scale demonstrations in simulation for fine-tuning. |
| Latency in VLM inference | Run LLaVA locally via **Ollama** for real-time inference and predictable response. |

---

## Timeline (Approx. 8 Weeks)

| Weeks | Milestones |
|--------|-------------|
| 1–2 | Literature review, SCAND baseline reproduction |
| 3–4 | BC primitive design, VLM integration (LLaVA via Ollama) |
| 5–6 | MuJoCo environment setup with humanoid social agents |
| 7–8 | Evaluation, analysis, and final report preparation |

---

## Resource Needs

**Compute:** RTX 4090 GPU for BC training and LLaVA inference.
**Software:** Mujoco (main simulator), Webots (optional visualization), ROS2 for integration.  
**Dataset:** SCAND dataset (publicly available).
**Hardware:** Unitree Go1 quadruped robot for real-world deployment. 

---

## Expected Contributions
- A modular framework that unites demonstration-grounded BC and VLM-based semantic reasoning for interpretable social navigation. 
- A benchmark of socially compliant navigation behaviors in Mujoco using humanoid-based social contexts. 
- A reproducible open-source implementation bridging LLaVA reasoning with ROS2 motion primitives.

---
