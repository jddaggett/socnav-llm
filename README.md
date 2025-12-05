# Learning Socially Compliant Navigation with Vision–Language Models and Motion Primitives

**Authors:**  
Heyang Huang · heyang.huang@colorado.edu  
Jackson Daggett · jackson.daggett@colorado.edu  

---

##  Overview

Social navigation requires robots to move in ways that not only avoid collisions but also respect human norms—maintaining interpersonal distance, yielding appropriately, and behaving predictably in shared spaces. Traditional geometric planners struggle with these context-dependent behaviors, while end-to-end imitation learning often lacks interpretability and fails to generalize.

This project introduces a **modular, interpretable social navigation framework** that combines:

- A Vision–Language Model (LLaVA) for high-level semantic reasoning  
-  A library of **8 socially grounded motion primitives**  
-  A real-time MuJoCo simulation with Unitree Go1 and human agents  
-  Deterministic primitive mapping for safe, consistent execution  

The system enables a robot to reason about human–robot interactions through language, while ensuring reliable, real-time control through interpretable action primitives.

---

##  Key Contributions

- **Hybrid VLM + Motion Primitive Architecture**  
  Structured JSON outputs from a VLM map deterministically to 8 primitives (e.g., `FORWARD_MED`, `TURN_LEFT`, `YIELD`) that encode social navigation behaviors.

- **Semantic, Human-Aligned Navigation**  
  The VLM reasons about flow direction, personal space, and interaction context directly from first-person Go1 camera images.

- **SCAND-Inspired Social Abstractions**  
  We distill 12 human social navigation scenarios into a compact and interpretable primitive library.

- **Real-Time Closed-Loop Execution**  
  LLaVA runs locally (via **Ollama**) for low-latency control, enabling practical real-time action selection.

- **Simulation Benchmarking**  
  We evaluate on MuJoCo in a Sidewalk scenario containing both “Following-with-Traffic’’ and “Head-On Approaching’’ interactions.

- **Improved Social Compliance over Geometric Baselines**  
  Experiments show smoother trajectories, fewer oscillations, and better adherence to proxemics than RRT-based planners.

---

##  System Architecture

Our navigation stack includes four layers:

### **1. Perception Layer (Go1 FPV Camera)**
- Captures first-person RGB images from the Unitree Go1 model in MuJoCo.

### **2. Vision–Language Reasoning (LLaVA via Ollama)**
- Receives FPV image and a structured system prompt.  
- Produces **constrained JSON** specifying:  
  ```json
  {
      "action": "FOLLOW_FLOW_LEFT",
      "principle": "Maintain safe distance and match crowd direction."
  }

### **3. Scenario Interpreter & Primitive Mapper**

- Maps high-level actions to sequences of first-order motion primitives
   (e.g., `FOLLOW_FLOW_LEFT → [STRAFE_LEFT, FORWARD_MED]`).

### **4. Low-Level Execution**

- Sends velocity commands to the Go1 controller.
- Ensures safety (min-distance filters, clipped velocities).

## Motion Primitive Library

We define **8 atomic primitives**, grounded in SCAND’s taxonomy:

| Primitive           | Description                             |
| ------------------- | --------------------------------------- |
| `FORWARD_SLOW`      | Careful approach or following           |
| `FORWARD_MED`       | Normal walking speed                    |
| `FORWARD_FAST`      | Catching up or overtaking               |
| `STRAFE_LEFT/RIGHT` | Side-stepping to respect personal space |
| `TURN_LEFT/RIGHT`   | Heading adjustment                      |
| `STOP`              | Yielding or waiting                     |

These primitives serve as a **socially meaningful action vocabulary**, enabling interpretable planning.

##  Social Scenario Abstractions

Based on SCAND, we define **12 canonical social navigation scenarios**, including:

- Head-on approaching
- Following with traffic
- Crossing paths
- Merging into flow
- Passing / overtaking
- Group avoidance

The VLM is prompted with these definitions to ground its semantic reasoning.

## Experiments & Results

### **Scenario: Sidewalk Navigation**

The robot encounters two human agents—one moving with traffic, one moving against it.

<img src="file:///C:/Users/Huang%20Heyang/Desktop/Project/sidewalk.png" alt="img" style="zoom:67%;" />

### **Results**

- The RRT baseline shows **large oscillations** and frequent violations of the **1.2 m personal space threshold**.
- The VLM-driven policy maintains **smooth, monotonic interpersonal distances**.
- Both following and head-on interactions are handled gracefully.

### **User Study**

A pilot survey (N=20) indicates that participants perceive VLM-driven behavior as:

- More natural
- More polite
- More comfortable to walk around

<img src="file:///C:/Users/Huang%20Heyang/Desktop/Project/survey_q1_preference.png" alt="img" style="zoom:30%;" />

### **How To Run It?**

#### 1. Prerequisites

- Python 3.9+ 
- A GPU with CUDA support (RTX 30/40)  
- MuJoCo (tested on MuJoCo Python bindings v3.1.1) 
- Ollama for local LLaVA inference (or another locally hosted VLM)  
---

#### 2. Clone the Repository

```bash
git clone https://github.com/jddaggett/socnav-llm.git
cd socnav-llm
```

#### **3. Install MuJoCo + GLFW**

Mac/Linux:

```
sudo apt install libglfw3-dev
```

Windows:
 Follow the MuJoCo installation docs.

#### **4. Install Ollama + LLaVA**

```
brew install ollama
ollama pull llava
```
### Running the Simulation
Start the VLM Server
ollama run llava

Launch Navigation Controller
python sim/vlm_socnav.py

## Known Limitations & Future Work

- Evaluation currently limited to simulation
- VLM inference at low frequency (~2 Hz)
- Primitive library may need expansion for dense crowds
- Future versions: reactive pedestrians, end-to-end VLA refinement

## Acknowledgments

We thank the authors of SCAND, LLaVA, and Unitree Go1 models.
This project was developed for CSCI 5322: Algorithmic Foundations of Human-Robot Interaction at CU Boulder.
