#!/usr/bin/env python3
"""
Real-time Vision-Language Social Navigation for Go1 in MuJoCo
- Supports multiple dynamic pedestrians
- Captures FPV frames during simulation
- Sends frames to LLaVA with system_prompt.txt
- Parses JSON output and executes motion primitives
- Maintains forward progress per SCAND guidelines
"""

import time
from pathlib import Path
import cv2
import os
import numpy as np
import json
import sys
import re

import mujoco
from mujoco import MjvCamera, MjvOption, MjrContext, MjvScene, MjrRect
import glfw

# Import your modules
from motion_primitives import ACTIONS
from control import update_base_kinematics

# Import your LLaVA function
sys.path.append(str(Path(__file__).parent / "vlm"))
from prompt_llava import prompt_llava, load_system_prompt

# --- Configuration ---
ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "sim" / "go1_sidewalk_scene.xml"  # Change per scenario
IMAGE_OUTPUT_DIR = "captured_images_sidewalk"
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

SYSTEM_PROMPT = load_system_prompt("system_prompt.txt")
if not SYSTEM_PROMPT:
    raise RuntimeError("system_prompt.txt not found!")

LLAVA_MODEL = "llava"
VLM_INTERVAL_STEPS = 250     # ~2 Hz analysis
FRAME_BUFFER_SIZE = 2
PRIMITIVE_DURATION = 0.4

class VLMActionController:
    def __init__(self):
        self.primitives = ["FORWARD_SLOW"]
        self.current_idx = 0
        self.t_in_primitive = 0.0
        self.frame_buffer = []
        self.last_vlm_step = -1

    def update_from_vlm(self, vlm_json: dict, step: int):
        try:
            primitives = vlm_json["motion_primitives"]["primitives"]
            valid_primitives = [p for p in primitives if p in ACTIONS]
            if not valid_primitives:
                raise ValueError("No valid primitives found")
            self.primitives = valid_primitives
            self.current_idx = 0
            self.t_in_primitive = 0.0
            self.last_vlm_step = step
            print(f" VLM @ step {step}: {vlm_json['selected_action']['action']} → {primitives}")
        except Exception as e:
            print(f" VLM parse fallback: {e}")
            self.primitives = ["FORWARD_SLOW"]
            self.current_idx = 0

    def get_current_action(self, dt: float):
        if not self.primitives:
            return ACTIONS["FORWARD_SLOW"]
        action = ACTIONS[self.primitives[self.current_idx]]
        self.t_in_primitive += dt
        if self.t_in_primitive >= PRIMITIVE_DURATION:
            self.current_idx = (self.current_idx + 1) % len(self.primitives)
            self.t_in_primitive = 0.0
        return action

def clean_llava_json(raw: str) -> str:
    cleaned = raw.replace('\\_', '_')
    cleaned = re.sub(r'\\(?!["\\/bfnrt])', '', cleaned)
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    return match.group() if match else cleaned

def analyze_with_vlm(image_paths):
    user_prompt = """
You are a Vision-Language Social Navigation Policy Selector for a robot in a MuJoCo simulation.
CRITICAL SCENE CONTEXT:
- This is a REAL SIDEWALK scenario (width ≈ 3 m) with mixed pedestrian traffic:
  a) An ONCOMING pedestrian moving TOWARD the robot (leftward, -X direction) → "Against Traffic"
  b) A SAME-DIRECTION pedestrian moving AHEAD of the robot (rightward, +X direction) → "With Traffic"
- The robot must COMBINE two social behaviors simultaneously:
  1. Yield laterally to the oncoming agent by KEEPING RIGHT
  2. Maintain slow forward progress behind the same-direction agent
- This is NOT a "Narrow Doorway" — the space is wide enough for safe lateral maneuvering.

Apply the following rules from system_prompt.txt:
1. Use SCAND labels: ["Against Traffic", "With Traffic", "Sidewalk"]
2. Select SECOND-ORDER ACTION: **KEEP_RIGHT_AGAINST_TRAFFIC**
   - Reason: The dominant interaction is with the oncoming pedestrian; the same-direction agent only requires speed modulation.
3. Map to motion primitives: **["STRAFE_RIGHT", "FORWARD_SLOW"]**
   - STRAFE_RIGHT creates lateral clearance from the oncoming agent (≥1 m)
   - FORWARD_SLOW maintains safe following distance behind the same-direction agent
4. DO NOT select FOLLOW_FLOW — the oncoming agent takes precedence.
5. DO NOT use TURN_* or BYPASS_* — no full detour is needed on a wide sidewalk.

Analyze frame sequence to confirm:
- Oncoming pedestrian is within 3–4 m and closing
- Same-direction pedestrian is 1–2 m ahead, moving at ~0.3 m/s
- Right-side clearance is ≥1 m (left side may be constrained)

Output ONLY valid JSON as per system_prompt.txt.
"""
    response = prompt_llava(
        user_prompt=user_prompt,
        image_paths=image_paths,
        system_prompt=SYSTEM_PROMPT,
        model=LLAVA_MODEL,
        verbose=False
    )
    try:
        cleaned = clean_llava_json(response)
        return json.loads(cleaned)
    except (json.JSONDecodeError, Exception) as e:
        print(f" VLM JSON error: {e}")
        print(f"Raw: {response[:200]}...")
        return None

def get_base_body_id(model):
    for name in ["trunk", "base", "base_link", "pelvis"]:
        try:
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        except ValueError:
            continue
    return 0

def main():
    print("=== VLM SOCIAL NAVIGATION SYSTEM ===")
    
    # Load model
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    dt = model.opt.timestep
    
    # =================================================================
    # Parse ALL human joints (person1, person2, ...)
    # =================================================================
    person_joints = []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name and name.startswith("person"):
            qaddr = model.jnt_qposadr[j]
            person_joints.append((name, qaddr))
    
    print(f"Found {len(person_joints)} persons:", [name for name, _ in person_joints])
    
    # Set initial positions based on XML
    initial_positions = {
        "person1_slide": 2.0,  # match pos="2 0.3 0"
        "person2_slide": 3.0,  # match pos="3 -0.2 0"
    }
    for name, qaddr in person_joints:
        init_pos = initial_positions.get(name, 2.0)
        data.qpos[qaddr] = init_pos
    
    # Same speed for all (With Traffic)
    person_speeds = {name: 0.45 for (name, addr) in person_joints}
    
    # Setup offline rendering
    glfw.init()
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE)  # Show window
    window = glfw.create_window(800, 600, "VLM Social Nav", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(0)  # Disable VSync for max FPS

    scene = MjvScene(model, maxgeom=20000)
    cam = MjvCamera()
    opt = MjvOption()
    ctx = MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    
    # FPV camera aligned with robot's forward direction (+X)
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    camid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'fpv')
    cam.fixedcamid = camid
    
    W, H = 800, 600
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    vp = MjrRect(0, 0, W, H)
    
    controller = VLMActionController()
    step_count = 0
    t_last = time.time()
    
    print("Robot + humans in motion. VLM analysis every ~0.5s...")
    
    try:
        while not glfw.window_should_close(window):
            now = time.time()
            
            # Time-stepping loop
            while (now - t_last) >= dt:
                # Update ALL persons
                for (pname, qaddr) in person_joints:
                    data.qpos[qaddr] += person_speeds[pname] * dt
                
                # Capture frame periodically
                if step_count % 30 == 0:
                    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
                    mujoco.mjr_render(vp, scene, ctx)
                    mujoco.mjr_readPixels(rgb, None, vp, ctx)
                    rotated = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    filename = f"{IMAGE_OUTPUT_DIR}/frame_{step_count:06d}.png"
                    cv2.imwrite(filename, rotated[...,::-1])
                    controller.frame_buffer.append(filename)
                    if len(controller.frame_buffer) > FRAME_BUFFER_SIZE:
                        controller.frame_buffer.pop(0)
                
                # Trigger VLM analysis
                if (step_count % VLM_INTERVAL_STEPS == 0 and 
                    len(controller.frame_buffer) >= FRAME_BUFFER_SIZE and
                    step_count != controller.last_vlm_step):
                    print(f"\n VLM analysis @ step {step_count}...")
                    result = analyze_with_vlm(controller.frame_buffer.copy())
                    if result:
                        controller.update_from_vlm(result, step_count)
                
                # Apply robot control
                action = controller.get_current_action(dt)
                update_base_kinematics(model, data, action.vx, action.vy, action.wz, dt, action.stop)
                
                # Sync all state changes
                mujoco.mj_forward(model, data)
                
                step_count += 1
                t_last += dt
            
            # Render to screen
            viewport = MjrRect(0, 0, 800, 600)
            mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
            mujoco.mjr_render(viewport, scene, ctx)
            glfw.swap_buffers(window)
            glfw.poll_events()
            
    except KeyboardInterrupt:
        print("\n Stopped by user.")
    finally:
        glfw.terminate()
        print(" Simulation ended.")

if __name__ == "__main__":
    main()