#!/usr/bin/env python3
"""
RRT* Geometric Navigation with FPV Video Capture
- Pure geometric planner (no VLM)
- Captures FPV frames identically to VLM system
- Generates comparable video for evaluation
- Uses dynamic obstacle modeling from MuJoCo state
"""

import time
from pathlib import Path
import cv2
import os
import numpy as np
import random
import math
import sys

import mujoco
from mujoco import MjvCamera, MjvOption, MjrContext, MjvScene, MjrRect
import glfw

from motion_primitives import ACTIONS
from control import update_base_kinematics

# --- Configuration ---
ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "sim" / "go1_sidewalk_scene.xml"  # Use same scene as VLM
IMAGE_OUTPUT_DIR = "rrt_star_captured_images"
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

# RRT* parameters
GOAL_POS = np.array([8.0, 0.0])
HUMAN_RADIUS = 0.4
ROBOT_RADIUS = 0.3
MAX_ITER = 400
STEP_SIZE = 0.3
REPLAN_INTERVAL = 10  # steps

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def nearest(nodes, point):
    return min(nodes, key=lambda n: distance((n.x, n.y), point))

def steer(from_node, to_point, step_size=STEP_SIZE):
    dx = to_point[0] - from_node.x
    dy = to_point[1] - from_node.y
    dist = math.hypot(dx, dy)
    if dist == 0:
        return Node(from_node.x, from_node.y)
    scale = min(step_size, dist) / dist
    return Node(from_node.x + dx * scale, from_node.y + dy * scale)

def collision_free(p1, p2, obstacles, robot_radius=ROBOT_RADIUS, human_radius=HUMAN_RADIUS):
    for (ox, oy) in obstacles:
        A = p1[0] - p2[0]
        B = p1[1] - p2[1]
        C = p2[0] - p1[0]
        D = p2[1] - p1[1]
        dot = C * (ox - p1[0]) + D * (oy - p1[1])
        len_sq = C * C + D * D
        param = -1
        if len_sq != 0:
            param = dot / len_sq
        if param < 0:
            closest = p1
        elif param > 1:
            closest = p2
        else:
            closest = (p1[0] + param * C, p1[1] + param * D)
        dist_to_obstacle = distance(closest, (ox, oy))
        if dist_to_obstacle < (robot_radius + human_radius):
            return False
    return True

def rrt_star_planner(start, goal, obstacles, max_iter=MAX_ITER, step_size=STEP_SIZE):
    start_node = Node(start[0], start[1])
    nodes = [start_node]
    
    for _ in range(max_iter):
        if random.random() < 0.3:
            rand_point = goal
        else:
            rand_x = random.uniform(start[0], start[0] + 3.0)
            rand_y = random.uniform(start[1] - 1.5, start[1] + 1.5)
            rand_point = (rand_x, rand_y)
        
        nearest_node = nearest(nodes, rand_point)
        new_node = steer(nearest_node, rand_point, step_size)
        
        if not collision_free((nearest_node.x, nearest_node.y), (new_node.x, new_node.y), obstacles):
            continue
        
        new_node.parent = nearest_node
        new_node.cost = nearest_node.cost + distance((nearest_node.x, nearest_node.y), (new_node.x, new_node.y))
        nodes.append(new_node)
        
        if distance((new_node.x, new_node.y), goal) < step_size:
            path = []
            current = new_node
            while current is not None:
                path.append((current.x, current.y))
                current = current.parent
            path.reverse()
            return path[1] if len(path) >= 2 else path[0]
    
    return (start[0] + 0.3, start[1])  # fallback

def get_base_body_id(model):
    for name in ["trunk", "base", "base_link", "pelvis"]:
        try:
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        except ValueError:
            continue
    return 0

def main():
    print("=== RRT* GEOMETRIC NAVIGATION (FPV VIDEO CAPTURE) ===")
    
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    dt = model.opt.timestep
    
    # Parse person joints
    person_joints = []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name and name.startswith("person"):
            qaddr = model.jnt_qposadr[j]
            person_joints.append((name, qaddr))
    
    # Initial positions
    initial_positions = {"person1_slide": 4.0, "person2_slide": 1.0}
    for name, qaddr in person_joints:
        data.qpos[qaddr] = initial_positions.get(name, 2.0)
    
    person_speeds = {"person1_slide": -0.3, "person2_slide": 0.4}
    
    # Setup rendering
    glfw.init()
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
    window = glfw.create_window(800, 600, "RRT* Nav - FPV", None, None)
    glfw.make_context_current(window)
    
    scene = MjvScene(model, maxgeom=20000)
    cam = MjvCamera()
    opt = MjvOption()
    ctx = MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    
    # Use main_fpv camera (make sure it's in your XML!)
    try:
        camid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'fpv')
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = camid
    except ValueError:
        print(" 'main_fpv' camera not found. Using free camera.")
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 2.0
        cam.elevation = -30
        cam.azimuth = 90
        cam.lookat = np.array([0, 0, 0.3])
    
    W, H = 800, 600
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    vp = MjrRect(0, 0, W, H)
    
    step_count = 0
    t_last = time.time()
    image_count = 0
    rrt_target = None
    
    print("Starting RRT* navigation with FPV capture...")
    
    try:
        while not glfw.window_should_close(window):
            now = time.time()
            while (now - t_last) >= dt:
                # Update persons
                for (pname, qaddr) in person_joints:
                    data.qpos[qaddr] += person_speeds[pname] * dt
                
                # Get human world positions
                human_positions = []
                for (pname, qaddr) in person_joints:
                    # Find body associated with joint
                    for b in range(model.nbody):
                        if model.body_jntnum[b] > 0:
                            j0 = model.body_jntadr[b]
                            if qaddr == model.jnt_qposadr[j0]:
                                hx, hy = data.xpos[b][0], data.xpos[b][1]
                                human_positions.append((hx, hy))
                                break
                
                robot_pos = (data.qpos[0], data.qpos[1])
                
                # Re-plan every N steps
                if step_count % REPLAN_INTERVAL == 0:
                    rrt_target = rrt_star_planner(robot_pos, GOAL_POS, human_positions)
                
                # Track target
                if rrt_target is not None:
                    tx, ty = rrt_target
                    dx, dy = tx - robot_pos[0], ty - robot_pos[1]
                    dist = max(0.01, math.hypot(dx, dy))
                    vx = min(0.3, dx / dist * 0.3)
                    vy = min(0.3, dy / dist * 0.3)
                else:
                    vx, vy = 0.3, 0.0
                
                # Apply control
                update_base_kinematics(model, data, vx, vy, 0.0, dt, stop=False)
                mujoco.mj_forward(model, data)
                
                # Capture frame
                if step_count % 10 == 0:
                    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
                    mujoco.mjr_render(vp, scene, ctx)
                    mujoco.mjr_readPixels(rgb, None, vp, ctx)
                    rotated = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    filename = f"{IMAGE_OUTPUT_DIR}/frame_{image_count:06d}.png"
                    cv2.imwrite(filename, rotated[...,::-1])
                    image_count += 1
                
                step_count += 1
                t_last += dt
                
                # Stop when near goal
                if robot_pos[0] > 7.0:
                    print("Reached goal!")
                    break
            
            # Render to screen
            viewport = MjrRect(0, 0, 800, 600)
            mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
            mujoco.mjr_render(viewport, scene, ctx)
            glfw.swap_buffers(window)
            glfw.poll_events()
    
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        glfw.terminate()
        
        # Generate video
        if image_count > 0:
            img_array = []
            for i in range(image_count):
                filename = f"{IMAGE_OUTPUT_DIR}/frame_{i:06d}.png"
                img = cv2.imread(filename)
                if img is not None:
                    img_array.append(img)
            if img_array:
                height, width, _ = img_array[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('rrt_star_nav_demo.mp4', fourcc, 20.0, (width, height))
                for img in img_array:
                    out.write(img)
                out.release()
                print(" RRT* video saved as 'rrt_star_nav_demo.mp4'")
        else:
            print(" No frames captured")

if __name__ == "__main__":
    main()