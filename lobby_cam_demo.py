import time
from pathlib import Path
import cv2
import os
import numpy as np

import mujoco
from mujoco import MjvCamera, MjvOption, MjrContext, MjvScene, MjrRect
import glfw

from motion_primitives import ACTIONS
from control import update_base_kinematics

ROOT = Path(__file__).resolve().parents[1]
SCENE = (ROOT / "sim" / "go1_lobby_scene.xml").resolve()

# 创建输出文件夹
IMAGE_OUTPUT_DIR = "captured_images"
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

def get_base_body_id(model):
    for name in ["trunk", "base", "base_link", "pelvis"]:
        try:
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        except ValueError:
            continue
    return 0

def main():
    forward_action = ACTIONS["FORWARD_SLOW"]

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    dt = model.opt.timestep

    base_id = get_base_body_id(model)

    person_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "person1_slide")
    person_qaddr = model.jnt_qposadr[person_joint_id]

    data.qpos[person_qaddr] = 4.0
    person_speed = 0.15

    # 初始化 OpenGL 上下文
    glfw.init()
    glfw.window_hint(glfw.VISIBLE, False)
    window = glfw.create_window(800, 600, "Hidden Render", None, None)
    glfw.make_context_current(window)

    # 设置渲染上下文
    scene = MjvScene(model, maxgeom=20000)
    cam = MjvCamera()
    opt = MjvOption()
    ctx = MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

    # ✅ 重要：确保摄像头使用固定的摄像头模式
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    camid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'fpv')
    cam.fixedcamid = camid

    W, H = 800, 600
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    vp = MjrRect(0, 0, W, H)

    print("=== START PURE KINEMATIC DEMO (OFFLINE) ===")
    print("Capturing images every 10 steps...")

    step_count = 0
    image_count = 0
    t_last = time.time()

    try:
        while True:
            now = time.time()

            # 控制循环
            while (now - t_last) >= dt:
                vx, vy, wz, stop = forward_action.vx, forward_action.vy, forward_action.wz, forward_action.stop

                update_base_kinematics(model, data, vx, vy, wz, dt, stop)

                data.qpos[person_qaddr] += person_speed * dt

                mujoco.mj_forward(model, data)

                step_count += 1
                t_last += dt

                # 每 10 步捕获一帧图像
                if step_count % 20 == 0:
                    try:
                        
                        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
                        mujoco.mjr_render(vp, scene, ctx)
                        mujoco.mjr_readPixels(rgb, None, vp, ctx)

                        rotated_rgb = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)

                        # 保存图像
                        filename = f"{IMAGE_OUTPUT_DIR}/frame_{image_count:06d}.png"
                        cv2.imwrite(filename, rotated_rgb[...,::-1])
                        print(f"Saved {filename}")
                        image_count += 1
                        
                        if image_count >= 100:  # 限制数量
                            print("Reached maximum frame count. Stopping...")
                            break

                    except Exception as e:
                        print(f"Error capturing image: {e}")
                        print(f"Error details: {e}")
                        # 如果渲染失败，跳过这次捕获
                        pass
            
            if image_count >= 100:
                break

    except KeyboardInterrupt:
        print(f"\nStopped by user. Captured {image_count} frames.")
    
    finally:
        # 清理资源
        glfw.destroy_window(window)
        glfw.terminate()
        print(f"Images saved to: {IMAGE_OUTPUT_DIR}")

if __name__ == "__main__":
    main()