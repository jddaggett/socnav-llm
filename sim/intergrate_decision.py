import time
from pathlib import Path
import cv2
import os
import numpy as np
import json
import queue
import threading
from pathlib import Path

import mujoco
from mujoco import MjvCamera, MjvOption, MjrContext, MjvScene, MjrRect
import glfw

from motion_primitives import ACTIONS
from control import update_base_kinematics

# 从你的 prompt_llava 导入函数
from prompt_llava import prompt_llava

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

def analyze_frame_with_vlm(image_path, step_count):
    """使用 VLM 分析单帧图像"""
    try:
        # VLM 提示词：社交导航分析
        user_prompt = f"""
        分析这幅机器人视角的图像，用于社交导航决策：
        
        1. 场景描述：描述图像中看到的环境和人物
        2. 距离估计：估算机器人与前方人物的距离
        3. 行为建议：建议机器人下一步应采取的行动（前进/等待/避让/转向）
        4. 安全性：评估当前场景的安全性
        5. 详细分析：提供具体的导航建议
        
        请以JSON格式返回分析结果，包含以下字段：
        - scene_description: 场景描述
        - distance_estimate: 距离估计
        - action_recommendation: 动作建议
        - safety_level: 安全等级（1-5，5最安全）
        - detailed_analysis: 详细分析
        """
        
        # 调用 VLM
        response = prompt_llava(
            user_prompt,
            image_paths=[image_path],
            system_prompt="你是一个专业的社交导航AI助手，专门分析机器人视觉输入并提供导航决策。请提供准确、安全、礼貌的导航建议。"
        )
        
        print(f"VLM Analysis for Step {step_count}:")
        print(response)
        print("-" * 80)
        
        # 尝试解析 JSON 响应
        try:
            # 提取 JSON 部分（如果有的话）
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                analysis = json.loads(json_str)
            else:
                analysis = {"raw_response": response}
        except:
            analysis = {"raw_response": response}
        
        return analysis
        
    except Exception as e:
        print(f"VLM Analysis Error: {e}")
        return {"error": str(e)}

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

    # 设置摄像头
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    camid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'fpv')
    cam.fixedcamid = camid

    W, H = 800, 600
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    vp = MjrRect(0, 0, W, H)

    print("=== START SOCIAL NAVIGATION WITH VLM ===")
    print("Capturing images and analyzing with VLM every 20 steps...")

    step_count = 0
    image_count = 0
    t_last = time.time()

    # 存储 VLM 分析结果
    analysis_results = []

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

                # 每 20 步捕获图像并分析（避免 VLM 负载过重）
                if step_count % 20 == 0:
                    try:
                        # 渲染图像
                        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
                        mujoco.mjr_render(vp, scene, ctx)
                        mujoco.mjr_readPixels(rgb, None, vp, ctx)

                        # 旋转图像（你之前发现的正确角度）
                        rotated_rgb = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)

                        # 保存图像
                        filename = f"{IMAGE_OUTPUT_DIR}/frame_{image_count:06d}_step_{step_count}.png"
                        cv2.imwrite(filename, rotated_rgb[...,::-1])
                        print(f"Saved {filename}")

                        # 分析图像
                        analysis = analyze_frame_with_vlm(filename, step_count)
                        analysis_results.append({
                            "step": step_count,
                            "image": filename,
                            "analysis": analysis
                        })

                        # 根据 VLM 建议调整机器人行为（可选）
                        if "action_recommendation" in analysis:
                            action = analysis["action_recommendation"]
                            print(f"VLM suggests: {action}")
                            
                            # 这里可以添加基于 VLM 建议的控制逻辑
                            # 例如：如果建议避让，可以临时改变机器人速度

                        image_count += 1
                        
                        if image_count >= 50:  # 分析 50 张图像后停止
                            print("Reached maximum analysis count. Stopping...")
                            break

                    except Exception as e:
                        print(f"Error in capture/analysis: {e}")
                        continue
            
            if image_count >= 50:
                break

    except KeyboardInterrupt:
        print(f"\nStopped by user. Analyzed {image_count} frames.")
    
    finally:
        # 保存分析结果
        with open(f"{IMAGE_OUTPUT_DIR}/analysis_results.json", "w") as f:
            json.dump(analysis_results, f, indent=2)
        
        # 清理资源
        glfw.destroy_window(window)
        glfw.terminate()
        print(f"Results saved to: {IMAGE_OUTPUT_DIR}")
        print("Social navigation analysis complete!")

if __name__ == "__main__":
    main()