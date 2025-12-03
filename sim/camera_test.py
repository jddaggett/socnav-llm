import os
os.environ["MUJOCO_GL"] = "glfw"

import mujoco
import numpy as np
import glfw
import cv2

# 初始化 GLFW
glfw.init()
glfw.window_hint(glfw.VISIBLE, False)
window = glfw.create_window(800, 600, "Hidden Window", None, None)
glfw.make_context_current(window)

# 加载模型
xml_path = "sim/go1_lobby_scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

scene = mujoco.MjvScene(model, maxgeom=20000)
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

# 让机器人稳定下来
for _ in range(400):
    mujoco.mj_step(model, data)

# 设置摄像头为 FPV 摄像头
cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
camid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'fpv')
cam.fixedcamid = camid

W, H = 800, 600
rgb = np.zeros((H, W, 3), dtype=np.uint8)
vp = mujoco.MjrRect(0, 0, W, H)

# 渲染
mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
mujoco.mjr_render(vp, scene, ctx)
mujoco.mjr_readPixels(rgb, None, vp, ctx)

rotated_rgb = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)

# 保存图像
cv2.imwrite("fpv_camera_corrected.png", rotated_rgb[...,::-1])
print("Saved fpv_camera_corrected.png")

glfw.destroy_window(window)
glfw.terminate()