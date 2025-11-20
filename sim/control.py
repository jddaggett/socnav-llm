# controller for the go1 robot base as a point mass with yaw

import mujoco
import numpy as np

def quat2yaw(q):
    # yaw = 2arctan2(qz, qw)
    qw, qx, qy, qz = q
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw

def yaw2quat(yaw):
    # q(yaw) = [cos(yaw/2), 0, 0, sin(yaw/2)]
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    return np.array([cy, 0.0, 0.0, sy])

def update_base_kinematics(model, data, vx, vy, wz, dt, stop=False):
    if stop: return

    # mujoco root pose uses position vector and orientation quaternion
    x, y, z = data.qpos[:3]
    q = data.qpos[3:7]
 
    # [dx, dy] = R(yaw) * [vx, vy]
    # -> add delta pos to current pos
    yaw = quat2yaw(q)
    c, s = np.cos(yaw), np.sin(yaw)
    new_x = x + (c * vx - s * vy) * dt
    new_y = y + (s * vx + c * vy) * dt

    # no change in z dir, yaw updated by wz and converted back to quat
    new_z = z
    new_yaw = yaw + (wz * dt)
    new_q = yaw2quat(new_yaw)

    # write to data and update mujoco state
    data.qpos[:3] = [new_x, new_y, new_z]
    data.qpos[3:7] = new_q
    mujoco.mj_forward(model, data)

# ==============================================================================
# Controller class
class Go1BaseVelocityController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.dt = model.opt.timestep

    def apply_action(self, action):
        if action.stop:
            self.data.qvel[0:6] = 0.0
            return
        vx_body = action.vx
        vy_body = action.vy
        wz      = action.wz

        # current base orientation
        q = self.data.qpos[3:7]
        yaw = quat2yaw(q)
        c, s = np.cos(yaw), np.sin(yaw)

        # rotate body-frame velocity into world frame
        vx_world = c * vx_body - s * vy_body
        vy_world = s * vx_body + c * vy_body

        # set base linear and angular velocity in world frame
        # qvel[0:3] = linear vel (x,y,z)
        # qvel[3:6] = angular vel (about x,y,z)
        self.data.qvel[0:3] = [vx_world, vy_world, 0.0]
        self.data.qvel[3:6] = [0.0, 0.0, wz]

    def step(self, action):
        self.apply_action(action)
        mujoco.mj_step(self.model, self.data)
