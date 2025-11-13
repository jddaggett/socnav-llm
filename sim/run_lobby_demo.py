# Run a "lobby head-on encounter" demo:

import time
import argparse
from pathlib import Path

import mujoco
import mujoco.viewer as viewer

from motion_primitives import ACTIONS
from control import update_base_kinematics


ROOT = Path(__file__).resolve().parents[1]
SCENE = (ROOT / "sim" / "go1_lobby_scene.xml").resolve()


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

    dt = model.opt.timestep
    base_id = get_base_body_id(model)

    person_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "person1_slide")
    person_qaddr = model.jnt_qposadr[person_joint_id]

    person_speed = 0.3  

    data.qpos[3:7] = [0, 0, 1, 0]#rotation quaternion
    mujoco.mj_resetData(model, data)

    t_last = time.time()

    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            now = time.time()
            
            while (now - t_last) >= dt:
                vx, vy, wz, stop = forward_action.vx, forward_action.vy, forward_action.wz, forward_action.stop

                update_base_kinematics(model, data, vx, vy, wz, dt, stop)

                data.qpos[person_qaddr] += person_speed * dt
                mujoco.mj_step(model, data)

                x, y, z = data.xpos[base_id]

                t_last += dt

            v.sync()


if __name__ == "__main__":
    main()
