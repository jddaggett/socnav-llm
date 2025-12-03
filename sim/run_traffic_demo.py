import time
from pathlib import Path
import mujoco
import mujoco.viewer as viewer

# -----------------------------------------
# YOUR MODULE IMPORTS (unchanged)
# -----------------------------------------
from motion_primitives import ACTIONS
from control import update_base_kinematics

# -----------------------------------------
# SETUP
# -----------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SCENE = (ROOT / "sim" / "scene_with_traffic_2.xml").resolve()

def get_base_body_id(model):
    """Find the robot base (usually 'trunk')."""
    for name in ["trunk", "base", "base_link", "pelvis"]:
        try:
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        except ValueError:
            continue
    return 0


# =====================================================================
#                      MAIN CONTROL LOOP
# =====================================================================
def main():

    # Choose your primitive (robot following slow traffic)
    action = ACTIONS["FORWARD_SLOW"]

    # Load XML
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)

    # First forward pass initializes xpos/xmat
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep

    # Robot base
    base_id = get_base_body_id(model)

    # -----------------------------------------
    # Parse all humans (person1, person2, …)
    # -----------------------------------------
    person_joints = []

    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name is None:
            continue
        if name.startswith("person"):
            qaddr = model.jnt_qposadr[j]
            person_joints.append((name, qaddr))

    print(f"Found {len(person_joints)} persons:", person_joints)

    # Assign each person a speed (same direction, same flow)
    person_speeds = {name: 0.35 for (name, addr) in person_joints}

    t_last = time.time()
    step_count = 0

    # =================================================================
    #                      VIEWER + CONTROL LOOP
    # =================================================================
    with viewer.launch_passive(model, data) as v:
        print("=== Running Scene: WITH TRAFFIC 2 ===")

        while v.is_running():
            now = time.time()

            while (now - t_last) >= dt:

                # -----------------------------------------------------
                # ROBOT KINEMATIC CONTROL
                # -----------------------------------------------------
                vx, vy, wz, stop = action.vx, action.vy, action.wz, action.stop
                update_base_kinematics(model, data, vx, vy, wz, dt, stop)

                # -----------------------------------------------------
                # MOVE ALL PEOPLE
                # -----------------------------------------------------
                for (pname, qaddr) in person_joints:
                    data.qpos[qaddr] += person_speeds[pname] * dt

                # Update kinematics only
                mujoco.mj_forward(model, data)

                # ---- Debug print every 50 steps ----
                if step_count % 50 == 0:
                    robot_x = data.qpos[0]
                    px = data.xpos[base_id][0]
                    print(f"[{step_count}] robot_x={robot_x:.3f}")

                step_count += 1
                t_last += dt

            v.sync()


# =====================================================================
# ENTRY
# =====================================================================
if __name__ == "__main__":
    main()
