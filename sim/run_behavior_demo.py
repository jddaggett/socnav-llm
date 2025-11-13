# run_behavior_demo.py
import argparse
import json
from pathlib import Path
import time
import mujoco
import mujoco.viewer as viewer

from motion_primitives import ACTIONS
from behavior_map import BEHAVIORS
from control import update_base_kinematics

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "sim" / "go1_scene.xml"

def load_behavior_script(path):
    path = Path(path)
    with path.open("r") as f:
        behavior_list = json.load(f)
    return behavior_list

def expand_behavior(behavior_name):
    return BEHAVIORS[behavior_name]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=str, required=True,
                        help="Path to JSON script containing behavior sequence.")
    args = parser.parse_args()

    behavior_list = load_behavior_script(args.script)

    action_sequence = []
    for behavior in behavior_list:
        behavior_name = behavior["behavior"]
        action_sequence += expand_behavior(behavior_name)

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)

    dt = model.opt.timestep
    step_i = 0
    t_in_step = 0.0
    t_last = time.time()

    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            now = time.time()
            while (now - t_last) >= dt:
                name = action_sequence[step_i]["action"]
                dur  = action_sequence[step_i]["duration"]
                act  = ACTIONS[name]

                update_base_kinematics(model, data, act.vx, act.vy, act.wz, dt, act.stop)
                mujoco.mj_step(model, data)

                t_in_step += dt
                t_last += dt

                if t_in_step >= dur:
                    step_i = (step_i + 1) % len(action_sequence)
                    t_in_step = 0.0

                print(f"[Behavior Runner] Action={name}, dur={dur:.2f}s, t={t_in_step:.2f}s")

            v.sync()

if __name__ == "__main__":
    main()
