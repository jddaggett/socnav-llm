# Run a demo of Go1 robot executing a sequence of motion primitives
# from a JSON script in test_scripts/
# Example usage: mjpython sim/run_go1_demo.py --script test_scripts/walk.json

import argparse
import json
from pathlib import Path
import time
import mujoco
import mujoco.viewer as viewer

from motion_primitives import ACTIONS
from control import update_base_kinematics

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "sim" / "go1_scene.xml"

# assumes script is well-formed
def load_script(path):
    path = Path(path)
    with path.open("r") as f:
        act_seq = json.load(f)
    return act_seq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=str, required=True,
                        help="Path to JSON script defining motion primitive sequence.")
    args = parser.parse_args()
    SCRIPT = load_script(args.script)

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
                name = SCRIPT[step_i]["action"]
                dur = SCRIPT[step_i]["duration"]
                act = ACTIONS[name]
                vx, vy, wz, stop = act.vx, act.vy, act.wz, act.stop

                update_base_kinematics(model, data, vx, vy, wz, dt, stop)
                mujoco.mj_step(model, data)

                t_in_step += dt
                t_last += dt

                if t_in_step >= dur:
                    step_i = (step_i + 1) % len(SCRIPT)
                    t_in_step = 0.0
                
                print(f"Step {step_i}: Action={name}, dur={dur:.2f}s, t_in_step={t_in_step:.2f}s")

            v.sync()

if __name__ == "__main__":
    main()