from pathlib import Path
import mujoco

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "sim" / "go1_scene.xml"

def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)

    import mujoco.viewer as viewer
    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_step(model, data)

if __name__ == "__main__":
    main()