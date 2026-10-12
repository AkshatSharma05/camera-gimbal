import mujoco
import mujoco.viewer
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENE_PATH = PROJECT_ROOT / "models" / "onshape-export" / "scene.xml"
DISPLAY_SCRIPT = Path(__file__).with_name("opencv_display.py")

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
data = mujoco.MjData(model)

renderer = mujoco.Renderer(model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)

camera_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_CAMERA, "gimbal_camera"
)

# Find the mocap target's index
target_body_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "mocap_target"
)
mocap_id = model.body_mocapid[target_body_id]

if mocap_id == -1:
    raise ValueError("mocap_target is not a mocap body")

# Random waypoint generator
rng = np.random.default_rng()

def random_waypoint():
    return np.array([
        rng.uniform(-2.2, 2.2),  # x
        rng.uniform(-2.2, 2.2),  # y
        rng.uniform(0.4, 2.5),   # z
    ])

target_position = data.mocap_pos[mocap_id].copy()
waypoint = random_waypoint()
target_speed = 0.8  # metres per second

display = subprocess.Popen(
    [
        sys.executable,
        str(DISPLAY_SCRIPT),
        str(IMAGE_WIDTH),
        str(IMAGE_HEIGHT),
    ],
    stdin=subprocess.PIPE,
)

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if display.poll() is not None:
                if display.returncode != 0:
                    raise subprocess.CalledProcessError(display.returncode, display.args)
                break

            # Move toward the current waypoint
            delta = waypoint - target_position
            distance = np.linalg.norm(delta)

            if distance < 0.05:
                waypoint = random_waypoint()
            else:
                step_distance = min(
                    target_speed * model.opt.timestep,
                    distance,
                )
                target_position += (delta / distance) * step_distance

            # Update the mocap body's world position
            data.mocap_pos[mocap_id] = target_position
            
            mujoco.mj_step(model, data)

            renderer.update_scene(data, camera=camera_id)
            image = renderer.render()
            viewer.sync()

            try:
                display.stdin.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR).tobytes())
                display.stdin.flush()
            except BrokenPipeError:
                if display.poll() is None:
                    raise
                if display.returncode != 0:
                    raise subprocess.CalledProcessError(display.returncode, display.args)
                break
finally:
    if display.stdin is not None:
        try:
            display.stdin.close()
        except BrokenPipeError:
            pass
    display_returncode = display.wait()
    renderer.close()
    if display_returncode != 0 and sys.exc_info()[0] is None:
        raise subprocess.CalledProcessError(display_returncode, display.args)