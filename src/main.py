import mujoco
import mujoco.viewer
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
from control_loop import VisionTargetController

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENE_PATH = PROJECT_ROOT / "models" / "onshape-export" / "scene.xml"
DISPLAY_SCRIPT = Path(__file__).with_name("opencv_display.py")

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)

camera_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_CAMERA, "gimbal_camera"
)

controller = VisionTargetController(IMAGE_WIDTH, IMAGE_HEIGHT)
mocap_id = model.body_mocapid[
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mocap_target")
]
if mocap_id < 0:
    raise ValueError("mocap_target must be a mocap body")

actuator_ids = {
    name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    for name in ("yaw", "pitch")
}

rng = np.random.default_rng()


def random_waypoint():
    return np.array([
        rng.uniform(-2.2, 2.2),
        rng.uniform(-2.2, 2.2),
        rng.uniform(0.4, 1.5),
    ])


target_position = data.mocap_pos[mocap_id].copy()
waypoint = random_waypoint()
target_speed = 0.8

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

            delta = waypoint - target_position
            distance = np.linalg.norm(delta)
            if distance < 0.05:
                waypoint = random_waypoint()
            else:
                step_distance = min(target_speed * model.opt.timestep, distance)
                target_position += (delta / distance) * step_distance

            data.mocap_pos[mocap_id] = target_position

            mujoco.mj_forward(model, data)

            renderer.update_scene(data, camera=camera_id)
            image = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

            commands = controller.update(image, model.opt.timestep)
            if commands is not None:
                for joint_name, actuator_id in actuator_ids.items():
                    desired = data.ctrl[actuator_id] + commands[joint_name] * model.opt.timestep
                    if model.actuator_ctrllimited[actuator_id]:
                        lower, upper = model.actuator_ctrlrange[actuator_id]
                        desired = np.clip(desired, lower, upper)
                    data.ctrl[actuator_id] = desired

            mujoco.mj_step(model, data)
            viewer.sync()

            try:
                display.stdin.write(image.tobytes())
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