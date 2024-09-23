import mujoco
from mujoco_mpc import agent as agent_lib
import pathlib

# Cartpole model
model_path = (
    pathlib.Path(__file__).parent.parent.parent
    / "../../build/mjpc/tasks/quadrotor/task.xml"
)
model = mujoco.MjModel.from_xml_path(str(model_path))

# Run GUI
with agent_lib.Agent(
    server_binary_path=pathlib.Path(agent_lib.__file__).parent
    / "mjpc"
    / "ui_agent_server",
    task_id="Quadrotor",
    model=model,
) as agent:
  while True:
    pass
