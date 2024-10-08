import mujoco
import numpy as np
import pathlib
from mujoco_mpc import agent as agent_lib
import matplotlib.pyplot as plt
import mediapy as media

# Load the MuJoCo environment (assuming you have a quadrotor model XML file)
model_path_quad = (
pathlib.Path(__file__).parent.parent.parent
/ "../../build/mjpc/tasks/quadrotor/task.xml"
)
model_path_cart = (
pathlib.Path(__file__).parent.parent.parent
/ "../../build/mjpc/tasks/cartpole/task.xml"
)

model_mpc = mujoco.MjModel.from_xml_path(str(model_path_quad))
model_sim = mujoco.MjModel.from_xml_path(str(model_path_quad))

# Create data object for the simulation
data_controller = mujoco.MjData(model_mpc)  # For MPC controller
data_simulation = mujoco.MjData(model_sim)  # For the actual simulation

agent_mpc = agent_lib.Agent(task_id="Quadrotor", model=model_mpc)
agent_sim = agent_lib.Agent(task_id="Quadrotor", model=model_sim)


# Renderer for visualization
# renderer = mujoco.Renderer(model)

# Set the correct cost weights (lin vel, ang vel)
agent_mpc.set_cost_weights({"Lin. Vel.": 0.15})  
agent_mpc.set_cost_weights({"Ang. Vel.": 0.15 })
agent_sim.set_cost_weights({"Lin. Vel.": 0.15})  
agent_sim.set_cost_weights({"Ang. Vel.": 0.15 })
print("Cost weights:", agent_mpc.get_cost_weights())
# how to set task param

# Rollout horizon
T = 1500

# Initialize arrays to store the simulation data
qpos = np.zeros((model_sim.nq, T))
qvel = np.zeros((model_sim.nv, T))
ctrl = np.zeros((model_sim.nu, T - 1))
time = np.zeros(T)

# Costs
cost_total = np.zeros(T - 1)
cost_terms = np.zeros((len(agent_mpc.get_cost_term_values()), T - 1))

# Reset the simulation data
mujoco.mj_resetData(model_mpc, data_controller)
mujoco.mj_resetData(model_sim, data_simulation)

# Cache initial state
qpos[:, 0] = data_simulation.qpos
qvel[:, 0] = data_simulation.qvel
time[0] = data_simulation.time

# Initialize a list to store frames for video visualization
frames = []
FPS = 1.0 / model_sim.opt.timestep

# Simulation loop
for t in range(T - 1):
    if t % 100 == 0:
        print("t =", t)

    # Set the planner state
    agent_mpc.set_state(
        time=data_simulation.time,
        qpos=data_simulation.qpos,
        qvel=data_simulation.qvel,
        act=data_simulation.act,
        mocap_pos=data_simulation.mocap_pos,
        mocap_quat=data_simulation.mocap_quat,
        userdata=data_simulation.userdata,
    )

    agent_sim.set_state(
        time=data_simulation.time,
        qpos=data_simulation.qpos,
        qvel=data_simulation.qvel,
        act=data_simulation.act,
        mocap_pos=data_simulation.mocap_pos,
        mocap_quat=data_simulation.mocap_quat,
        userdata=data_simulation.userdata,
    )

    # Run the planner for multiple steps
    num_steps = 10
    for _ in range(num_steps):
        agent_mpc.planner_step()
        agent_sim.planner_step()

    # Get the action from the agent and apply it as control
    data_simulation.ctrl = agent_mpc.get_action()
    ctrl[:, t] = data_simulation.ctrl
    # Get and store costs
    cost_total[t] = agent_sim.get_total_cost()
    for i, c in enumerate(agent_sim.get_cost_term_values().items()):
        cost_terms[i, t] = c[1]

    if t == 100:    
        model_sim.body_mass *= 100
        print("mass *= 100 at time step 100")


    # Step 
    mujoco.mj_step(model_mpc, data_controller)
    mujoco.mj_step(model_sim, data_simulation)

    # Cache
    qpos[:, t + 1] = data_simulation.qpos
    qvel[:, t + 1] = data_simulation.qvel
    time[t + 1] = data_simulation.time

    if t % 20 == 0: 
        print("t = ", t, qpos[:, t])
        # print(qvel[:, t])
        # print(cost_total.max())
        # print(cost_total.min())

    # # Update renderer and save frames for visualization
    # renderer.update_scene(data)
    # pixels = renderer.render()
    # frames.append(pixels)

# Reset agent
agent_mpc.reset()
agent_sim.reset()


#display video
SLOWDOWN = 0.5
media.show_video(frames, fps=SLOWDOWN * FPS)

# Plot 
plt.figure()
plt.plot(time, qpos[0, :], label="q0", color="blue")
plt.plot(time, qpos[1, :], label="q1", color="orange")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Configuration")
plt.show()

# Plot velocity
plt.figure()
plt.plot(time, qvel[0, :], label="v0", color="blue")
plt.plot(time, qvel[1, :], label="v1", color="orange")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Velocity")
plt.show()

# Plot control
plt.figure()
plt.plot(time[:-1], ctrl[0, :], color="blue")
plt.xlabel("Time (s)")
plt.ylabel("Control")
plt.show()

# Plot costs
plt.figure()
for i, c in enumerate(agent.get_cost_term_values().items()):
    plt.plot(time[:-1], cost_terms[i, :], label=c[0])
plt.plot(time[:-1], cost_total, label="Total (weighted)", color="black")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Costs")
plt.show()
