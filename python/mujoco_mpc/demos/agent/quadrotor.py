import mujoco
import numpy as np
import pathlib
from mujoco_mpc import agent as agent_lib
import matplotlib.pyplot as plt
import mediapy as media

# Load the MuJoCo environment (assuming you have a quadrotor model XML file)
model_path = (
    pathlib.Path(__file__).parent.parent.parent
    / "../../build/mjpc/tasks/quadrotor/task.xml"
)
model = mujoco.MjModel.from_xml_path(str(model_path))

# Create data object for the simulation
data = mujoco.MjData(model)

# Run GUI
with agent_lib.Agent(
    server_binary_path=pathlib.Path(agent_lib.__file__).parent
    / "mjpc"
    / "ui_agent_server",
    task_id="Quadrotor",
    model=model,
) as agent:

    # Renderer for visualization
    renderer = mujoco.Renderer(model)

    # Set the correct cost weights (lin vel, ang vel)
    agent.set_cost_weights({"Lin. Vel.": 0.15})  
    agent.set_cost_weights({"Ang. Vel.": 0.15 })
    print("Cost weights:", agent.get_cost_weights())
    # how to set task param

    # Rollout horizon
    T = 1500

    # Initialize arrays to store the simulation data
    qpos = np.zeros((model.nq, T))
    qvel = np.zeros((model.nv, T))
    ctrl = np.zeros((model.nu, T - 1))
    time = np.zeros(T)

    # Costs
    cost_total = np.zeros(T - 1)
    cost_terms = np.zeros((len(agent.get_cost_term_values()), T - 1))

    # Reset the simulation data
    mujoco.mj_resetData(model, data)

    # Cache initial state
    qpos[:, 0] = data.qpos
    qvel[:, 0] = data.qvel
    time[0] = data.time

    # Initialize a list to store frames for video visualization
    frames = []
    FPS = 1.0 / model.opt.timestep

    # Simulation loop
    for t in range(T - 1):
        if t % 100 == 0:
            print("t =", t)

        # Set the planner state
        agent.set_state(
            time=data.time,
            qpos=data.qpos,
            qvel=data.qvel,
            act=data.act,
            mocap_pos=data.mocap_pos,
            mocap_quat=data.mocap_quat,
            userdata=data.userdata,
        )

        # Run the planner for multiple steps
        num_steps = 10
        for _ in range(num_steps):
            agent.planner_step()

        # Get the action from the agent and apply it as control
        data.ctrl = agent.get_action()
        ctrl[:, t] = data.ctrl
        # Get and store costs
        cost_total[t] = agent.get_total_cost()
        for i, c in enumerate(agent.get_cost_term_values().items()):
            cost_terms[i, t] = c[1]

        # Step 
        mujoco.mj_step(model, data)

        # Cache
        qpos[:, t + 1] = data.qpos
        qvel[:, t + 1] = data.qvel
        time[t + 1] = data.time

        # Update renderer and save frames for visualization
        renderer.update_scene(data)
        pixels = renderer.render()
        frames.append(pixels)

    # Reset agent
    agent.reset()

    # Properly delete the renderer to free resources
    del renderer

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
