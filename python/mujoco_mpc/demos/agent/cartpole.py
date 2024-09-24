# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %%
import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import numpy as np
import pathlib

# set current directory: mujoco_mpc/python/mujoco_mpc
from mujoco_mpc import agent as agent_lib

# %matplotlib inline

# %%
# model
model_path = (
    pathlib.Path(__file__).parent.parent.parent
    / "../../build/mjpc/tasks/cartpole/task.xml"
)
model = mujoco.MjModel.from_xml_path(str(model_path))

# data
data = mujoco.MjData(model)


with agent_lib.Agent(
    server_binary_path=pathlib.Path(agent_lib.__file__).parent
    / "mjpc"
    / "ui_agent_server",
    task_id="Cartpole",
    model=model,
) as agent:

  # renderer
  renderer = mujoco.Renderer(model)
  # %%
  # agent
  # agent = agent_lib.Agent(task_id="Cartpole", model=model)

  # weights
  agent.set_cost_weights({"Velocity": 0.15})
  print("Cost weights:", agent.get_cost_weights())

  # parameters
  agent.set_task_parameter("Goal", -1.0)
  print("Parameters:", agent.get_task_parameters())

  # %%
  # rollout horizon
  T = 100

  # trajectories
  qpos = np.zeros((model.nq, T))
  qvel = np.zeros((model.nv, T))
  ctrl = np.zeros((model.nu, T - 1))
  time = np.zeros(T)

  # costs
  cost_total = np.zeros(T - 1)
  cost_terms = np.zeros((len(agent.get_cost_term_values()), T - 1))

  # rollout
  mujoco.mj_resetData(model, data)

  # cache initial state
  qpos[:, 0] = data.qpos
  qvel[:, 0] = data.qvel
  time[0] = data.time

  # frames
  frames = []
  FPS = 1.0 / model.opt.timestep
  data_collection = {
      'x': [],
      'v': [],
      'theta': [],
      'omega': [],
      'F': [],
      'time': []
  }
  # Initialize RLS parameters
  n_params = 4
  theta_hat = np.zeros(n_params)
  P = np.eye(n_params) * 1000.0
  lambda_factor = 0.99

  # simulate
  for t in range(T - 1):
    if t % 100 == 0:
      print("t = ", t)

    # set planner state
    agent.set_state(
        time=data.time,
        qpos=data.qpos,
        qvel=data.qvel,
        act=data.act,
        mocap_pos=data.mocap_pos,
        mocap_quat=data.mocap_quat,
        userdata=data.userdata,
    )

    # run planner for num_steps
    num_steps = 10
    for _ in range(num_steps):
      agent.planner_step()

    # set ctrl from agent policy
    data.ctrl = agent.get_action()
    ctrl[:, t] = data.ctrl
    
    # get costs
    cost_total[t] = agent.get_total_cost()
    for i, c in enumerate(agent.get_cost_term_values().items()):
      cost_terms[i, t] = c[1]

    # step
    mujoco.mj_step(model, data)

    # Collect data for system identification
    x_k = data.qpos[0]
    theta_k = data.qpos[1]
    v_k = data.qvel[0]
    omega_k = data.qvel[1]
    F_k = data.ctrl[0]
    time_k = data.time

    # Store data for finite differences (need previous values)
    if t > 0:
        # Compute accelerations using finite differences
        dt = time_k - data_collection['time'][-1]
        dot_v_k = (v_k - data_collection['v'][-1]) / dt
        dot_omega_k = (omega_k - data_collection['omega'][-1]) / dt

        # Form regression vectors
        g = -model.opt.gravity[2]  # Positive gravity magnitude

        # Equation 1
        phi_1k = np.array([
            dot_v_k,
            g * theta_k,
            -omega_k**2 * theta_k,
            0.0
        ])
        y_1k = F_k

        # Equation 2
        phi_2k = np.array([
            -g * theta_k,
            -g * theta_k,
            omega_k**2 * theta_k,
            dot_omega_k
        ])
        y_2k = -F_k

        # Stack the regression vectors and outputs
        phi_k = np.vstack([phi_1k, phi_2k])
        y_k = np.array([y_1k, y_2k])

        # RLS update for each equation
        for i in range(2):  # For each equation
            phi_i = phi_k[i].reshape(-1, 1)
            y_i = y_k[i]

            # Compute the gain vector
            K = (P @ phi_i) / (lambda_factor + phi_i.T @ P @ phi_i)

            # Update the parameter estimate
            theta_hat = theta_hat + (K.flatten() * (y_i - phi_i.T @ theta_hat))

            # Update the covariance matrix
            P = (P - K @ phi_i.T @ P) / lambda_factor

    # Append data to the collection
    data_collection['x'].append(x_k)
    data_collection['theta'].append(theta_k)
    data_collection['v'].append(v_k)
    data_collection['omega'].append(omega_k)
    data_collection['F'].append(F_k)
    data_collection['time'].append(time_k)

    # cache
    qpos[:, t + 1] = data.qpos
    qvel[:, t + 1] = data.qvel
    time[t + 1] = data.time

    # render and save frames
    renderer.update_scene(data)
    pixels = renderer.render()
    frames.append(pixels)
  import numpy as np
  from numpy.linalg import lstsq
  # After simulation loop
  m_c_est = theta_hat[0]
  m_p_est = theta_hat[1]
  m_p_l_est = theta_hat[2]
  l_m_c_est = theta_hat[3]
  # Estimate l from m_p_l_est and m_p_est
  if m_p_est != 0:
      l_est_from_mp = m_p_l_est / m_p_est
  else:
      l_est_from_mp = np.nan

  # Estimate l from l_m_c_est and m_c_est
  if m_c_est != 0:
      l_est_from_mc = l_m_c_est / m_c_est
  else:
      l_est_from_mc = np.nan

  # Final estimate of l
  if not np.isnan(l_est_from_mp) and not np.isnan(l_est_from_mc):
      l_estimate_final = (l_est_from_mp + l_est_from_mc) / 2
  elif not np.isnan(l_est_from_mp):
      l_estimate_final = l_est_from_mp
  else:
      l_estimate_final = l_est_from_mc
  print(f"Estimated m_c: {m_c_est}")#, True m_c: {true_m_c}")
  print(f"Estimated m_p: {m_p_est}")#, True m_p: {true_m_p}")
  print(f"Estimated l: {l_estimate_final}")#, True l: {true_l}")
  # # Update the model parameters
  # model.body_mass[model.body_name2id('cart')] = m_c_est
  # model.body_mass[model.body_name2id('pole')] = m_p_est

  # # Update the length of the pole
  # pole_geom_id = model.geom_name2id('pole_geom')
  # model.geom_size[pole_geom_id][1] = l_estimate_final  # Update half-length

  # # Recompile the model if necessary
  # model = mujoco.MjModel.from_xml_path(str(model_path))
  # data = mujoco.MjData(model)

  # # Verify the estimates
  # true_m_c = model.body_mass[model.body_name2id('cart')]
  # true_m_p = model.body_mass[model.body_name2id('pole')]
  # true_l = model.geom_size[pole_geom_id][1]

  # print(f"Estimated m_c: {m_c_est}, True m_c: {true_m_c}")
  # print(f"Estimated m_p: {m_p_est}, True m_p: {true_m_p}")
  # print(f"Estimated l: {l_estimate_final}, True l: {true_l}")

  # reset
  agent.reset()

  # display video
  SLOWDOWN = 0.5
  media.show_video(frames, fps=SLOWDOWN * FPS)
