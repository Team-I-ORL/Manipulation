# Trajectory Publisher and Isaac Sim ROS2 Nodes

## Overview

This repository contains two main ROS2 nodes: 

1. **Trajectory Publisher Node**: `curobo_ros.py`
2. **Isaac Sim ROS2 Node**: `fetch_sim.py`

These nodes are designed to interface with **Isaac Sim** and a **ROS2-based** robotics pipeline. The main purpose is to simulate and execute motion planning trajectories using Curobo within Isaac Sim.

## Usage

To run these scripts, use the **Isaac Sim Python Shell** (`omni_python`). 

### Example:
```bash
omni_python curobo_ros.py
omni_python fetch_sim.py
```

## Pipeline

### `curobo_ros.py` - Trajectory Publisher Node
This script spawns a ROS2 node that:

1. **Subscribes** to the `/joint_states` ROS2 topic to receive current joint states.
2. **Publishes** to the `/joint_trajectory` ROS2 topic to send planned joint trajectories.
3. **Subscribes** to the `/suction_status` ROS2 topic to monitor the suction state of the robot's end-effector.
4. Exposes a **service** at `/target_pose` to receive requests to plan a trajectory to a specified target pose.

Curobo is initialized when the program starts. Upon receiving a service call to `/target_pose`, Curobo generates a motion plan from the robot's current joint state (from /joint_states topic) to the requested goal pose.

Requires /joint_states to be published either through an actual robot or dummy publisher.

### `fetch_sim.py` - Isaac Sim ROS2 Node
This script spawns a different node that:

- **Subscribes** to the `/joint_trajectory` topic.
- Simulates the arm's motion within Isaac Sim according to the received trajectory.

## TODO

1. **Integrate `/suction_status` topic with motion planning pipeline**:
   - If suction status is "OFF", the pose isn't constrained.
   - If suction status is "ON", the pose is constrained.
   
2. **Remodel the mobile base**:
   - Current Curobo model is too conservative with its cuboid representation. Consider refining the base model for more accurate simulation.