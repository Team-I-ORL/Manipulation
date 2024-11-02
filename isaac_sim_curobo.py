from omni.isaac.core import World
from omni.isaac.core.utils.types import ArticulationAction
from helper import add_extensions, add_robot_to_scene
from curobo.util.usd_helper import UsdHelper
import numpy as np

def run_simulation(simulation_app, shared_data):
    with shared_data.lock:
        world_cfg = shared_data.world_cfg
        robot_cfg = shared_data.robot_cfg
        j_names = shared_data.j_names

    sim_world = World(stage_units_in_meters=1.0)
    add_extensions(simulation_app)
    stage = sim_world.stage
    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")

    usd_help = UsdHelper()

    usd_help.load_stage(sim_world.stage)
    usd_help.add_world_to_stage(world_cfg, base_frame="/World")
    robot, robot_prim_path = add_robot_to_scene(robot_cfg, sim_world)
    articulation_controller = robot.get_articulation_controller()

    while simulation_app.is_running():
        with shared_data.lock:
            solutions = shared_data.published_trajectory
            q_start = shared_data.start_js
        if solutions or q_start is None:
            continue
        else:
            print("Waiting for trajectory to be published")

        if not sim_world.is_playing():
            print("Click Play")
            continue
        step_index = sim_world.current_time_step_index
        if step_index < 2:
            sim_world.reset()
            robot._articulation_view.initialize()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(q_start, idx_list)

        if step_index < 20:
            continue

        for point in solutions.points:
            positions = point.positions
            articulation_action = ArticulationAction(joint_positions=positions)
            if isinstance(articulation_action.joint_positions, list) and len(articulation_action.joint_positions) == 7:
                articulation_controller.apply_action(articulation_action)
            else:
                continue
            for _ in range(10):
                sim_world.step(render=True)

        with shared_data.lock:
            shared_data.published_trajectory = None
            shared_data.start_js = None

        simulation_app.update()
    simulation_app.close()