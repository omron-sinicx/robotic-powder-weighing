#!/usr/bin/env python3

import math
import random
# import time
import numpy as np
from numpy import deg2rad, rad2deg, pi
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
import pickle
import socket
import select
import rospy
from control_msgs.msg import JointTrajectoryControllerState
from std_msgs.msg import Float64

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

rospy.init_node('isaacgym_ur')

def updateCameraCoordinates():
    camera_positions = [gymapi.Vec3(gym.get_actor_rigid_body_states(env,ur5_handle, gymapi.STATE_POS)[9][0][0][0]+0.15,
                                    gym.get_actor_rigid_body_states(env,ur5_handle, gymapi.STATE_POS)[9][0][0][1]+0.1,
                                    gym.get_actor_rigid_body_states(env,ur5_handle, gymapi.STATE_POS)[9][0][0][2]-0.384),
                        gymapi.Vec3(gym.get_actor_rigid_body_states(env,ur5_handle, gymapi.STATE_POS)[9][0][0][0]+0.15,
                                    gym.get_actor_rigid_body_states(env,ur5_handle, gymapi.STATE_POS)[9][0][0][1]+0.3,
                                    gym.get_actor_rigid_body_states(env,ur5_handle, gymapi.STATE_POS)[9][0][0][2]-0.01)
                       ]
    camera_targets = [gymapi.Vec3(gym.get_actor_rigid_body_states(env,ur5_handle, gymapi.STATE_POS)[9][0][0][0]+0.15,
                                  gym.get_actor_rigid_body_states(env,ur5_handle, gymapi.STATE_POS)[9][0][0][1]+0.1,
                                  0),
                      gymapi.Vec3(gym.get_actor_rigid_body_states(env,ur5_handle, gymapi.STATE_POS)[9][0][0][0]+0.15,
                                  0,
                                  gym.get_actor_rigid_body_states(env,ur5_handle, gymapi.STATE_POS)[9][0][0][2]-0.009)
                     ]
    
    return camera_positions, camera_targets

# Get 3DSystems Touch input
ur_state = JointTrajectoryControllerState()

def callbackURstate(data):
    global ur_state
    ur_state = data

forcePub = rospy.Publisher('/touch_state/force_feedback', Float64, queue_size=5)
rospy.Subscriber('/arm_controller/state', JointTrajectoryControllerState, callbackURstate)

# Initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(
    description="UR5 Pan Flipper",
    custom_parameters=[
        {"name": "--num_envs",
        "type": int, "default": 1,
        "help": "Number of environments to create"},
    ])

# Camera settings:
# 0 = Free
# 1 = Fixed
# 2 = Follow
cam_config = 0
MAX_BALLS = 200

sim_params = gymapi.SimParams()
if args.physics_engine == gymapi.SIM_FLEX:
    # solver parameters
    sim_params.flex.solver_type = 5
    # collision parameters
    sim_params.flex.shape_collision_margin = 0.1
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 10
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.use_gpu = True
    # solver parameters
    sim_params.physx.solver_type = 1
    # collision parameters
    sim_params.physx.num_position_iterations = 5
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = 8
    sim_params.physx.friction_offset_threshold = 0.0
    sim_params.physx.friction_correlation_distance = 0.1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.000001

sim_params.dt = 1 / 100.0
sim_params.substeps = 2
# sim_params.gravity = gymapi.Vec3(0,-0.5,0)

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# create viewer
cam_properties = gymapi.CameraProperties()
cam_properties.use_collision_geometry = False
# cam_properties.horizontal_fov = 1
viewer = gym.create_viewer(sim, cam_properties)
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# subscribe to spacebar event for reset
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_C, "switch_camera")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_G, "generate_ball")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "quit")

# load asset
asset_root = "/home/edgar/rllab/ros_workspaces/touch_ur_ws/src/touch_to_ur/scripts/ur5_pan_flipper/assets"
asset_files = ["urdf/ur5_description/ur5_joint_limited_robot.urdf",
               "urdf/test_obj_description/frying_pan.urdf",
               "urdf/test_obj_description/ball.urdf"]
asset_names = ["ur5_with_pan",
               "frying_pan",
               "ball"]

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True  # Fix base in place
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
asset_options.flip_visual_attachments = True
asset_options.use_mesh_materials = True
asset_options.vhacd_enabled = True
asset_options.vhacd_params.resolution = 800000
asset_options.vhacd_params.max_convex_hulls = 80000
asset_options.vhacd_params.convex_hull_downsampling = 4
asset_options.vhacd_params.max_num_vertices_per_ch = 1024
asset_options.vhacd_params.min_volume_per_ch = 0.0

# Don't modify
# asset_options.vhacd_params.plane_downsampling = 1
# asset_options.vhacd_params.alpha = 0.5
# asset_options.vhacd_params.beta = 0.5
# asset_options.vhacd_params.mode = 0
# asset_options.vhacd_params.pca = 1
ur5_asset = gym.load_asset(sim, asset_root, asset_files[0], asset_options)
# print(gym.get_asset_dof_properties(ur5_asset))

asset_options = gymapi.AssetOptions()
ball_asset = gym.load_asset(sim, asset_root, asset_files[2], asset_options)

num_envs = args.num_envs
num_per_row = int(math.sqrt(num_envs))
SPACING = 0.75
env_lower = gymapi.Vec3(-SPACING, 0.0, -SPACING)
env_upper = gymapi.Vec3(SPACING, SPACING, SPACING)
envs = []

ur5_handles = []
frying_pan_handles = []
ball_handles = []

for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    ur5_pose = gymapi.Transform()
    ur5_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
    ur5_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
    ur5_handle = gym.create_actor(env, ur5_asset, ur5_pose, asset_names[0], 1, 0)
    gym.enable_actor_dof_force_sensors(env, ur5_handle)
    # gym.set_actor_scale(env, ur5_handle, 10)
    # Set UR5 initial position
    ur5_init_pos = [0., -0.6870, 1.0769, -0.3899, 0., -1.2082]
    gym.set_actor_dof_states(env,ur5_handle, ur5_init_pos, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env,ur5_handle, ur5_init_pos)
    ur5_handles.append(ur5_handle)

#Initial camera coordinates
camera_state = 0
camera_positions, camera_targets = updateCameraCoordinates()

camera_pos = camera_positions[camera_state]
camera_target = camera_targets[camera_state]
gym.viewer_camera_look_at(viewer, None, camera_pos, camera_target)

# light_intensity = 1
# ambient_intensity = 1
# gym.set_light_parameters(sim, 0, gymapi.Vec3(light_intensity, light_intensity, light_intensity), gymapi.Vec3(ambient_intensity, ambient_intensity, ambient_intensity), gymapi.Vec3(0, 0, 1))

ur5_num_dofs = gym.get_asset_dof_count(ur5_asset)
default_dof_pos = np.zeros(ur5_num_dofs, dtype=np.float32)
default_dof_pos[1] = -math.pi/2
# create a local copy of initial state, which we can send back for reset
for env in envs:
    if len(ur5_handles)>0:
        for ur5_handle in ur5_handles:
            # gym.set_actor_dof_position_targets(env, ur5_handle, default_dof_pos)
            initial_rigid_ur5 = np.copy(gym.get_actor_rigid_body_states(env,ur5_handle, gymapi.STATE_ALL))
            initial_dof_ur5 = np.copy(gym.get_actor_dof_states(env,ur5_handle, gymapi.STATE_ALL))
            initial_pos_target_ur5 = np.copy(gym.get_actor_dof_position_targets(env,ur5_handle))
            initial_vel_target_ur5 = np.copy(gym.get_actor_dof_velocity_targets(env,ur5_handle))

cur_time = 0.0
DISTANCE = 0.05
ROUND = 2
generate_ball_toggle = False
ball_pose = gymapi.Transform()
ball_pose.p = gymapi.Vec3(0.765, 0.287, -0.191)

while not gym.query_viewer_has_closed(viewer):

    # print(gym.get_actor_rigid_body_states(env,ur5_handle, gymapi.STATE_POS)[9][0][0])
    # print(gym.get_camera_transform(sim, env, viewer))
    # print(round(gym.get_sim_time(sim),ROUND), cur_time)
    # print()
    # print(ur_state.joint_names)
    # print(ur_state.actual.positions)
    # print(ur_state.actual.velocities)

    for env in range(len(envs)):
        gym.set_actor_dof_position_targets(envs[env],ur5_handles[env], ur_state.actual.positions)
        forces = gym.get_actor_dof_forces(envs[env], ur5_handles[env])
        # print(round(forces[5],3))
        force = Float64()
        force.data = round(forces[5],3)
        forcePub.publish(force)

    if generate_ball_toggle and round(gym.get_sim_time(sim)-DISTANCE,ROUND) == cur_time and len(ball_handles) < MAX_BALLS:
        for env in envs:
            cur_time = round(gym.get_sim_time(sim),ROUND)
            ball_handle = gym.create_actor(env, ball_asset, ball_pose, asset_names[2], 1, 0)
            color = gymapi.Vec3(0, 1, 0)
            gym.set_rigid_body_color(env, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            ball_handles.append(ball_handle)
            print(len(ball_handles),"balls")
    
    if generate_ball_toggle and len(ball_handles) >= MAX_BALLS:
        print("Ball limit ("+str(MAX_BALLS)+" balls)")
        generate_ball_toggle = 0

    # Get input actions from the viewer and handle them appropriately
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
            if len(ur5_handles)>0:
                for ur5_handle in ur5_handles:
                    gym.set_actor_rigid_body_states(env, ur5_handle, initial_rigid_ur5, gymapi.STATE_ALL)
                    gym.set_actor_dof_states(env, ur5_handle, initial_dof_ur5, gymapi.STATE_ALL)
                    gym.set_actor_dof_position_targets(env,ur5_handle, initial_pos_target_ur5)
                    gym.set_actor_dof_velocity_targets(env,ur5_handle, initial_vel_target_ur5)
            try:
                ball_states
            except NameError:
                pass
            else:
                for ball_state in ball_states:
                    gym.set_actor_rigid_body_states(env,ball_state[0],ball_state[1],gymapi.STATE_ALL)
        
        if evt.action == "switch_camera" and evt.value > 0:
            camera_state = (camera_state + 1)%len(camera_positions)
            # print(camera_state)

            camera_pos = camera_positions[camera_state]
            camera_target = camera_targets[camera_state]
            gym.viewer_camera_look_at(viewer, None, camera_pos, camera_target)
            print("Changed to", camera_state)

        if evt.action =="generate_ball" and evt.value > 0:
            if generate_ball_toggle:
                generate_ball_toggle = 0
            else:
                cur_time = round(gym.get_sim_time(sim),ROUND)
                generate_ball_toggle = 1

        if evt.action == "quit" and evt.value > 0:
            gym.destroy_viewer(viewer)

    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Camera position and target
    if cam_config == 1:
        camera_pos = camera_positions[camera_state]
        camera_target = camera_targets[camera_state]
        gym.viewer_camera_look_at(viewer, None, camera_pos, camera_target)
    elif cam_config == 2:
        camera_positions, camera_targets = updateCameraCoordinates()
        camera_pos = camera_positions[camera_state]
        camera_target = camera_targets[camera_state]
        gym.viewer_camera_look_at(viewer, None, camera_pos, camera_target)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)
    
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)