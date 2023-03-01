import numpy as np
from isaacgym import gymutil
from isaacgym import gymapi
import os
import torch
import gc
import cv2

from .WeighingDomainInfo import WeighingDomainInfo
from .BallGenerator import BallGenerator


class WeighingEnvironment:
    def __init__(self, userDefinedSettings=None, domain_range=None, create_env_flag=True, parameters=None):
        if userDefinedSettings is not None:
            self.userDefinedSettings = userDefinedSettings
            self.render_flag = userDefinedSettings.RENDER_FLAG
            self.save_image_flag = userDefinedSettings.save_image

        self.assets_path = "./Environment/Weighing/"
        self.urdf_dir = 'Environment/Weighing/assets/urdf/spoon/'
        self.between_ball_space = 0.03
        self.initial_height_ball_pyramid = 2.5
        self.initial_height_spoon = 2
        self.spoon_size_rate = 0.05
        self.joy_speed_scale = 1.
        self.ball_ground_threshold_height = self.initial_height_spoon - 0.5
        self.action_loop_num_per_step = 60
        self.world_size_rate = 2.
        self.mass_gap_sim_rate = 0.5

        self.gym_package_args = gymutil.parse_arguments(
            description="Collision Filtering: Demonstrates filtering of collisions within and between environments",
            custom_parameters=[
                {"name": "--num_envs", "type": int, "default": 36, "help": "Number of environments to create", },
                {"name": "--all_collisions", "action": "store_true", "help": "Simulate all collisions", },
                {"name": "--no_collisions", "action": "store_true", "help": "Ignore all collisions", },
                {"name": "--env", "type": str, "help": "Ignore", },
                {"name": "--test", "action": "store_true", "help": "Ignore", },
                {"name": "--render", "action": "store_true", "help": "Ignore", },
                {"name": "--save_image", "action": "store_true", "help": "Ignore", },
                {"name": "--dir", "type": str, "help": "Ignore", },
                {"name": "--notDR", "action": "store_true", "help": "Ignore", },
                {"name": "--flag", "type": list, "help": "Ignore", },
                {"name": "--gpu", "type": str, "help": "Ignore", },
                {"name": "--path", "type": str, "help": "Ignore", },
                {"name": "--episode", "type": str, "help": "Ignore", },
                {"name": "--notLSTM", "type": str, "help": "Ignore", },
                {"name": "--goal", "type": str, "help": "Ignore", },
                {"name": "--top_num", "type": str, "help": "Ignore", }
            ],
        )

        if parameters is not None:
            incline_flag, shake_flag, ball_radius_flag, ball_mass_flag, ball_friction_flag, ball_layer_num_flag, spoon_friction_flag, goal_powder_amount_flag, shake_speed_weight_flag, gravity_flag = [int(i) for i in parameters]
            self.flag_list = [ball_radius_flag, ball_mass_flag, ball_friction_flag, ball_layer_num_flag, spoon_friction_flag, goal_powder_amount_flag, shake_speed_weight_flag, gravity_flag]
            self.incline_flag = bool(incline_flag)
            self.shake_flag = bool(shake_flag)
            print('incline', incline_flag,
                  'shake', shake_flag,
                  'radius', ball_radius_flag,
                  'mass', ball_mass_flag,
                  'b_fri', ball_friction_flag,
                  'layer', ball_layer_num_flag,
                  's_fri', spoon_friction_flag,
                  'goal', goal_powder_amount_flag,
                  'speed', shake_speed_weight_flag,
                  'gravity', gravity_flag)

        else:
            self.incline_flag = True
            self.shake_flag = True
            self.flag_list = None

        self.domainInfo = WeighingDomainInfo(userDefinedSettings=userDefinedSettings, domain_range=domain_range, flag_list=self.flag_list)
        self.DOMAIN_PARAMETER_DIM = self.domainInfo.get_domain_parameter_dim()
        self.MAX_EPISODE_LENGTH = 10
        self.ACTION_MAPPING_FLAG = True
        self.action_type = 'relative'  # absolute or  relative
        self.action_space = Action_space(action_type=self.action_type, shake_flag=self.shake_flag)
        self.STATE_DIM = 3
        self.ACTION_DIM = self.action_space.ACTION_DIM
        self.step_num = 0
        self.image_count = 0
        self.video_count = 0

        if create_env_flag:
            self.ball_radius, self.ball_mass, self.ball_friction, self.ball_layer_num, self.spoon_friction, self.goal_powder_amount, self.shake_speed_weight, self.gravity = self.domainInfo.get_domain_parameters()
            self.create_sim_params()
            self.create_env()

    def reset_world(self, reset_info):
        if self.render_flag:
            self.gym_package.destroy_viewer(self.viewer)
        self.gym_package.destroy_sim(self.sim)
        self.gym_package.destroy_env(self.env)
        del self.gym_package, self.sim_params, self.ball_handle_list, self.spoon, self.sim, self.env,
        gc.collect()
        torch.cuda.empty_cache()
        self.domainInfo.set_parameters(reset_info=reset_info)
        self.ball_radius, self.ball_mass, self.ball_friction, self.ball_layer_num, self.spoon_friction, self.goal_powder_amount, self.shake_speed_weight, self.gravity = self.domainInfo.get_domain_parameters()
        self.create_sim_params()
        self.create_env()
        if self.save_image_flag:
            self.create_render()

    def create_sim_params(self):
        self.sim_params = gymapi.SimParams()
        self.sim_params.dt = 1 / float(self.action_loop_num_per_step)
        self.sim_params.gravity = gymapi.Vec3(0.0, self.gravity, 0.)
        if self.gym_package_args.physics_engine == gymapi.SIM_FLEX:
            self.sim_params.flex.shape_collision_margin = 0.25
            self.sim_params.flex.num_outer_iterations = 4
            self.sim_params.flex.num_inner_iterations = 10
        elif self.gym_package_args.physics_engine == gymapi.SIM_PHYSX:
            self.sim_params.substeps = 1
            self.sim_params.physx.solver_type = 1
            self.sim_params.physx.num_position_iterations = 4
            self.sim_params.physx.num_velocity_iterations = 1
            self.sim_params.physx.num_threads = self.gym_package_args.num_threads
            self.sim_params.physx.use_gpu = self.gym_package_args.use_gpu

            if True:
                self.sim_params.physx.friction_offset_threshold = 0.001
                self.sim_params.physx.friction_correlation_distance = 0.0005
                self.sim_params.physx.contact_offset = 0.001
                self.sim_params.physx.rest_offset = 0.000001

        self.sim_params.use_gpu_pipeline = False

    def create_env(self):
        self.gym_package = gymapi.acquire_gym()

        self.sim = self.gym_package.create_sim(
            self.gym_package_args.compute_device_id,
            self.gym_package_args.graphics_device_id,
            self.gym_package_args.physics_engine,
            self.sim_params,
        )
        if self.sim is None:
            print("*** Failed to create self.sim")
            quit()

        if self.render_flag:
            self.viewer = self.gym_package.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                print("*** Failed to create self.viewer")
                quit()
            self.gym_package.viewer_camera_look_at(
                self.viewer,
                None,
                gymapi.Vec3(3, 3, -3),
                gymapi.Vec3(1.5, 1.5, 5.5),
            )
            self.gym_package.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset")

        num_per_row = 1
        env_spacing = 10
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

        self.env = self.gym_package.create_env(self.sim, env_lower, env_upper, num_per_row)
        self.create_ball()
        self.create_plane()
        self.create_spoon()
        self.change_parameters()
        self.set_actor()

    def create_render(self):
        camera_properties = gymapi.CameraProperties()
        self.image_width = 1920
        self.image_height = 1080
        camera_properties.width = self.image_width
        camera_properties.height = self.image_height

        self.camera_handle = self.gym_package.create_camera_sensor(self.env, camera_properties)
        camera_position = gymapi.Vec3(0.5, 2.5, -0.5)  # yoko,z,oku
        camera_target = gymapi.Vec3(-2.5, 0, 2.5)
        self.gym_package.set_camera_location(self.camera_handle, self.env, camera_position, camera_target)

    def change_parameters(self):
        self.change_spoon_friction()

    def render(self):
        pass

    def set_actor(self):
        self.slider_x = self.gym_package.find_actor_dof_handle(self.env, self.spoon, "bucket_slider_x")
        self.slider_r = self.gym_package.find_actor_dof_handle(self.env, self.spoon, "bucket_joint_r")

    def create_plane(self):
        plane_params = gymapi.PlaneParams()
        self.gym_package.add_ground(self.sim, plane_params)

    def create_ball(self):
        ballGenerator = BallGenerator()
        file_name = 'BallHLS.urdf'
        urdf_path = os.path.join(self.urdf_dir, file_name)
        if not os.path.exists(self.urdf_dir):
            os.makedirs(self.urdf_dir)

        ballGenerator.generate(file_name=urdf_path, ball_radius=self.ball_radius, ball_mass=self.ball_mass)
        asset = self.gym_package.load_asset(self.sim, self.urdf_dir, file_name, gymapi.AssetOptions())

        c = np.array([115, 78, 48]) / 255.0
        color = gymapi.Vec3(c[0], c[1], c[2])

        pose = gymapi.Transform()
        pose.r = gymapi.Quat(0, 0, 0, 1)

        n = int(self.ball_layer_num)
        ball_spacing = self.between_ball_space
        min_coord = -0.5 * (n - 1) * ball_spacing
        y = min_coord + self.initial_height_ball_pyramid

        self.ball_handle_list = []
        while n > 0:
            z = min_coord
            for j in range(n):
                x = min_coord
                for k in range(n):
                    pose.p = gymapi.Vec3(x, y, z)

                    collision_group = 0
                    collision_filter = 0

                    ball_handle = self.gym_package.create_actor(self.env, asset, pose, None, collision_group, collision_filter)

                    body_shape_prop = self.gym_package.get_actor_rigid_shape_properties(self.env, ball_handle)
                    body_shape_prop[0].friction = self.ball_friction
                    self.gym_package.set_actor_rigid_shape_properties(self.env, ball_handle, body_shape_prop)

                    self.ball_handle_list.append(ball_handle)
                    self.gym_package.set_rigid_body_color(self.env, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                    self.gym_package.set_actor_scale(self.env, ball_handle, self.world_size_rate)
                    x += ball_spacing
                z += ball_spacing
            y += ball_spacing
            n -= 1
            min_coord = -0.5 * (n - 1) * ball_spacing

        self.all_ball_num = len(self.ball_handle_list)

    def change_ball_property(self):
        ball_friction = 2.
        ball_restitution = 0.
        ball_rolling_friction = 2.
        ball_torsion_friction = 2.
        for ball_handle in self.ball_handle_list:
            body_shape_prop = self.gym_package.get_actor_rigid_shape_properties(self.env, ball_handle)
            for i in range(1):
                body_shape_prop[i].friction = ball_friction
                body_shape_prop[i].rolling_friction = ball_rolling_friction
                body_shape_prop[i].torsion_friction = ball_torsion_friction
                body_shape_prop[i].friction = ball_friction
                body_shape_prop[i].restitution = ball_restitution
            self.gym_package.set_actor_rigid_shape_properties(self.env, ball_handle, body_shape_prop)

    def get_number_in_spoon(self):
        ball_in_spoon = 0
        for ball_handle in self.ball_handle_list:
            body_states = self.gym_package.get_actor_rigid_body_states(self.env, ball_handle, gymapi.STATE_ALL)
            z = body_states['pose']['p'][0][1]
            if z > self.ball_ground_threshold_height:
                ball_in_spoon += 1
        return ball_in_spoon

    def create_spoon(self):
        file_name = 'spoon.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 300000
        asset_options.vhacd_params.max_convex_hulls = 10
        asset_options.vhacd_params.max_num_vertices_per_ch = 64
        asset = self.gym_package.load_asset(self.sim, self.urdf_dir, file_name, asset_options)
        collision_group = 0
        collision_filter = 0
        c = np.array([150, 150, 150]) / 255.0
        color = gymapi.Vec3(c[0], c[1], c[2])
        pose = gymapi.Transform()
        pose.r = gymapi.Quat(0, 0, 0, 1)
        pose.p = gymapi.Vec3(0, self.initial_height_spoon, 0)  # xzy

        self.spoon = self.gym_package.create_actor(self.env, asset, pose, None, collision_group, collision_filter)
        self.gym_package.set_rigid_body_color(self.env, self.spoon, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        props = self.gym_package.get_actor_dof_properties(self.env, self.spoon)
        self.gym_package.set_actor_scale(self.env, self.spoon, self.spoon_size_rate * self.world_size_rate)

        props["driveMode"] = (
            gymapi.DOF_MODE_POS,
            gymapi.DOF_MODE_POS,
        )
        props["stiffness"] = (
            1000000.,
            1000000.,
        )
        props["damping"] = (
            1000000.,
            1000000.,
        )

        self.gym_package.set_actor_dof_properties(self.env, self.spoon, props)

    def change_spoon_friction(self):
        body_shape_prop = self.gym_package.get_actor_rigid_shape_properties(self.env, self.spoon)
        for i in range(10):
            body_shape_prop[i].friction = self.spoon_friction
        self.gym_package.set_actor_rigid_shape_properties(self.env, self.spoon, body_shape_prop)

    def joy_loop(self):
        import pygame
        from .control_joy import JoyController
        pygame.init()
        self.joy = JoyController(0)

        while not self.gym_package.query_viewer_has_closed(self.viewer):

            # get action
            eventlist = pygame.event.get()
            self.joy.get_controller_value(eventlist)
            x = -self.joy.l_hand_x * 0.3
            r = -self.joy.r_hand_x * 0.4
            action = np.array([x, r])

            # Get input actions from the self.viewer and handle them appropriately
            for evt in self.gym_package.query_viewer_action_events(self.viewer):
                if evt.action == "reset" and evt.value > 0:
                    self.reset()

            # get current pos
            x_pos = self.gym_package.get_dof_position(self.env, self.slider_x)
            r_pos = self.gym_package.get_dof_position(self.env, self.slider_r)

            # move bucket
            self.gym_package.set_dof_target_position(self.env, self.slider_x, x_pos + x)
            self.gym_package.set_dof_target_position(self.env, self.slider_r, r_pos + r)

            # step the physics
            self.gym_package.simulate(self.sim)
            self.gym_package.fetch_results(self.sim, True)

            # update the self.viewer
            self.gym_package.step_graphics(self.sim)
            self.gym_package.draw_viewer(self.viewer, self.sim, True)

            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            self.gym_package.sync_frame_time(self.sim)

            reward, current_ball_amount = self.calc_reward()
            print('amount: {:.3f} [mg] | reward: {:.3f}'.format(current_ball_amount * 1e3, reward))

    def simulate_forward(self, time, sync_frame_time_flag=True):
        for time_step in range(time):
            # step the physics
            self.gym_package.simulate(self.sim)
            self.gym_package.fetch_results(self.sim, True)

            if self.render_flag:
                # update the viewer
                self.gym_package.step_graphics(self.sim)
                self.gym_package.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time_flag:
                    self.gym_package.sync_frame_time(self.sim)
                print('refresh')

            if time_step % 10 == 0:
                self.make_image()

    def make_image(self):
        if self.save_image_flag:
            self.gym_package.render_all_camera_sensors(self.sim)
            get_image = self.gym_package.get_camera_image(self.sim, self.env, self.camera_handle, gymapi.IMAGE_COLOR)
            get_image = get_image.reshape(self.image_height, self.image_width, 4)
            get_image = get_image[:, :, :3]
            get_image = cv2.cvtColor(get_image, cv2.COLOR_RGB2BGR)
            image = get_image
            cv2.rectangle(image,
                          pt1=(self.image_width - 450, 100),
                          pt2=(self.image_width - 50, 300),
                          color=(255, 255, 255),
                          thickness=-1,
                          lineType=cv2.LINE_4,
                          shift=0)
            reward, current_ball_amount = self.calc_reward()
            cv2.putText(image,
                        text='{:.2f}'.format(current_ball_amount * 1000),
                        org=(self.image_width - 440, 240),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=4.,
                        color=(0, 0, 0),
                        thickness=5,
                        lineType=cv2.LINE_4)

            self.image_list.append(image)

            self.image_count += 1

            if self.step_num == self.MAX_EPISODE_LENGTH - 1 and self.image_save_done is False:
                self.image_save_done = True
                size = (self.image_width, self.image_height)
                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                video_filename = os.path.join(os.environ['HOME'], self.userDefinedSettings.HOST_NAME, 'videos', str(self.video_count) + '.mp4')
                save = cv2.VideoWriter(video_filename, fourcc, 10.0, size)
                for img in self.image_list:
                    save.write(img)
                save.release()
                self.video_count += 1
                self.image_list.clear()

    def incline_spoon(self, action_r, time=None):
        r_max = self.action_space.action_range['max']
        r_min = self.action_space.action_range['min']
        r = action_r
        if self.r_pos + r > r_max or self.r_pos + r < r_min:
            r = 0.
        self.gym_package.set_dof_target_position(self.env, self.slider_r, self.r_pos + r)
        self.r_pos = self.r_pos + r

        if time is not None:
            self.simulate_forward(time=time, sync_frame_time_flag=False)
        else:
            self.simulate_forward(time=self.action_loop_num_per_step, sync_frame_time_flag=False)

    def incline_spoon_absolute(self, action_r, time=None):
        r = action_r
        self.gym_package.set_dof_target_position(self.env, self.slider_r, r)

        if time is not None:
            self.simulate_forward(time=time, sync_frame_time_flag=False)
        else:
            self.simulate_forward(time=self.action_loop_num_per_step, sync_frame_time_flag=False)

    def shake_spoon(self, action_x):
        x = action_x * self.shake_speed_weight
        self.gym_package.set_dof_target_position(self.env, self.slider_x, self.x_pos + x)
        self.simulate_forward(time=int(self.action_loop_num_per_step / 3), sync_frame_time_flag=False)
        self.gym_package.set_dof_target_position(self.env, self.slider_x, self.init_x_pos)
        self.simulate_forward(time=self.action_loop_num_per_step, sync_frame_time_flag=False)

    def step(self, action, get_task_achievement=False):
        if self.ACTION_MAPPING_FLAG:
            action = self.mapping_action(action)
        action_x, action_r = action

        if self.incline_flag:
            if self.action_type == 'absolute':
                self.incline_spoon_absolute(action_r)
            else:
                self.incline_spoon(action_r)
        if self.shake_flag:
            self.shake_spoon(action_x)

        self.simulate_forward(time=self.action_loop_num_per_step * 3, sync_frame_time_flag=False)  # wait
        done = self.calc_done()
        reward, current_ball_amount = self.calc_reward()
        next_state = self.get_state()
        domain_parameter = None

        domain_parameter = self.domainInfo.get_domain_parameters()
        task_achievement = current_ball_amount

        self.task_achievement = task_achievement
        self.step_num += 1

        if get_task_achievement is True:
            return next_state, reward, done, domain_parameter, task_achievement
        else:
            return next_state, reward, done, domain_parameter

    def calc_done(self):
        return False

    def get_ball_amount(self):
        ball_num = self.get_number_in_spoon()
        current_ball_amount = ball_num * self.ball_mass * self.mass_gap_sim_rate
        return current_ball_amount

    def calc_reward(self):
        current_ball_amount = self.get_ball_amount()
        target_ball_amount = self.goal_powder_amount  # [g]

        reward_scaling = 1000.
        reward = -np.abs(current_ball_amount - target_ball_amount) * reward_scaling

        if np.abs(current_ball_amount - target_ball_amount) < 0.001:
            reward += 1.  # 1.

        return reward, current_ball_amount

    def reset(self, reset_info=None):
        self.image_list = []
        self.reset_world(reset_info)
        self.step_num = 0
        self.x_pos = self.gym_package.get_dof_position(self.env, self.slider_x)
        self.r_pos = self.gym_package.get_dof_position(self.env, self.slider_r)
        self.init_x_pos = self.x_pos
        self.init_r_pos = self.r_pos
        self.incline_spoon(-10. * np.pi / 180., time=self.action_loop_num_per_step * 5)
        state = self.get_state()

        self.image_list.clear()

        self.image_save_done = False
        print(self.domainInfo.get_domain_parameters())
        return state

    def get_state(self):
        reward, current_ball_amount = self.calc_reward()
        state = np.array([current_ball_amount * 500., self.r_pos * 30., self.goal_powder_amount * 500.]).reshape(-1)
        return state

    def mapping_action(self, action):
        assert (action.any() >= -1) and (action.any() <= 1), 'expected actions are \"-1 to +1\". input actions are {}'.format(action)
        low = self.action_space.low
        high = self.action_space.high
        action = low + (action + 1.0) * 0.5 * (high - low)  # (-1,1) -> (low,high)
        action = np.clip(action, low, high)  # (-X,+Y) -> (low,high)
        return action

    def random_action_sample(self):
        action = self.action_space.sample()
        if self.ACTION_MAPPING_FLAG:
            low = self.action_space.low
            high = self.action_space.high
            action = 2 * (action - low) / (high - low) - 1
        return action

    def user_direct_set_domain_parameters(self, domain_info):
        self.domainInfo.set_parameters(domain_info, type='set_split2')

    def __del__(self):
        if self.render_flag:
            self.gym_package.destroy_viewer(self.viewer)
        self.gym_package.destroy_sim(self.sim)
        self.gym_package.destroy_env(self.env)
        del self.gym_package
        torch.cuda.empty_cache()
        print('env deleted')


class Action_space:
    def __init__(self, action_type='absolute', shake_flag=True):
        if action_type == 'absolute':
            self.low = np.array([0., -30. * np.pi / 180.])
            self.high = np.array([1., -10. * np.pi / 180.])
        else:
            if shake_flag:
                self.low = np.array([0., -3. * np.pi / 180.])
                self.high = np.array([1., 3. * np.pi / 180.])
                self.action_range = {'min': -30. * np.pi / 180., 'max': -10. * np.pi / 180., }
            else:
                self.low = np.array([0., -10. * np.pi / 180.])
                self.high = np.array([1., 10. * np.pi / 180.])
                self.action_range = {'min': -70. * np.pi / 180., 'max': -10. * np.pi / 180., }
        self.ACTION_DIM = len(self.high)

    def sample(self):
        action = np.random.rand()
        action = 2. * (action - 0.5)
        return action
