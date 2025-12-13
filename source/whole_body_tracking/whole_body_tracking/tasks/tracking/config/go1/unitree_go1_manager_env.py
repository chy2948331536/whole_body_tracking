import torch
from isaaclab.envs import ManagerBasedRLEnv
from pprint import pprint

class UnitreeGo1EnvPlay(ManagerBasedRLEnv):
    # Flag to indicate play mode - used by MotionCommand to start from frame 0
    play_mode = True
    
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        from whole_body_tracking.utils.ros.publish import OneFilePublisher
        
        # Existing nodes
        self.torque_node = OneFilePublisher("torque", 12, "torque_node")
        # Termination monitoring nodes
        # anchor_pos_z: [error_z, threshold]
        self.termination_anchor_pos_z_node = OneFilePublisher("termination_anchor_pos_z", 1, "termination_anchor_pos_z_node")
        # anchor_ori: [error_ori, threshold]
        self.termination_anchor_ori_node = OneFilePublisher("termination_anchor_ori", 1, "termination_anchor_ori_node")
        # ee_body_pos_z: [error_left_ankle, error_right_ankle, error_left_wrist, error_right_wrist, max_error, threshold]
        self.termination_ee_body_pos_z_node = OneFilePublisher("termination_ee_body_pos_z", 1, "termination_ee_body_pos_z_node")

        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        self.play_mode = True
        # Initialize previous episode sums for reward logging
        self._prev_episode_sums = None
        # Create reward publisher nodes for each reward term
        self.reward_nodes = {}
        for reward_name in self.reward_manager.active_terms:
            topic_name = f"reward_{reward_name}"
            node_name = f"reward_{reward_name}_node"
            self.reward_nodes[reward_name] = OneFilePublisher(topic_name, 1, node_name)

    def _log_reward_increments(self):
        """Log and print the current step reward increments averaged across all environments.
        
        This function computes the difference between current and previous episode sums,
        averages across all environments, and prints the result using pprint.
        """
        current_episode_sums = self.reward_manager._episode_sums
        
        if self._prev_episode_sums is None:
            # First call: initialize previous sums with current values
            self._prev_episode_sums = {
                name: value.clone() for name, value in current_episode_sums.items()
            }
            return
        
        # Compute current time for publishing (use first env's time)
        # episode_length_buf is a tensor, step_dt is a float
        current_time = (self.episode_length_buf[0] * self.step_dt).item()
        
        # Compute increments (current - previous) and average across all envs
        reward_increments = {}
        for name in current_episode_sums.keys():
            increment = current_episode_sums[name] - self._prev_episode_sums[name]
            # Average across all environments
            avg_increment = torch.mean(increment).item()
            reward_increments[name] = avg_increment
            
            # Publish reward increment via ROS node
            if hasattr(self, 'reward_nodes') and name in self.reward_nodes:
                self.reward_nodes[name].cb([avg_increment], current_time)
        
        # Print the reward increments dictionary
        # pprint(reward_increments)
        
        # Update previous episode sums for next call
        self._prev_episode_sums = {
            name: value.clone() for name, value in current_episode_sums.items()
        }

    def step(self, action: torch.Tensor):
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # process actions
        self.action_manager.process_action(action.to(self.device))
        # action = torch.zeros_like(action)
        # self.action_manager._terms["joint_pos"]._processed_actions[:, :] = self.robot.data.default_joint_pos
        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        # Log reward increments
        self._log_reward_increments()

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)

        self.current_time = self.episode_length_buf * self.step_dt

        if hasattr(self, 'torque_node') and hasattr(self, 'scene'):
            robot = self.scene["robot"]
            if "base_legs" in robot.actuators:
                torques = robot.actuators["base_legs"].applied_effort
                self.torque_node.cb(torques.cpu().numpy().flatten().tolist(), self.current_time.item())

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
