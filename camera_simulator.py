import time

import mujoco as mj
import mujoco.viewer as mj_viewer
import numpy as np
from manipulated_object import ManipulatedObject


class CameraSimulator:
    def __init__(self, resolution=(500, 500), fovy=45, world_file="./data/world_mug.xml", launch_viewer=False,
                 real_time=False):
        self.real_time = real_time

        self.model = mj.MjModel.from_xml_path(world_file)
        self.data = mj.MjData(self.model)

        self.viewer = None
        if launch_viewer:
            self.viewer = mj_viewer.launch_passive(self.model, self.data)

        self.model.cam_fovy = fovy
        self.model.vis.global_.fovy = fovy  # probably don't need this line

        self.manipulated_object = ManipulatedObject(self.model, self.data)
        self.manipulated_object.set_orientation_euler([0, 0, 0])
        # self.manipulated_object.zero_velocities()

        self.renderer = mj.Renderer(self.model, resolution[0], resolution[1])
        self.depth_renderer = mj.Renderer(self.model, resolution[0], resolution[1])
        self.depth_renderer.enable_depth_rendering()

    def set_manipulated_object_position(self, position):
        self.manipulated_object.set_position(position)

    def set_manipulated_object_orientation_euler(self, orientation):
        self.manipulated_object.set_orientation_euler(orientation)

    def render(self, rotation_matrix, position):
        mj.mj_forward(self.model, self.data)
        self.data.cam_xpos = position
        self.data.cam_xmat = rotation_matrix.flatten()
        self.renderer.update_scene(self.data, camera=0)
        return self.renderer.render()

    def render_depth(self, rotation_matrix, position):
        mj.mj_forward(self.model, self.data)
        self.data.cam_xpos = position
        self.data.cam_xmat = rotation_matrix.flatten()
        self.depth_renderer.update_scene(self.data, camera=0)
        return self.depth_renderer.render()

    def step_simulation(self):
        step_start = time.time()

        mj.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

        if self.real_time:
            time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    def simulate_seconds(self, seconds):
        max_iter = int(seconds / self.model.opt.timestep)
        for i in range(max_iter):
            self.step_simulation()


if __name__ == "__main__":
    sim = CameraSimulator(resolution=(300,300), fovy=60, launch_viewer=True, real_time=True)
    pass

