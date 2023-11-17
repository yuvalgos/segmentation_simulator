import mujoco as mj
import mujoco.viewer as mj_viewer
import numpy as np


class CameraSimulator:
    def __init__(self, resolution=(500, 500), fovy=45, world_file="./data/world_table.xml",launch_viewer=False):
        self.model = mj.MjModel.from_xml_path(world_file)
        self.data = mj.MjData(self.model)

        if launch_viewer:
            self.viewer = mj_viewer.launch_passive(self.model, self.data)

        self.model.vis.global_.fovy = fovy

        self.renderer = mj.Renderer(self.model, resolution[0], resolution[1])

    def render(self, rotation_matrix, position):
        mj.mj_forward(self.model, self.data)
        self.data.cam_xpos = position
        self.data.cam_xmat = rotation_matrix.flatten()
        self.renderer.update_scene(self.data, camera=0)
        return self.renderer.render()

