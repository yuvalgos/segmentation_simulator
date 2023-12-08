import scipy


class ManipulatedObject:
    '''
     represent, query and manually manipulate the manipulated object
     assuming it's body name is 'manipulated_object'
    '''
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.jntadr = model.body('manipulated_object').jntadr[0]

    def set_pose(self, pose):
        assert len(pose) == 7
        self.data.qpos[self.jntadr:self.jntadr + 7] = pose

    def set_position(self, position):
        assert len(position) == 3
        self.data.qpos[self.jntadr:self.jntadr + 3] = position

    def zero_velocities(self):
        self.data.qvel[self.jntadr:self.jntadr + 7] = [0.0, ] * 7

    def set_orientation_quat(self, orientation):
        assert len(orientation) == 4
        self.data.qpos[self.jntadr + 3:self.jntadr + 7] = orientation

    def set_orientation_euler(self, orientation):
        assert len(orientation) == 3
         # use scipy to convert euler to quat
        orientation_quat = scipy.spatial.transform.Rotation.from_euler('xyz', orientation).as_quat()
        self.data.qpos[self.jntadr + 3:self.jntadr + 7] = orientation_quat

    def get_orientation_euler(self):
        return scipy.spatial.transform.Rotation.from_quat(self.data.qpos[self.jntadr + 3:self.jntadr + 7]).as_euler('xyz')

    def get_orientation_quat(self):
        return self.data.qpos[self.jntadr + 3:self.jntadr + 7]

    def get_position(self):
        return self.data.qpos[self.jntadr:self.jntadr + 3]

    def get_pose(self):
        return self.data.qpos[self.jntadr:self.jntadr + 7]
