import numpy as np
from scipy.spatial.transform import Rotation


def unify_transfer_matrix(
    rotation_matrix=None,
    quaternion_wxyz=None,
    translation_xyz=None
):
    if rotation_matrix is None:
        assert quaternion_wxyz is not None

        w, x, y, z = quaternion_wxyz
        rotation_matrix = Rotation.from_quat(np.array([x, y, z, w])).as_matrix()
    
    if translation_xyz is None:
        translation_xyz = np.zeros(3)

    transfer_matrix = np.eye(4, dtype=np.float32)
    transfer_matrix[:3, :3] = rotation_matrix
    transfer_matrix[:3, 3] = translation_xyz

    return transfer_matrix