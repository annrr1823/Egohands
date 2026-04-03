import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md"],
    pythonpath=True,
    dotenv=True,
)

import torch

from model.smpl.utils.rot6d import rotation_6d_to_matrix, matrix_to_rotation_6d


def transfer_mano_params_A_to_B(
    T_A_to_B,
    global_orient_A,
    transl_A,
    wrist_offset
):
    global_orient_A_mat = rotation_6d_to_matrix(torch.from_numpy(global_orient_A)).numpy()
    global_orient_B_mat =  T_A_to_B[None, :3, :3] @ global_orient_A_mat
    global_orient_B = matrix_to_rotation_6d(torch.from_numpy(global_orient_B_mat)).numpy()

    transl_A = transl_A + wrist_offset
    transl_B =  (T_A_to_B[None, :3, :3] @ transl_A[..., None]).squeeze(axis=-1)
    transl_B = transl_B + T_A_to_B[None, :3, 3]

    transl_B = transl_B - wrist_offset

    return global_orient_B, transl_B