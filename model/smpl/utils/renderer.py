from typing import List

import os

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md"],
    pythonpath=True,
    dotenv=True,
)

import cv2
import numpy as np
import pickle
import pyrender
import torch
import trimesh

from model.smpl.mano import MANO_PATH

if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def make_translation(t):
    return make_4x4_pose(torch.eye(3), t)

def make_4x4_pose(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    return (*, 4, 4)
    """
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (
        torch.tensor([0, 0, 0, 1], device=R.device)
        .reshape(*(1,) * len(dims), 1, 4)
        .expand(*dims, 1, 4)
    )
    return torch.cat([pose_3x4, bottom], dim=-2)


def make_rotation(rx=0, ry=0, rz=0, order="xyz"):
    Rx = rotx(rx)
    Ry = roty(ry)
    Rz = rotz(rz)
    if order == "xyz":
        R = Rz @ Ry @ Rx
    elif order == "xzy":
        R = Ry @ Rz @ Rx
    elif order == "yxz":
        R = Rz @ Rx @ Ry
    elif order == "yzx":
        R = Rx @ Rz @ Ry
    elif order == "zyx":
        R = Rx @ Ry @ Rz
    elif order == "zxy":
        R = Ry @ Rx @ Rz
    return make_4x4_pose(R, torch.zeros(3))

def rotx(theta):
    return torch.tensor(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ],
        dtype=torch.float32,
    )

def roty(theta):
    return torch.tensor(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        dtype=torch.float32,
    )

def rotz(theta):
    return torch.tensor(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )


def get_light_poses(n_lights=5, elevation=np.pi / 3, dist=12):
    # Get lights in a circle around origin at elevation.
    thetas = elevation * np.ones(n_lights)
    phis = 2 * np.pi * np.arange(n_lights) / n_lights
    poses = []
    trans = make_translation(torch.tensor([0, 0, dist]))
    for phi, theta in zip(phis, thetas):
        rot = make_rotation(rx=-theta, ry=phi, order="xyz")
        poses.append((rot @ trans).numpy())
    return poses


def create_raymond_lights():
    """
    Return raymond light nodes for the scene.
    """
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes


class MANORenderer:
    def __init__(self):
        mano_path = os.path.join(MANO_PATH, 'MANO_RIGHT.pkl')
        with open(mano_path, 'rb') as mano_file:
            model_data = pickle.load(mano_file, encoding='latin1')

        faces = model_data["f"]

        # Add faces that make the hand mesh watertight.
        faces_extra = np.array([
            [92, 38, 234],
            [234, 38, 239],
            [38, 122, 239],
            [239, 122, 279],
            [122, 118, 279],
            [279, 118, 215],
            [118, 117, 215],
            [215, 117, 214],
            [117, 119, 214],
            [214, 119, 121],
            [119, 120, 121],
            [121, 120, 78],
            [120, 108, 78],
            [78, 108, 79]
        ])
        faces = np.concatenate([faces, faces_extra], axis=0)
        
        self.faces_right = faces
        self.faces_left = faces[:, [0, 2, 1]]

    def vertices_to_trimesh(
        self, 
        vertices,
        is_rhand=1,
        mesh_base_color=[1.0, 1.0, 0.9],
        rot_axis=[1, 0, 0],
        rot_angle=0,
    ):
        vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])

        if is_rhand:
            mesh = trimesh.Trimesh(vertices.copy(), self.faces_right.copy(), vertex_colors=vertex_colors)
        else:
            mesh = trimesh.Trimesh(vertices.copy(), self.faces_left.copy(), vertex_colors=vertex_colors)
        
        rot = trimesh.transformations.rotation_matrix(
            np.radians(rot_angle), rot_axis)
        mesh.apply_transform(rot)

        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)

        return mesh     

    def add_lighting(self, scene, cam_node, color=np.ones(3), intensity=0.3):
        # from phalp.visualize.py_renderer import get_light_poses
        light_poses = get_light_poses()
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            node = pyrender.Node(
                name=f"light-{i:02d}",
                light=pyrender.DirectionalLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)

    def add_point_lighting(self, scene, cam_node, color=np.ones(3), intensity=0.3):
        # from phalp.visualize.py_renderer import get_light_poses
        light_poses = get_light_poses(dist=0.5)
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            node = pyrender.Node(
                name=f"plight-{i:02d}",
                light=pyrender.PointLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)

    def render(
        self,
        vertices: np.array,
        is_rhand,
        render_res,
        focal_length,
        camera_center,
        rot_axis=[1, 0, 0],
        rot_angle=0,
        mesh_base_color=[LIGHT_BLUE],
        background_color=(0, 0, 0),
    ):
        scene = pyrender.Scene(
            bg_color=[*background_color, 0.0],
            ambient_light=(0.2, 0.2, 0.2)
        )

        for i, (v, r, c) in enumerate(zip(vertices, is_rhand, mesh_base_color)):
            trimesh = self.vertices_to_trimesh(vertices=v, is_rhand=r, mesh_base_color=c, rot_axis=rot_axis, rot_angle=rot_angle)
            mesh = pyrender.Mesh.from_trimesh(trimesh)
            scene.add(mesh, f'mesh_{i}')

        camera = pyrender.IntrinsicsCamera(
            fx=focal_length[0], fy=focal_length[1],
            cx=camera_center[0], cy=camera_center[1], 
            zfar=1e12
        )
        camera_pose = np.eye(4)

        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)
        self.add_point_lighting(scene, camera_node)
        self.add_lighting(scene, camera_node)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        renderer = pyrender.OffscreenRenderer(
            viewport_width=render_res[0],
            viewport_height=render_res[1],
            point_size=1.0
        )

        render_rgba, render_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        render_rgba = render_rgba.astype(np.float32) / 255.0
        renderer.delete()

        return render_rgba, render_depth
    
    def render_with_image(
        self,
        image,
        vertices: np.array,
        is_rhand,
        focal_length,
        camera_center,
        render_factor: float = 2.0,
        mesh_base_color=[LIGHT_BLUE],
    ):
        batch_size = vertices.shape[0]
        render_res = (image.shape[1], image.shape[0])

        if render_factor != 1:
            render_res = (int(render_res[0] * render_factor), int(render_res[1] * render_factor))
            focal_length = (int(focal_length[0] * render_factor), int(focal_length[1] * render_factor))
            camera_center = (int(camera_center[0] * render_factor), int(camera_center[1] * render_factor))

        render_rgba, render_depth = self.render(
            vertices=vertices, 
            is_rhand=is_rhand,
            render_res=render_res, 
            focal_length=focal_length,
            camera_center=camera_center,
            rot_axis=[1, 0, 0],
            rot_angle=0,
            mesh_base_color=mesh_base_color * batch_size,
            background_color=(1, 1, 1),
        )

        image = image.astype(np.float32)[:, :, ::-1] / 255.0
        image = np.concatenate([image, np.ones_like(image[:, :, :1])], axis=2)

        render_rgba = cv2.resize(render_rgba, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)

        render_bgr_with_image = image[:, :, :3] * (1 - render_rgba[:, :, 3:]) + render_rgba[:, :, :3] * render_rgba[:, :, 3:]
        render_bgr_with_image = render_bgr_with_image[:, :, ::-1] * 255
        render_bgr_with_image = render_bgr_with_image.astype(np.uint8)

        return render_bgr_with_image, render_rgba, render_depth
