import pymeshlab
import os
import glob
import sys
import polyscope as ps
import numpy as np
from thlog import *

thlog = Thlogger(LOG_INFO, VIZ_INFO, "meshlab-pclouds")

def sample_point_cloud_from_mesh(vertices: np.ndarray, faces: np.ndarray, n_points: int, save_filename=None):
    """ uses meshlab's montecarlo point cloud sampling from a mesh.
    Returns (point array, normal array), each of shape (n_points, 3).
    Optionally, if save_filename is present, saves the point cloud (with point
    normals) to a file (preferably a .ply file because that format seems more
    standard for oriented point clouds)
    """
    mlmesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(mlmesh)
    ms.generate_sampling_montecarlo(samplenum=n_points)  # will give exactly n_points
    if save_filename:
        ms.save_current_mesh(save_filename
            , save_vertex_color = False
            , save_vertex_normal = True  # save point cloud sampled normals...
            , save_face_color = False
            , save_wedge_texcoord = False
            , save_wedge_normal = False)

    pcloud = ms.current_mesh()
    assert pcloud.is_point_cloud(), "sampled point cloud still has some faces"
    pcloud_np_points = mesh.vertex_matrix()
    pcloud_np_normals = mesh.vertex_normal_matrix()
    
    return pcloud_np_points, pcloud_np_normals


def sample_point_cloud_from_obj_file(fname, n_points:int, save_filename = None):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(fname)
    ms.generate_sampling_montecarlo(samplenum=n_points)  # will give exactly n_points
    if save_filename:
        ms.save_current_mesh(save_filename
            , save_vertex_color = False
            , save_vertex_normal = True  # save point cloud sampled normals...
            , save_face_color = False
            , save_wedge_texcoord = False
            , save_wedge_normal = False)

    pcloud = ms.current_mesh()
    assert pcloud.is_point_cloud(), "sampled point cloud still has some faces"
    pcloud_np_points = pcloud.vertex_matrix()
    pcloud_np_normals = pcloud.vertex_normal_matrix()
    
    return pcloud_np_points, pcloud_np_normals
    


def sample_point_clouds_for_directory_of_meshes(directory: str, n_points: int, recursive_glob=False):
    """ dataroot is the path to the dataset's main directory.
        subdir is the folder under it; i.e. 'train', or 'test'.

        Samples point clouds of a whole folder of .obj files, and saves out
        a single "sampled_point_clouds_<n_meshes>x<n_points>p.npz" file in the directory
        that contains a pts array of shape (n_meshes, n_points, 3), and a 
        normals array of shape (n_meshes, n_points, 3).
    """
    fnames = glob.glob(os.path.join(directory, "*.obj"))
    all_pcloud_points = []
    all_pcloud_normals = []
    for fname in fnames:
        thlog.info(f"{fname}")
        points, normals = sample_point_cloud_from_obj_file(fname, n_points)
        all_pcloud_points.append(points)
        all_pcloud_normals.append(normals)

    all_pcloud_points = np.array(all_pcloud_points)
    all_pcloud_normals = np.array(all_pcloud_normals)
    thlog.info(f"There are {all_pcloud_points.shape[0]} point clouds, each with {all_pcloud_points.shape[1]} points.")
    save_filename = os.path.join(directory, f"sampled_point_clouds_{all_pcloud_points.shape[0]}x{all_pcloud_points.shape[1]}p.npz")

    thlog.info(f"Saving to {save_filename}")
    np.savez_compressed(save_filename
        , points = all_pcloud_points
        , normals = all_pcloud_normals
        )
    return save_filename

def load_point_cloud_from_dataset_pclouds_npz(fname, return_normals_too = False):
    """ convenience loader that by default just loads points and not the normals """ 
    with np.load(fname) as npz:
        points = npz['points']
        if return_normals_too:
            normals = npz['normals']
    if return_normals_too:
        return points, normals
    else:
        return points

def visualize_point_cloud_npz(npz_fname, mesh_index):
    thlog.init_polyscope()
    if thlog.guard(VIZ_INFO, needs_polyscope=True):
        points_for_all_meshes = load_point_cloud_from_dataset_pclouds_npz(npz_fname)
        ps.register_point_cloud(f"pcloud {mesh_index}", points_for_all_meshes[mesh_index])
        ps.show()

if __name__=="__main__":
    """
    usage: meshlab_point_clouds.py <sample|view_npz> <n_points|index of mesh to view pcloud> <dir>
    """
    mode = sys.argv[1]
    if mode == "sample":
        n_points = int(sys.argv[2])
        sample_point_clouds_for_directory_of_meshes(sys.argv[3], n_points=n_points)
    elif mode == "view_npz":
        visualize_point_cloud_npz(sys.argv[3], int(sys.argv[2]))
    else:
        thlog.err("unknown mode for this script.\n"
        "usage: meshlab_point_clouds.py <sample|view_npz> <n_points|index_of_mesh_to_view_pcloud> <dir|path_to_npz_file>")