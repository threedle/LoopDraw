import pymeshlab
import os
import sys
import polyscope as ps
import numpy as np
from thlog import *

thlog = Thlogger(LOG_INFO, VIZ_INFO, "meshlab")

CLOSE_HOLES_LIMIT = 250

def do_poisson_reco(points: np.array, normals: np.array, depth=8, save_filename=None):
    """
    Uses the MeshLab Python API to do poisson reconstruction on a point cloud
    with associated normals per point.

    Parameters:
    - points: np.array (n_points, 3) containing coordinates of points
    - normals: np.array (n_points, 3) containing the normal at each point
    optional:
    - depth: depth parameter for poisson reconstruction
    - save_filename: if a string, saves the resulting mesh to the filename. if None
        or an empty string, don't save the mesh to a file.

    Returns: a tuple (vertices, faces) (essentially the obj format for meshes)
    - vertices: np.array (n_mesh_verts, 3) of the vertices of the reconstructed
        mesh 
    - faces: integer np.array (n_faces, 3) defining faces in terms of
        indices for the vertices array
    """
    assert points.shape == normals.shape
    # manually create the meshlab Mesh object containing the oriented pt cloud
    mlpcloud = pymeshlab.Mesh(vertex_matrix=points, v_normals_matrix=normals)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(mlpcloud)
    thlog(LOG_INFO, "Starting meshlab's poisson reconstruction...")
    ms.generate_surface_reconstruction_screened_poisson(depth=depth) 
    
    ms.meshing_repair_non_manifold_edges(method=0)
    ms.meshing_close_holes(maxholesize=CLOSE_HOLES_LIMIT)

    # grab the current (the reconstructed mesh) and export its values
    mesh = ms.current_mesh()
    mesh_np_verts = mesh.vertex_matrix()
    mesh_np_faces = mesh.face_matrix()

    if save_filename:
        ms.save_current_mesh(save_filename
            , save_vertex_color = False
            , save_vertex_normal = False
            , save_face_color = False
            , save_wedge_texcoord = False
            , save_wedge_normal = False)
        thlog.info(f"saved reconstructed mesh to {save_filename}")

    return mesh_np_verts, mesh_np_faces



def run_meshlab_reco_main(argv):
    assert len(argv) > 3
    save_mode = argv[1]
    if (save_mode not in ["save", "view"]):
        thlog(LOG_INFO, "Usage: ./meshlab_poisson_reco.py <'save'|'view'> <path to .ply file of oriented point cloud>")
        return 0
    ply_path = argv[2]
    ply_name = os.path.basename(ply_path)
    ply_dir = os.path.dirname(ply_path)

    depth = int(argv[3])

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(ply_path)
    print(ply_path)

    ms.generate_surface_reconstruction_screened_poisson(depth=depth) 
    ms.meshing_repair_non_manifold_edges(method=0)
    ms.meshing_close_holes(maxholesize=CLOSE_HOLES_LIMIT)
    
    if save_mode == "save":
        out_path = os.path.join(ply_dir, os.path.splitext(ply_name)[0] + "-poissreco.obj")
        ms.save_current_mesh(out_path
        , save_vertex_color = False
        , save_vertex_normal = False
        , save_face_color = False
        , save_wedge_texcoord = False
        , save_wedge_normal = False)
        thlog(LOG_INFO, f"saved reconstructed mesh to {out_path}")

    elif save_mode == "view":
        mesh = ms.current_mesh()
        mesh_np_verts = mesh.vertex_matrix()
        mesh_np_faces = mesh.face_matrix()
        thlog.init_polyscope()
        if thlog.guard(VIZ_INFO, needs_polyscope=True):
            ps.register_surface_mesh(mesh_np_verts, mesh_np_faces)
            ps.show()

if __name__ == "__main__":
    run_meshlab_reco_main(sys.argv)
