import pymeshlab
import os
import sys
import polyscope as ps
import numpy as np
from thlog import *

thlog = Thlogger(LOG_INFO, VIZ_INFO, "meshlab-load")

def merge_glb_multimeshes(ms : pymeshlab.MeshSet):
    current_union_result_id = 0
    for mesh_id in range(len(ms)):
        if mesh_id == 0:
            continue
        thlog.info(f"unioning mesh layer {current_union_result_id} and {mesh_id}")
        ms.generate_boolean_union(first_mesh=current_union_result_id, second_mesh=mesh_id)
        current_union_result_id = ms.current_mesh_id()
    ms.set_current_mesh(current_union_result_id)

def load_via_meshlab(fname: str, convert_to_obj=False, simplify_to: int = -1):
    """ 
    Takes a filename to a 3D mesh format file (any format supported by meshlab,
    actually) and optionally returns the vertex and face arrays/saves as .obj
    Optional kwargs:
    - convert_to_object: if True, converts the file to .obj in the same folder,
        and disables the np array return, instead returning the output path.
    - simplify_to: (default: -1). If >0, then upon being loaded, the mesh will
        also be simplified using Meshlab's edge-collapse decimation to reach
        the target number of faces specified in simplify_to.  If simplify_to
        is negative, no simplification will be done.
    """
    mesh_name = os.path.basename(fname)
    mesh_dir = os.path.dirname(fname)
    thlog.info(f"Loading mesh {mesh_name}")
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(fname)
    
    if len(ms) > 1:
        thlog.info(f"{mesh_name}: has {len(ms)} mesh layers; unioning them")
        merge_glb_multimeshes(ms)

    if simplify_to > 0:
        thlog.info(f"{mesh_name}: simplifying to {simplify_to} faces")
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=simplify_to, autoclean=True)
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_repair_non_manifold_vertices()
    # ms.meshing_repair_non_manifold_edges(method=0)
    ms.set_selection_none()
    ms.meshing_repair_non_manifold_edges(method=0)
    ms.set_selection_none()
    ms.meshing_close_holes(maxholesize = 125)
    ms.set_selection_none()
    ms.compute_matrix_from_translation(traslmethod=1, alllayers=True, freeze=True)
    ms.set_selection_none()
    ms.compute_matrix_from_scaling_or_normalization(alllayers=True, unitflag=True, freeze=True)
    ms.set_selection_none()
    # translate to origin; so the mesh will end up centered in a unit cube.
    
        

    if convert_to_obj:
        
        out_path = os.path.join(mesh_dir, os.path.splitext(mesh_name)[0] + 
            (f"-{simplify_to}f" if simplify_to > 0 else "clean") + 
            ".obj")
        thlog.info(f"{mesh_name}: exporting to {out_path}")
        ms.save_current_mesh(out_path
            , save_textures = False 
            , save_vertex_color = False
            , save_vertex_normal = False
            , save_face_color = False
            , save_wedge_texcoord = False
            , save_wedge_normal = False)
        return out_path
    else:
        mesh = ms.current_mesh()
        mesh_np_verts = mesh.vertex_matrix()
        mesh_np_faces = mesh.face_matrix()
        return mesh_np_verts, mesh_np_faces


def meshlab_load_main(simplify_to, fnames):
    for fname in fnames:
        load_via_meshlab(fname, convert_to_obj=True, simplify_to=simplify_to)

if __name__ == "__main__":
    argv = sys.argv
    try:
        simplify_to = int(argv[1])
    except:
        thlog.err("Invalid integer for simplify_to (argument 1)")
        simplify_to = -1
    if len(argv) < 3:
        thlog.err("No filename argument. Usage: ./meshlab_load.py <#faces to simplify to> <filenames...>")
    meshlab_load_main(simplify_to, argv[2:])