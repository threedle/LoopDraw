# converts the two types of npz loop files we use into the point cloud array format 
# used in meshlab_point_clouds.py. 
# NOTE that the point clouds generated here are points ONLY SAMPLED ALONG LOOP SEGMENTS.
# The point clouds are NOT sampled from a reconstructed mesh. 
# No mesh reconstruction happens in this file.

# the first kind of file we read is the preprocessed_cache.npz file used as data arrays.
# the second kind is the slices.npz that is spit out during inference --save runs.

# outputs to the same directory a file called loopsampled_point_clouds_<n_meshes>x<n_points>p.npz

import numpy as np
import os
import sys
import glob

import loop_representations as lreprs
import get_loops
from thlog import *

thlog = Thlogger(LOG_INFO, VIZ_NONE, "npz-to-pclouds", imports=[lreprs.thlog, get_loops.thlog])

np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)

MIN_N_POINTS_PER_LOOP = 24

def loop_seq_to_oriented_point_cloud(pred_seq, loop_repr_type, planes, n_points_to_sample_for_point_cloud, use_eos_token:bool):
    """ 
    the first argument is named pred_seq but the sequence need not have been
    from a model. This could also just be a dataset sequence taken from a
    preprocessed_cache.npz dataset file. Code taken from
    loop_models_inference.py.

    Parameters:
    - pred_seq: array of shape (any_n_timesteps, loop_repr_n_features)
    (i.e. for loop_repr_type == 3 (fixed-res-polyline), there are 65 features)
    - loop_repr_type: int, representing the loop representation type to interpret
    - planes: iterable of arrays of shape (4, 3) each describing a plane
    - n_points_to_sample_for_point_cloud: n points to sample from the oriented point cloud.
    Returns: (points, normals), each (n_points,)
    - use_eos_token: whether to detect an EOS token embedding (all-zeros with lvl up flag of 1)
        to stop the sequence early in point cloud recovery
    """

    ### Switch the pointcloud recovery process depending on the loop repr used
    if loop_repr_type == lreprs.LOOP_REPR_ELLIPSE_SINGLE:
        loop_repr_object = \
            lreprs.EllipseSingle(planes, pred_seq)

    elif loop_repr_type == lreprs.LOOP_REPR_ELLIPSE_MULTIPLE:
        loop_repr_object = \
            lreprs.EllipseMultiple(planes, pred_seq, 
            segmentize_resolution=64, n_points_to_sample_for_point_cloud=64, 
            use_eos_token=use_eos_token)

    elif loop_repr_type == lreprs.LOOP_REPR_FIXED_RES_POLYLINE:
        loop_repr_object = \
            lreprs.EllipseMultiple(planes, pred_seq,
            segmentize_resolution=64, 
            # n_points_to_sample_for_point_cloud=48,
            n_points_to_sample_for_point_cloud=n_points_to_sample_for_point_cloud,
            run_in_fixed_resolution_polylines_mode=True,
            min_n_points_per_loop=MIN_N_POINTS_PER_LOOP,
            use_eos_token = use_eos_token,
            postprocessing_heuristics=
                ['prevent-thin-loops', 'smooth-normals']
                # NOTE NO 'caps' heuristic!! this is a difference compared to
                # the usual pipeline (pred_seq -> loop_repr_object -> oriented
                # pcloud -> reconstructed mesh) as done in
                # loop_models_inference.py. For that pipeline, to prevent holes,
                # we add caps. But here we only want points belonging to the
                # loops to compare only loops with loops! Thus we don't add any
                # caps.
            ) 
    else:
        raise NotImplementedError(f"loop representation {loop_repr_type} not"
                                   "yet implemented")
    
    all_points, all_normals = loop_repr_object.get_oriented_point_cloud()
    return all_points, all_normals



def convert_preprocessed_cache_npz_to_point_clouds(fname, n_points_to_sample_for_point_cloud, use_eos_token):
    directory = os.path.dirname(fname)
    all_pcloud_points = []
    all_pcloud_normals = []
    with np.load(fname) as npz:
        meshes_reprs = npz['meshes_reprs']
        planes = npz['planes']
        # from spec (due to when we implemented this key) if there is no loop_repr_type then
        # assume that the file is of ellipse-multiple representation.
        loop_repr_type = npz.get('loop_repr_type', lreprs.LOOP_REPR_ELLIPSE_MULTIPLE)
    for data_seq in meshes_reprs:
        points, normals = loop_seq_to_oriented_point_cloud(data_seq, loop_repr_type, planes, n_points_to_sample_for_point_cloud, use_eos_token)
        if points.shape[0] > n_points_to_sample_for_point_cloud:
            pick_indices =  np.random.choice(points.shape[0], n_points_to_sample_for_point_cloud, replace=False)
            points = points[pick_indices]
            normals = normals[pick_indices]
        all_pcloud_points.append(points)
        all_pcloud_normals.append(normals)
    # save out
    all_pcloud_points = np.array(all_pcloud_points)
    all_pcloud_normals = np.array(all_pcloud_normals)
    
    save_filename = os.path.join(directory, f"loopsampled_point_clouds_{all_pcloud_points.shape[0]}x{all_pcloud_points.shape[1]}p.npz")
    thlog.info(f"saving loop point clouds npz file to {save_filename}")
    np.savez_compressed(save_filename
        , points = all_pcloud_points
        , normals = all_pcloud_normals
        )



def convert_slices_npz_to_point_clouds(directory, n_points_to_sample_for_point_cloud):
    fnames = glob.glob(os.path.join(directory, "*slices.npz"))    
    all_pcloud_points = []
    all_pcloud_normals = []
    for fname in fnames:
        # despite these being called "loop_*" for the sake of consistency with
        # the names used in MeshOneSlice, these arrays (and the same ones in
        # MeshOneSlice) are points and conn for a cross-section of a mesh, i.e.
        # containing all disjoint closed loops on the same plane. (this is
        # because consolidate_into_one_curve_network=False. If this were True we
        # would be getting one set of arrays for the whole mesh's worth of
        # loops, not split up in a list one for each slice plane.)
        with np.load(fname) as npz:
            planes = npz['planes']
        
        mesh_slices = []
        for loop_pts, loop_conn_arr, loop_segment_normals, plane in \
            zip(*lreprs.read_mesh_slices_npz_file_for_polyscope(fname, consolidate_into_one_curve_network=False), planes):
            
            mesh_slices.append(get_loops.MeshOneSlice(
                plane, predicted_loops=(loop_pts, loop_conn_arr, loop_segment_normals)))

        # NOTE that the slices.npz file is exported from the EllipseMultiple
        # object AFTER the prevent-thin-loops heuristic has been applied (in its
        # process_into_mesh_slices function), but not 'smooth-normals' and
        # 'caps'. So we can apply smooth-normals HERE by setting normal_lerping
        # = True. (like how the EllipseMultiple object calls this
        # sample_from_list_of_mesh_slices function). Moreover, we don't want to
        # apply caps, so this sample_ call is all we need here. (In
        # EllipseMultiple, it followed this call up with manually adding cap
        # points.)
        points, normals = lreprs.sample_from_list_of_mesh_slices(mesh_slices, 
            n_points=n_points_to_sample_for_point_cloud, 
            distribute_points_by_loop_length=True,
            min_n_points_per_loop=MIN_N_POINTS_PER_LOOP,
            normal_lerping=True)
        
        # same procedure to prune to n_points as with convert_preprocessed_cache_npz_to_point_clouds
        if points.shape[0] > n_points_to_sample_for_point_cloud:
            pick_indices =  np.random.choice(points.shape[0], n_points_to_sample_for_point_cloud, replace=False)
            points = points[pick_indices]
            normals = normals[pick_indices]

        all_pcloud_points.append(points)
        all_pcloud_normals.append(normals)
        
    all_pcloud_points = np.array(all_pcloud_points)
    all_pcloud_normals = np.array(all_pcloud_normals)    

    # save out 
    save_filename = os.path.join(directory, f"loopsampled_point_clouds_{all_pcloud_points.shape[0]}x{all_pcloud_points.shape[1]}p.npz")
    np.savez_compressed(save_filename
        , points = all_pcloud_points
        , normals = all_pcloud_normals
        )


def npz_to_point_clouds_main():
    """ 
    usage: 
    npz_to_point_clouds.py <preprocessed_cache|slices> <filepath to preprocessed_cache.npz|directory of slices.npz files> <n_points_per_pcloud>

    converts the two main types of loop data npz files in this project into
    fixed-point-count batched point cloud npz files. The point clouds are
    sampled along loop segments; there is no mesh reconstruction or mesh pcloud
    sampling here.

    The output file is named loopsampled_point_clouds_<n_points>x<n_points_per_pcloud>p.npz 
    saved in the same directory as the source directoroy/preprocessed_cache file
    and contains a "points" array and a "normals" array, both of
    shape (n_point_clouds, n_points_per_pcloud, 3). This is the same 
    file format used by meshlab_point_clouds.py (the point cloud npz files can 
    be viewed with that script.)
    """
    if len(sys.argv) != 4:
        print(npz_to_point_clouds_main.__doc__)
        exit(1)

    if sys.argv[1] == "preprocessed_cache":
        fname = sys.argv[2]
        n_points_to_sample_for_point_cloud = int(sys.argv[3])
        convert_preprocessed_cache_npz_to_point_clouds(fname, n_points_to_sample_for_point_cloud, True)
    
    elif sys.argv[1] == "slices":
        directory = sys.argv[2]
        n_points_to_sample_for_point_cloud = int(sys.argv[3])
        convert_slices_npz_to_point_clouds(directory, n_points_to_sample_for_point_cloud)
    else:
        print("unknown mode.")
        print(npz_to_point_clouds_main.__doc__)
        exit(1)

if __name__ == "__main__":
    npz_to_point_clouds_main()