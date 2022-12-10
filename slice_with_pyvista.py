# slice a mesh using pyvista and return the arrays needed by
# get_loops.MeshOneSlice.
import numpy as np
import vtk 
import pyvista

def slice_mesh_using_pyvista(fname: str, plane: np.ndarray):
    """
    return loop_pts, loop_conn_arr, normals_at_loop_pts:
    - loop_pts: array (n_points,3) of nodes in the polyline representing the 
        slice/cross-section of the mesh at its intersection with plane
    - loop_conn_arr: connectivity array, int array of shape (n_segments, 2)
        describing the segments of the polyline in terms of
        their indices into loop_pts
    - normals_at_loop_pts: always None for now, but to keep compatibility with
        the get_loops.MeshOneSlice code.

    plane is of shape (4,3) (really only (2,3) is needed for this function but
    (4,3) is used by all other functions in get_loops.py and others)
    - plane[0] should be the plane normal vector, plane[1] should be the origin
    of the coordinate system on the plane. plane[2] and plane[3] are x-axis
    and y-axis vectors respectively for a coordinate system centered at plane[1]
    and with a z-axis of plane[0].
    """
    polydata_mesh = pyvista.PolyData(fname)
    polydata_slice = polydata_mesh.slice(normal=plane[0], origin=plane[1])
    polydata_slice_lines = polydata_slice.lines
    # print(polydata_slice_lines)
    # polydata_mesh.plot()
    # polydata_slice.plot(show_edges=True)
    indexing_into_lines_array = np.arange(0, len(polydata_slice_lines), 3)
    
    first_point_indices = polydata_slice_lines[indexing_into_lines_array + 1]
    second_point_indices = polydata_slice_lines[indexing_into_lines_array + 2]
    
    # this connectivity array is correct but it's not in the right order for
    # our distinguish_disjoint_loops thing.
    connectivity = np.transpose(np.vstack((first_point_indices, second_point_indices)))
    
    # try to rearrange pairs in `connectivity` array so that we get the form
    # [[0,1],[1,x],[x,y],[y,w], etc etc. (adjacent pairs are adjacent segments)
    # (making sure that the indices loop back to a previously-seen index when
    # a loop is complete)

    conn_dict = {}
    for s_i, seg in enumerate(connectivity):
        conn_dict[seg[0]] = s_i
    
    conn_seg_indices_remaining = set(range(len(connectivity)))
    traced_conn = None  # this is where the result of rearranging will go
    curr_seg_conn_index = 0
    while conn_seg_indices_remaining:
        curr_seg = connectivity[curr_seg_conn_index]
        traced_conn = curr_seg if traced_conn is None else np.vstack((traced_conn, curr_seg))
        conn_seg_indices_remaining -= {curr_seg_conn_index}

        next_source_vtx_index = curr_seg[1]
        conn_seg_index_of_next_source_vtx = conn_dict[next_source_vtx_index]

        if conn_seg_index_of_next_source_vtx in conn_seg_indices_remaining:
            curr_seg_conn_index = conn_seg_index_of_next_source_vtx
        else:
            # we have completed a loop (the 'next vertex' was already popped
            # from the set of unaccounted-for indices into the `connectivity`
            # array), select an arbitrary loop to enter into next, if there are
            # still any!
            if conn_seg_indices_remaining:
                curr_seg_conn_index = conn_seg_indices_remaining.pop()
            else:
                break
    
    # unfortunately it's not possible to "calculate the mesh's normals
    # but at the points corresponding to the points on the slice segment nodes"
    # so we'll just suck it and leave it as None (forcing callers to run
    # estimate_loop_segment_normals_simple later )
    assert traced_conn is not None, "plane does not intersect the mesh!"
    return polydata_slice.points, traced_conn, None
    

if __name__ == "__main__":
    # test main
    slice_mesh_using_pyvista("../raw-3d-data/blender-procedural-vases/ofek-gen-new-07-30/chair_hndl_profile_blend_0_2889_0029.obj", 
        np.array([[0,1,0],[0,0.1,0]]))