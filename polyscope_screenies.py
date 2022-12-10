import polyscope as ps
import sys
import utils
import geometry_utils
import os
import numpy as np

from loop_representations import read_mesh_slices_npz_file_for_polyscope

# Takes .obj filenames (space separated) as arguments.
# Shows the GUI once (for the user to adjust viewing angles etc) and then goes
# through the rest of the input meshes taking polyscope screenshots at that same
# viewing angle without showing the GUI again. 
# (if the environment variable NO_GUI is defined, then no GUI will be
# shown at all, and the default polyscope camera angle is used.)

# This script will always initialize polyscope, and thus will not work without
# a GPU or OpenGL device of some sort.

# main view config: choose loop indices to highlight, and choose what happens to
# non highlighted loops
__TRANSPARENT = True
__NOGROUND = True
__SHADOW = True
# Loop highlighting options. Only matters if the meshname.obj has a meshname-slices.npz file
# in the same location, providing loop sequence information. 
__HIGHLIGHT_EDITS = [] # a list of loop indices to highlight (e.g. as edited). For example,  list(range(8, 12))
__HIDE_NON_EDITS = False  # specify this to just hide any curve net that isn't specified in __HIGHLIGHT_EDITS
__APPEARANCE_ABOVE_EDIT = "color"  # "hide", "color" (uses PS_COLOR_CUNET_ABOVE_EDIT), "unchanged", "flip" (only show loops above edit/highlighted, and hide below)

# draw a rectangle slice plane around the listed loops
__DRAW_PLANE_AT_LOOP = []
__DRAW_HIGHLIGHTED_LOOPS_NODES = []

# animate (i.e. export frames showing loops unveiling one at a time)
__ANIMATE = False
__ANIMATE_MAX_FRAMES = 132
""" if __ANIMATE=True, export a bunch of frames that build up to the final mesh.
Otherwise uses the same highlighting settings as specified above.
""" 

# color and appearance config
color_red = (1, 0, 0)
color_palepink = (255/255, 138/255, 233/255)
color_blue = (0, 0, 1)
color_paleyellow = (255/255, 255/255, 145/255)
color_bababa = (186/255, 186/255, 186/255)

PS_COLOR_CUNET_AT_EDIT = color_red
PS_COLOR_CUNET_BELOW_EDIT = utils.PS_COLOR_CURVE_NETWORK
PS_COLOR_CUNET_ABOVE_EDIT = color_palepink
PS_MESH_OPACITY_WITH_HIGHLIGHT = 1.0

PS_COLOR_SLICEPLANE = (178/255, 146/255, 255/255)
PS_COLOR_SLICEPLANE_BORDER = (29/255, 11/255, 32/255)



def make_fake_plane_mesh(loop_pts, register_ps_slice_plane=False):
    """only works if the loop_pts are fully on a plane parallel to the xy or yz or zx plane. 
    (hardcoded hack to draw rectangle points)
    """
    p0 = loop_pts[0]
    p1 = loop_pts[1]
    p2 = loop_pts[2]

    v0 = p1 - p0
    v1 = p2 - p1
    normal = np.cross(v0, v1)
    normal = normal / np.linalg.norm(normal)
    # the coordinate that is 1 is the axis direction
    normal_direction = list(np.abs(normal)).index(1)

    # i.e. if the plane goes +z then z coord should be the same for all loop_pts
    plane_height_coord = p0[normal_direction]
    if normal_direction == 0: # normal is x axis
        remaining_coords = (2, 1)
    elif normal_direction == 1: # normal is y
        remaining_coords = (0, 2)
    elif normal_direction == 2: # normal is z
        remaining_coords = (0, 1)

    # extent of the loop pts, to draw a nice rectangle
    x_max, y_max = tuple(np.max(loop_pts[:, remaining_coords], axis=0))
    x_min, y_min = tuple(np.min(loop_pts[:, remaining_coords], axis=0))
    # x_range = x_max - x_min
    # y_range = y_max - y_min
    x_range = 0.25
    y_range = 0.25
    expand_factor = 1
    rect_points = [np.zeros_like(p0) for _ in range(4)]
    for i in range(4):
        fake_x = (-1 if (i % 2) == 0 else 1) * expand_factor * x_range
        fake_y = (-1 if i < 2 else 1) * expand_factor * y_range

        rect_points[i][normal_direction] = plane_height_coord
        rect_points[i][remaining_coords[0]] = fake_x
        rect_points[i][remaining_coords[1]] = fake_y
        
    # p0 is bottom left, p1 is bottom right, p2 is top left, p3 is top right
    plane_mesh_points = np.stack(rect_points)
    plane_mesh_conn = np.array([[0, 1, 2], [2,1, 3]])
    plane_border_cunet = np.array([[0,1],[1,3],[3,2],[2,0]])

    if register_ps_slice_plane:
        ps_plane = ps.add_scene_slice_plane()
        ps_plane.set_draw_plane(False)
        ps_plane.set_draw_widget(False)
        ps_plane.set_pose(p0 + 5e-3 * normal, -normal)
    
    return plane_mesh_points, plane_mesh_conn, plane_border_cunet





def distinguish_disjoint_loops_lite(loop_ptses, loop_conns, loop_seg_normalses):
    """same as the same-named fuinction in get_loops.py but operates on lists
    of mesh-slice-defining arrays instead of one MeshOneSlice object.  
    The reason I need this function is because by default the slice object
    stores all closed polygons on the same plane height as ONE slice. But when
    we speak of highlighting particular loops for visualizing loop edits,
    we actually want to highlight on the scale of individual closed polygon
    loops, so we must further split those combined per-slice arrays 
    (pts, conn, normals) into per-closed-polygon arrays.
    """
    def do_one_slice(loop_conn_arr):
        entering_new_loop = True
        curr_polygonstart = -1

        curr_loop_conn = []
        disjoint_loop_conns = []
        
        for segment in loop_conn_arr:
            curr_i = segment[0]
            next_i = segment[1]
            if entering_new_loop:
                curr_polygonstart = curr_i
                entering_new_loop = False
            
            curr_loop_conn.append(segment)
            # seg_vec = loop_pts[next_i] - loop_pts[curr_i]
            # loop_seg_vectors[curr_i] = seg_vec

            if next_i == curr_polygonstart:
                # we've wrapped around and finished a polygon, so skip this iter
                # and remember to update the curr_polygonstart index next iter

                # add in the last segment for the conn of the current loop
                # curr_loop_conn.append(segment)
                # add the populated current loop to the list of disjoint loops
                disjoint_loop_conns.append(curr_loop_conn)
                # and reset curr_ for the next loop
                curr_loop_conn = []
                entering_new_loop = True
                continue
        disjoint_loop_conns = list(map(np.array, disjoint_loop_conns))
        return disjoint_loop_conns
    
    closed_loops = []
    for pts, conn, normals in zip(loop_ptses, loop_conns, loop_seg_normalses):
        disjoint_conns = do_one_slice(conn)  # a list, each item being the conn arr of one closed polygon loop
        for conn_this_closed_loop in disjoint_conns:
            pts_this_closed_loop = pts[conn_this_closed_loop[:, 0]]
            normals_this_closed_loop = normals[conn_this_closed_loop[:, 0]]
            closed_loops.append((pts_this_closed_loop, (conn_this_closed_loop - np.min(conn_this_closed_loop)), normals_this_closed_loop))
    return closed_loops
            

def main(current_animation_step: int = None):
    """ current_animation_step is only used in __ANIMATE=True mode.  """

    obj_fnames = sys.argv[1:]
    do_ps_show = (os.environ.get("NO_GUI") is None)
    this_is_the_last_animation_frame = False
    ps.init()
    if __NOGROUND:
        ps.set_ground_plane_mode("shadow_only" if __SHADOW else "none")
    for viz_i, fname in enumerate(obj_fnames):
        ps.remove_all_structures()
        mesh_name = os.path.basename(fname)
        print(f"{viz_i}: {fname}")
        try:
            polysoup = utils.PolygonSoup.from_obj(fname)
        except:
            print(f"failed loading mesh file {fname}, skipping")
            continue
        mesh = ps.register_surface_mesh("mesh", polysoup.vertices, polysoup.indices,
            color=utils.PS_COLOR_SURFACE_MESH)
        mesh.set_transparency(PS_MESH_OPACITY_WITH_HIGHLIGHT if __HIGHLIGHT_EDITS else 1.0)

        # try loading the loops as one curve network
        slices_npz_fname = os.path.join(os.path.dirname(fname), os.path.splitext(mesh_name)[0] + "-slices.npz")
        do_highlight_edits = True if (__HIGHLIGHT_EDITS or current_animation_step is not None) else False
        if os.path.isfile(slices_npz_fname):
            print(f"loading file {slices_npz_fname} for loops")
            
            loop_pts, loop_conn_arr, loop_segment_normals = \
                read_mesh_slices_npz_file_for_polyscope(slices_npz_fname, consolidate_into_one_curve_network=not do_highlight_edits)
            
            if not do_highlight_edits:
                cunet = ps.register_curve_network("loops", loop_pts, loop_conn_arr, color=utils.PS_COLOR_CURVE_NETWORK, radius=0.004)
                cunet.add_vector_quantity("loopnormals", loop_segment_normals, length=0.15, enabled=False)
            else:
                closed_loops = distinguish_disjoint_loops_lite(loop_pts, loop_conn_arr, loop_segment_normals)
                # for loop_i, (pts, conn, normals) in enumerate(zip(loop_pts, loop_conn_arr, loop_segment_normals)):
                min_highlight_edit = min(__HIGHLIGHT_EDITS) if __HIGHLIGHT_EDITS else 99999
                n_closed_loops = len(closed_loops)
                for loop_i, (pts, conn, normals) in enumerate(closed_loops):
                    if loop_i < min_highlight_edit:
                        cunet_color = PS_COLOR_CUNET_BELOW_EDIT
                        cunet_enabled = False if __APPEARANCE_ABOVE_EDIT == "flip" else True
                    elif loop_i in __HIGHLIGHT_EDITS:
                        cunet_color = PS_COLOR_CUNET_AT_EDIT
                        cunet_enabled = True
                    elif loop_i > min_highlight_edit:
                        # not an explicitly edited loop, but still
                        cunet_color = PS_COLOR_CUNET_ABOVE_EDIT if __APPEARANCE_ABOVE_EDIT in ("color", "flip") else \
                            PS_COLOR_CUNET_BELOW_EDIT
                        cunet_enabled = False if __APPEARANCE_ABOVE_EDIT == "hide" else True
                    cunet_enabled = (loop_i in __HIGHLIGHT_EDITS) if __HIDE_NON_EDITS else cunet_enabled
                    # if we hit a loop that is configured to be always hidden,
                    # then the animation will stop here because subsequent
                    # frames will just look the same as this one (no new loops
                    # revealed, because once a loop is hidden, we can assume
                    # that all subsequent loops are also hidden. However, this
                    # assumption does NOT hold when the __APPEARANCE_ABOVE_EDIT
                    # mode is "flip". (in which case, skip all these hidden-loop
                    # frames until we get a non-hidden loop, above which all
                    # subsequent loops will be shown.))
                    this_is_the_last_animation_frame = (not cunet_enabled) and __APPEARANCE_ABOVE_EDIT != "flip"

                    # the above defines the static (in an animation, the final
                    # frame) appearance of the loops. Here we stack on a
                    # condition that the loop should still be hidden if the
                    # animation step has not gotten to this loop index yet. But
                    # only apply (&&) if current_animation_step is defined.
                    cunet_enabled = cunet_enabled and ((current_animation_step is None) or (loop_i <= current_animation_step))

                    cunet = ps.register_curve_network(f"cunet{loop_i}", pts, conn, color=cunet_color, enabled=cunet_enabled, radius=0.004)
                    cunet.add_vector_quantity(f"normals{loop_i}", normals, length=0.15, enabled=False)

                    if loop_i in __DRAW_PLANE_AT_LOOP:
                        plane_mesh_points, plane_mesh_conn, plane_mesh_cunet = make_fake_plane_mesh(pts, register_ps_slice_plane=True)
                        fake_plane_mesh = ps.register_surface_mesh(f"plane at loop {loop_i}", plane_mesh_points, plane_mesh_conn, color=PS_COLOR_SLICEPLANE)
                        fake_plane_mesh.set_transparency(0.5)
                        ps.register_curve_network(f"planeborder at loop {loop_i}", plane_mesh_points, plane_mesh_cunet, color=PS_COLOR_SLICEPLANE_BORDER)
                    if loop_i in __DRAW_HIGHLIGHTED_LOOPS_NODES:
                        ps.register_point_cloud(f"loop nodes at loop {loop_i}", pts)
                # finished iterating thru all loops... now decide whether this was the last frame
                this_is_the_last_animation_frame = this_is_the_last_animation_frame or \
                    (current_animation_step is None or (current_animation_step > n_closed_loops))

        mesh.set_enabled(this_is_the_last_animation_frame or (not __ANIMATE))
        if viz_i == 0 and (current_animation_step is None or current_animation_step == 0) and do_ps_show:
            print("showing the polyscope GUI for you to set the camera angle for the rest of the screenshots")
            ps.show()
        # screenshot
        screenshot_fname = os.path.join(
            os.path.dirname(fname),
            os.path.splitext(mesh_name)[0] + 
                (f"frame{current_animation_step}" if current_animation_step is not None else "") + 
                "-img.png")
        ps.screenshot(filename=screenshot_fname, transparent_bg=__TRANSPARENT)
    return this_is_the_last_animation_frame

if __name__ == "__main__":
    if not __ANIMATE:
        main()
    else:
        for animation_step_i in range(__ANIMATE_MAX_FRAMES):
            this_is_the_last_animation_frame = main(animation_step_i)
            if this_is_the_last_animation_frame:
                break