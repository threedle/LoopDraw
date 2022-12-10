import numpy as np

def sample_from_ellipse(num_points, polar_params_array):
    x, y, a, b, φ = tuple(polar_params_array)
    δ = np.linspace(0, 2 * np.pi, num_points)
    xs = a * np.sin(δ)
    ys = b * np.cos(δ)
    pts = np.stack([xs, ys], axis=-1)
    rot_mat = np.array([
        [np.cos(φ), -np.sin(φ)],
        [np.sin(φ), np.cos(φ)]
    ])
    pts = pts @ rot_mat.T
    pts = pts + [x, y]
    return pts


def svd_fit_ellipse(pts):
    """
    pts: array, of shape (n_pts, 2)
    returns (x, y, a, b, φ) where φ is in radians
    - x, y is coords of center of ellipse
    - a, b are the two axis lengths of the ellipse
    - φ is the rotation angle w.r.t. the horizontal axis

    """
    n, d = pts.shape
    assert (d == 2)
    mean = pts.mean(axis=0)
    pts = pts - mean
    U, σ, V_h = np.linalg.svd(pts, full_matrices=False)
    
    try:
        assert len(σ) == 2, "in ellipse fit, only 1 singular value found; expected 2"
    except AssertionError as e:
        print("[geometry_utils | ERROR] " 
             f"SVD produced the following singular value (expected two): {σ} \n" 
             f"Here is the pts array: \n {pts} \n"
             f"Here are U and V_h: \n {U} \n {V_h}")
        raise e
    
    a, b = ((σ ** 2) / n * 2) ** 0.5
    x, y = mean
    major_axis = V_h[0]
    φ = np.arctan2(major_axis[1], major_axis[0])
    if φ < 0:
        # an ellipse rotates by periods of 180 degrees
        φ += np.pi
    return x, y, a, b, φ


def orientation_of_3pts_in_2d(p: np.ndarray, q:  np.ndarray, r: np.ndarray):
    """
    calculate the orientation of 3 points in 2D.
    returns -1 if counterclockwise, 0 if collinear, 1 if clockwise.
    """
    return np.sign(np.cross(q - p, q - r))


def unit(arr):
    return arr / np.linalg.norm(arr)

def extract_useful_data_from_plane(plane: np.ndarray):
    """ Since this keeps coming up over in loop_representations.py.
    From a (4,3) plane representation, return the following:
        ( plane_n: normal vector of plane
        , plane_p: origin point of coord system on plane
        , onplane_xaxis: x-axis vector of coord system on plane
        , onplane_yaxis: y-axis vector of coord system on plane 
            (note that plane_n is the z axis in this system)
        , plane_basis: the basis for this coord system, represented as a change-
            of-basis matrix from the plane's basis to world coords (R3)
        , change_to_plane_basis: the change-of-basis matrix from world coords
            to the plane basis coord system
        , plane_p_in_plane_basis: the point to consider an anchor in the plane's
            coord system, in that basis
        )

    """
    assert plane.shape == (4,3)
    plane_n = plane[0]
    plane_p = plane[1]
    onplane_xaxis = plane[2]
    onplane_yaxis = plane[3]
    # make a coordinate system / ortho basis with the x and y vectors flat
    # on the plane and the z vector is the plane's normal in 3d (so the
    # point's 2d coords on the plane should be the x and y coordinates in
    # this basis)
    plane_basis = np.array([
        unit(onplane_xaxis), 
        unit(onplane_yaxis), 
        unit(plane_n)])
    change_to_plane_basis = np.linalg.inv(plane_basis)
    # consider plane_p as the origin of this coordinate system
    plane_p_in_plane_basis = np.dot(change_to_plane_basis, plane_p)

    return \
        ( plane_n
        , plane_p
        , onplane_xaxis
        , onplane_yaxis
        , plane_basis
        , change_to_plane_basis
        , plane_p_in_plane_basis
        )
