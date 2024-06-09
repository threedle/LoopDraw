# this module is intended to be hot-reloadable during runtime of 
# the InferenceREPL, called from loop_inference_config_and_repl.py.
from typing import Optional
import numpy as np
import torch

def apply_rotation_matrix(unn_ts: np.ndarray, angle_in_degrees: float):
    """
    helper function for loop editing: apply rotation to a loop.
    unn_ts is an unnormalized timestep representation vector, with shape (1, 2N+1)
    """
    # unn_ts has shape (1, 2N+1)
    #  to get the x,y array we do this first
    n_coords_per_loop = unn_ts.shape[-1] - 1
    points = unn_ts[:, :-1].reshape(-1, 2)
    # then we apply a rotation matrix
    angle = np.deg2rad(angle_in_degrees)
    sin = np.sin(angle)
    cos = np.cos(angle)
    rot = np.array([[cos, -sin], [sin, cos]])
    rotated_points = points @ rot.T
    recovered_unn_ts = np.concatenate((rotated_points.reshape(unn_ts.shape[0], n_coords_per_loop), unn_ts[:, -1:]), axis=-1)
    return recovered_unn_ts

def turn_into_square(unn_ts: np.ndarray, angle_in_degrees: float, scale: Optional[float]):
    """
    helper function for loop editing: turn a loop into a square
    unn_ts is an unnormalized timestep representation vector, with shape (1, 2N+1)
    """
    # if unn_ts[0, -1] < 0.5:
    #     return unn_ts
    n_points_per_loop = unn_ts.shape[-1] // 2
    if n_points_per_loop % 4 != 0:
        return unn_ts
    points = unn_ts[:, :-1].reshape(-1, 2).copy()
    points_per_side = n_points_per_loop // 4
    side_coords = np.arange(-0.5, 0.5, 1/points_per_side)
    for side_i in range(4):
        start = side_i * points_per_side
        end = (side_i + 1) * points_per_side
        if side_i == 0:
            points[start:end, 0] = side_coords
            points[start:end, 1] = 0.5
        elif side_i == 1:
            points[start:end, 0] = 0.5
            points[start:end, 1] = -side_coords
        elif side_i == 2:
            points[start:end, 0] = -side_coords
            points[start:end, 1] = -0.5
        elif side_i == 3:
            points[start:end, 0] = -0.5
            points[start:end, 1] = side_coords
    if scale is not None:
        points *= scale
    else:
        points *= np.linalg.norm(points, axis=-1).mean()
    angle = np.deg2rad(angle_in_degrees)
    sin = np.sin(angle)
    cos = np.cos(angle)
    rot = np.array([[cos, -sin], [sin, cos]])
    points = points @ rot.T
    recovered_unn_ts = np.concatenate((points.reshape(unn_ts.shape[0], 2 * n_points_per_loop), unn_ts[:, -1:]), axis=-1)
    return recovered_unn_ts

def scale_shift_loop(unn_ts: np.ndarray, scalex: float, scaley: float, dx: float, dy: float):
    """
    helper function for loop editing: scale and shift a loop in the plane coordinate system
    unn_ts is an unnormalized timestep representation vector, with shape (1, 2N+1)
    """
    unn_ts[:, 0:-1:2] = unn_ts[:, 0:-1:2] * scalex + dx
    unn_ts[:, 1:-1:2] = unn_ts[:, 1:-1:2] * scaley + dy
    return unn_ts

# edit this function to experiment with loop intervention!!
def loop_generation_intervention(dataset, ts_i_state, ts):
    # ts is the newly generated timestep, of shape (1, 2N+1), still normalized.
    # ts_i_state() is the current loop's index in the sequence
    
    # Some preliminaries before we can edit.
    ts_i = ts_i_state()  # grab the index of the current timestep we're editing
    # unnormalize ts temporarily, for easier editing on world-coords scale
    unn_ts = ts.cpu().detach().numpy() * dataset.dataset_std + dataset.dataset_mean
    # =========================================================================

    # (note that ts is the still-normalized data. Experiments should generally be done
    # by editing the unnormalized ts, unn_ts. Make sure to remove the renormalize code
    # below if re-running the experiments that edit ts directly.)
    # Moreover, for N=32, ts and unn_ts are of shape (1, 65). 1 is a batch size of 1.
    # Here are some useful arrays to use:
    # > unn_ts[:, :64] (or unn_ts[:, :-1]) are all point coordinates, 
    # > unn_ts[:, -1] is the plane flag,
    # > unn_ts[:, 0:-1:2] is the array of all x coordinates,
    # > unn_ts[:, 1:-1:2] is the array of all y coordinates.

    # # Example of a basic edit:
    # if ts_i in range(4,5):
    #     unn_ts[:, :64] = unn_ts[:, :64] * 0.2 + 0.15
    # (try other offsets)

    # # Example: Transplant a loop (saved previously) into a different mesh
    # hexagony_pred_seq = np.loadtxt("inference/loop-editing-tests/hexagony-transplant-into-widepanbase-10-11/hexagony-original-predseq.txt")
    # if ts_i in range(15, 18):
    #     # transplant by copying a loop from the loaded sequence into this current one
    #     unn_ts[:, :] = hexagony_pred_seq[ts_i]
    #     unn_ts[:, :-1] *= 0.3

    # blenderprocvase_gt3 = np.loadtxt("checkpoints/remote/transformers/blender-procvases2-40p-FRP-EOS-512x8-lat64-8head-12-21/inference/gt3/gt3-pred-seq.txt")
    # # center it
    # copied_xes = blenderprocvase_gt3[:,0:-1:2]
    # copied_yes = blenderprocvase_gt3[:,1:-1:2]
    # copied_mean_x = np.mean(copied_xes)
    # copied_mean_y = np.mean(copied_yes)
    # blenderprocvase_gt3[:,0:-1:2] -= copied_mean_x
    # blenderprocvase_gt3[:,1:-1:2] -= copied_mean_y
    # # copyrange = (18,27)
    # copyrange = (21, 29)
    # targetrange_start = 8
    # if ts_i in range(targetrange_start, targetrange_start + (copyrange[1] - copyrange[0])):
    #     # transplant by copying from the loaded pred seq
    #     unn_ts[:, :] = blenderprocvase_gt3[ts_i + (copyrange[0] - targetrange_start)]
    #     unn_ts[:, :-1] *= 0.7


    # Handle transplant:
    # good_2handles_pred_seq = np.loadtxt("inference/good-2handles-sample-yeah/inference-0-predseq.txt")
    # if ts_i in range(22, 25):
    #     # for "widepanbase", range(19,24) works very well!
    #     # transplant by stealing a loop from the loaded sequence into this current one!
    #     unn_ts[:, :] = good_2handles_pred_seq[ts_i] * 1.0

    # Example: Self-transplant (copying out the untampered predicted sequence of a shape, then
    # not just transplanting some loops from that but also edit them, to see
    # The reason we need this is the
    # above examples will modify the newly
    # generated loops on the fly (causing a cascading effect on subsequent
    # generated loops which are then edited as well), and not simply inject
    # modified versions of loops from the original predicted sequence.)
    # funnytarget_pred_seq = np.loadtxt("inference/loop-editing-tests/transplant-handles-test-10-29/funnytarget-predseq.txt")
    # if ts_i in range(22,25):
    #     unn_ts[:, :] = funnytarget_pred_seq[ts_i]
    #     unn_ts[:, :64] = unn_ts[:, :64] * 0.13 + 0.16
    # for this experiment (funnytarget-modded-from-original-range12to17) this worked well:
    # if ts_i in range(12,17):
    #     unn_ts[:, :] = funnytarget_pred_seq[ts_i]
    #     unn_ts[:, :64] = unn_ts[:, :64] * 0.13 - 0.27

    # 3d) sofa cross section transplant test, from "flatty" to "target" (11-08)
    # flatty_pred_seq = np.loadtxt("inference/loop-editing-tests/transplant-flat-to-squarey-11-08/flatty-pred-seq.txt")
    # if ts_i in range(8, 15):
    #     unn_ts[:, :] = flatty_pred_seq[ts_i - 6]
    # 3d.1) sofa cross section transplant FROM Ground truth sofa index 2
    # gt2_pred_seq = np.loadtxt("inference/loop-editing-tests/transplant-flat-to-squarey-11-08/gt2-pred-seq.txt")
    # if ts_i in range(8, 14):
    #     unn_ts[:, :] = gt2_pred_seq[ts_i]
    #     unn_ts[:, 1:-1:2] *= 0.85
    #     unn_ts[:, 1:-1:2] -= 0.1


    # =========================================================================
    # then normalize the unn_ts (which has been edited) to return it back.
    ts = torch.from_numpy((unn_ts - dataset.dataset_mean) / (dataset.dataset_std)).to(ts)
    # ========================================================================

    # =========================================================================
    # increment "the state". 
    # (the state just holds the index of the current loop in the
    # sequence) we have to do this fancy currying shenanigans because
    # the call signature for the function to feed into the inference
    # function in loop_models.py only has one argument, so we
    # 'partially apply' this function with some outer stateful
    # variable/reference so that this function can refer to the outside
    # world while it runs during the inference
    ts_i_state.modify(lambda u: u+1)
    return ts
