# this module is intended to be hot-reloadable during runtime of 
# the InferenceREPL, called from loop_inference_config_and_repl.py.
import numpy as np
import torch

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
    # if ts_i in range(1,4): 
    #     unn_ts[:, :64] = unn_ts[:, :64] * 0.4 + 0.12 
    # (try other offsets, e.g. 0.05 to 0.065 to 0.08)

    # # Example: Transplant a loop (saved previously) into a different mesh
    # hexagony_pred_seq = np.loadtxt("inference/loop-editing-tests/hexagony-transplant-into-widepanbase-10-11/hexagony-original-predseq.txt")
    # if ts_i in range(15, 18):
    #     # transplant by copying a loop from the loaded sequence into this current one
    #     unn_ts[:, :] = hexagony_pred_seq[ts_i]
    #     unn_ts[:, :-1] *= 0.3

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
