import numpy as np
import torch
from functools import partial

import loop_models as lmodels
import loop_model_options as loptions
import loop_representations as lreprs
import get_loops
import meshlab_poisson_reco
import utils   # for StatefulValue

# contains a configuration-data class as well as a whole REPL/interactive
# command line processor for experimenting with inference!! See that module
# for docs. ( To use, run this module the same way as before, with all
# model-defining arguments, but instead of having to specify inference options
# inside this file, now we have a console to submit and view inference runs.)
import loop_inference_config_and_repl as linferconfig  

from thlog import *


thlog = Thlogger(LOG_DEBUG, VIZ_DEBUG, "infer", 
    imports=[lmodels.thlog, lreprs.thlog, get_loops.thlog])

########## Inference code ##############

def do_lstm_inference( model
                     , dataset_to_get_norm: lmodels.LoopSeqDataset
                     , start_split: np.ndarray # (dataset.n_input_features)
                     , latent_z: np.ndarray # (opt.latent_size)
                     , binary_flags_to_use: np.ndarray = None # (n_steps, 1)
                     , fn_after_each_timestep=None
                     ):
    """ Run LSTM inference with the decoder of the model. 
    Parameters:
    - model: an instance of the model class
    - dataset_to_get_norm: the Dataset object to get the normalization stats 
        (mean, stdev). Should be the same one used to train the model
    - start_split: a vector of the same shape as a single timestep in the model's
        inputs. Used as the "start-token" of the LSTM sequence decoding
    - latent_z: latent code to seed the decoder
    - binary_flags_to_use: (obsolete, don't use this argument)
    - fn_after_each_timestep (default=None): a function that should take one 
        argument being a newly generated timestep of shape (1, features) and
        returns with the same shape a possibly modified time step. This
        function is applied to each newly generated timestep before it is
        autoregressively fed into the decoder again for the next timestep.
    """
    n_time_steps = dataset_to_get_norm.n_steps + 1 
    # the start split is now the dummy split of all zeroes. We have to predict
    # all steps 1..L now
    current_generated_seq = np.expand_dims(start_split, axis=0)
    if dataset_to_get_norm.do_norm and not np.all(start_split == 0):  
        #  the 2nd clause is because the Start Of Sequence (dummy) split step is
        #  always 0, regardless of norm scaling/shifting
        current_generated_seq -= dataset_to_get_norm.dataset_mean
        current_generated_seq = \
            current_generated_seq / dataset_to_get_norm.dataset_std
    curr_state_tuple = None
    for i in range(1, n_time_steps):
        # thlog.info(f"predicting time step {i}")
        curr_data_item = {'inp': current_generated_seq[-1:], 'z': latent_z}
        if binary_flags_to_use is not None:
            # 2022/5/20: binary flags from ground truth, to see if the flags
            # is what is bottlenecking
            curr_data_item['inp'][0][-1] = binary_flags_to_use[i-1]
        if isinstance(model, lmodels.LoopSeqEncoderDecoderModel):
            latest_generated_step, (curr_h_state, curr_c_state), _ = \
                model.run_decoder_one_step(curr_data_item, 
                    prev_states=curr_state_tuple)

            # special handling for the binary-levelup-flag representation
            
            if lreprs.loop_repr_uses_binary_levelup_flags(model.opt.loop_repr_type):
                # the model does not apply sigmoid here, so we must do it manually,
                # to match the training scenario (for autoregressive feeding)
                latest_generated_step[:,:,-1] = torch.sigmoid(latest_generated_step[:,:,-1])
            # latest_generated_step.shape = (1,1,features)
            latest_generated_step[0] = \
                (fn_after_each_timestep if callable(fn_after_each_timestep) \
                else (lambda u: u))(latest_generated_step[0])

            
        else: 
            # to support the older models that have the decoder exposed from
            # forward_and_backward instead of the EncDec pair which runs both 
            latest_generated_step, (curr_h_state, curr_c_state), _ = \
                model.forward_and_backward(curr_data_item, 
                    prev_states=curr_state_tuple)
        curr_state_tuple = (curr_h_state, curr_c_state)
        current_generated_seq = \
            np.concatenate((current_generated_seq, 
                latest_generated_step.cpu().detach().squeeze(1).numpy()))
        
        
    if dataset_to_get_norm.do_norm:
        # unnormalize
        current_generated_seq *= dataset_to_get_norm.dataset_std
        current_generated_seq += dataset_to_get_norm.dataset_mean

    return current_generated_seq[1:] # to get rid of the dummy step 
    # return shape is (timesteps, n features)

def do_transformer_inference( model
                     , dataset_to_get_norm: lmodels.LoopSeqDataset
                     , latent_z: np.ndarray # (opt.latent_size)
                     , fn_after_each_timestep=None
                     ):
    """  
    Parameters:
    - model: model class (LoopSeqEncoderDecoderModel)
    - dataset_to_get_norm: dataset to use to obtain normalization stats (mean, std)
        to unnormalize the generated results
    - latent_z: np array of size (opt.latent_size), to use to seed the decoder
    - fn_after_each_timestep (default=None): see docs for do_lstm_inference;
        a function that takes only one argument being a timestep (1, features)
        and returns with the same shape a possibly modified time step. This
        gets applied after each new timestep before it is autoregressively
        fed into the decoder for the next timestep.
        
    """
    assert lmodels.architecture_type_has_a_transformer_decoder(model.opt.architecture), \
        "model architecture needs to have a transformer decoder to run this function"
    
    # the dummy zeros start-of-sequence step counts now, hence +1
    n_time_steps = dataset_to_get_norm.n_steps + 1
    # but we get rid of the start-of-sequence when we return 
    pred_seq = model.run_transformer_inference(latent_z, n_time_steps, 
        fn_after_each_timestep=fn_after_each_timestep)[1:]
    pred_seq = pred_seq.cpu().detach().numpy()
    if dataset_to_get_norm.do_norm:
        # unnormalize
        pred_seq *= dataset_to_get_norm.dataset_std
        pred_seq += dataset_to_get_norm.dataset_mean
    return pred_seq
    


def do_reconstruction_test(
    model: lmodels.LoopSeqEncoderDecoderModel, 
    dataset: lmodels.LoopSeqDataset, 
    seq_i: int,
    no_random_sampling: bool
    ):
    """ Check teacher-forced reconstruction performance by the model. Feeds in a data item
    (like it would be during training) and returns the latent vector predicted by the
    encoder as well as the teacher-forced decoder output """
    data_item = dataset[seq_i]
    
    pred_seq, _,_, latent_z = \
     model.forward_and_backward(data_item, is_train=False, return_latent=True, 
        no_random_sampling=no_random_sampling)
    
    #  manual sigmoiding here, too, because the sigmoid is not part
    # of the nn.module (just like in do_lstm_inference)
    if lreprs.loop_repr_uses_binary_levelup_flags(model.opt.loop_repr_type):
        pred_seq[:,:,-1] = torch.sigmoid(pred_seq[:,:,-1])
    thlog.trace(f"Reconstruction: encoder generated this latent vector: \n{latent_z}")
    pred_seq = pred_seq.cpu().detach().numpy()
    
    if dataset.do_norm:
        pred_seq *= dataset.dataset_std
        pred_seq += dataset.dataset_mean
    
    return pred_seq.squeeze(1), \
        latent_z.cpu().detach().squeeze(0).numpy()


def run_inference_and_viz(model: lmodels.LoopSeqEncoderDecoderModel, 
                          dataset: lmodels.LoopSeqDataset,
                          latent_z: np.array,    # opt.latent_size
                          loop_repr_type: int,    # see lreprs
                          using_the_sequence: np.array = None,
                          accept_any_sequence_length: bool = False,
                          save_filename_suffix: str = None, # if None no saving; if a string then save contour and obj files
                          fn_after_each_timestep = None 
                          # either None or a callable to apply to each timestep
                          # before it is autoregressively fed into the next
                          # decoder timestep
                         ):
    """
    Main entry point for inference; also handles reconstruction into mesh slices,
    polyscope visualizations, and saving to .contour and .obj files.

    Parameters
    - model: an instance of the model class (can be None if using_the_sequence is provided)
    - dataset: dataset to get the normalization stats to unnormalize the model's output
    - latent_z: latent code to seed the decoder (can be None if using_the_sequence is provided)
    - loop_repr_type: loop representation to recover from the predicted data.
        (see loop_representations.py for more info)
    - using_the_sequence (default=None): bypass autoregressive decoding and just use
        the given sequence data to the visualizer/file-saving steps. Useful 
        for checking out dataset ground truth reconstructions or teacher-forced
        predictions from do_reconstruction_test
    - accept_any_sequence_length (default=False): if False, throw an error if the predicted
        or manually-provided sequence does not match the expected sequence length
        and n_features as specified in the dataset. If True, any sequence length
        is fine (though the number of features per timestep must still be right)
    - save_filename_suffix (default=None): if None, don't save anything. Otherwise,
        this will be appended to the file name of the saved .contour and .obj files.
        These output files will be saved in the checkpoint directory of the model.
    - fn_after_each_timestep (default=None): if a callable, then apply this
        function to each timestep (of shape (1, features), note the batch size of 1))
        before that timestep is autoregressively fed into the decoder again 
        to generate the next timestep.

    Returns (predicted_sequence, latent_z (if applicable; None otherwise))
    """
    if thlog.guard(VIZ_NONE, needs_polyscope=True):
        ps.remove_all_structures()
    if using_the_sequence is None:
        if model.opt.architecture in ("lstm", "lstm+transformer"):
            # for the LSTM we don't have learned start embeddings, we just used 
            # allzero.
            start_split = np.zeros(dataset.n_input_features) 
            pred_seq = do_lstm_inference(model, dataset, start_split, latent_z, 
            fn_after_each_timestep=fn_after_each_timestep)
        elif model.opt.architecture in ("transformer", "pointnet+transformer") :
            # for transformers, we do have a learned start embedding, and the
            # model will fill that in for us so no need to make start_split here
            pred_seq = do_transformer_inference(model, dataset, latent_z, 
            fn_after_each_timestep=fn_after_each_timestep)
        else:
            raise NotImplementedError(f"unhandled inference functionality for architecture type {model.opt.architecture}")
    elif using_the_sequence.shape == (dataset.n_steps,) :
        raise NotImplementedError("deprecated")
        pred_seq = do_lstm_inference(model, dataset, start_split, latent_z, 
        binary_flags_to_use=using_the_sequence, fn_after_each_timestep=fn_after_each_timestep)
    else:
        pred_seq = using_the_sequence
    if not accept_any_sequence_length:
        assert pred_seq.shape == (dataset.n_steps, dataset.n_input_features), \
            "pred_seq wrong shape,  shape is {}".format(pred_seq.shape)
    
    if thlog.guard(VIZ_INFO, needs_polyscope=True):
        # clear out any polyscope things still left over from a prev run
        # (since the EllipseMultiple/EllipseSingle processing will re-register
        # curve nets and points etc for this run of run_inference_and_viz)
        ps.remove_all_structures()

    ### Switch the pointcloud recovery process depending on the loop repr used
    if loop_repr_type == lreprs.LOOP_REPR_ELLIPSE_SINGLE:
        loop_repr_object = \
            lreprs.EllipseSingle(dataset.planes, pred_seq)

    elif loop_repr_type == lreprs.LOOP_REPR_ELLIPSE_MULTIPLE:
        loop_repr_object = \
            lreprs.EllipseMultiple(dataset.planes, pred_seq, 
            segmentize_resolution=64, n_points_to_sample_for_point_cloud=64, 
            use_eos_token=model.opt.use_eos_token)

    elif loop_repr_type == lreprs.LOOP_REPR_FIXED_RES_POLYLINE:
        loop_repr_object = \
            lreprs.EllipseMultiple(dataset.planes, pred_seq,
            segmentize_resolution=64, n_points_to_sample_for_point_cloud=48,
            run_in_fixed_resolution_polylines_mode=True,
            min_n_points_per_loop=24,
            use_eos_token = model.opt.use_eos_token,
            postprocessing_heuristics=
                [ 'caps', 'smooth-normals']
            ) 
    else:
        raise NotImplementedError(f"loop representation {loop_repr_type} not"
                                   "yet implemented")
    
    all_points, all_normals = loop_repr_object.get_oriented_point_cloud()
    # new 2022-08-24; reconstruction contour file format
    # 2022-09-20: we could use the npz slices format instead 
    if save_filename_suffix is not None:
        loptions.mkdir(os.path.join(model.save_dir, "inference"))
        # loop_repr_object.export_as_contour_file(
        #     os.path.join(model.save_dir, "inference", f"inference-{save_filename_suffix}.contour"))
        loop_repr_object.export_as_npz_file(
            os.path.join(model.save_dir, "inference", f"inference-{save_filename_suffix}-slices.npz"))
    
    if thlog.guard(VIZ_INFO, needs_polyscope=True) or save_filename_suffix is not None:
        # normalize all the normal vectors for output
        all_normals /= np.transpose(
            np.tile(np.linalg.norm(all_normals, axis=1), (3, 1)))

        reco_vertices, reco_faces = meshlab_poisson_reco.do_poisson_reco(
            all_points, all_normals, 
            save_filename=os.path.join(model.save_dir, "inference",
                          f"inference-{save_filename_suffix}.obj") \
                          if save_filename_suffix is not None else None)
        if thlog.guard(VIZ_INFO, needs_polyscope=True):
            ps.register_surface_mesh("reconstructed", reco_vertices, reco_faces,
                color=utils.PS_COLOR_SURFACE_MESH)
            ps.show()
    else:
        # no outputs (neither polyscope or .obj filesaving are enabled...)
        thlog.info("WARNING: no output formats (polyscope or .obj saving) are enabled.")
    
    return pred_seq, latent_z

    


def inference_sub_main(opt, 
    model: lmodels.LoopSeqEncoderDecoderModel,
    reference_dataset: lmodels.LoopSeqDataset, 
    inference_cfg: linferconfig.LoopInferenceSettings,
    inference_repl: linferconfig.InferenceREPL):
    """ 
    As side effects, this sets the "last_z" and "last_pred_seq" variables in
    inference_cfg_and_repl's REPL (interactive console) state.
    """
    seed_test_data_item = reference_dataset[inference_cfg.TEST_DATA_ITEM]
    
    # test reco
    if inference_cfg.JUST_VIEW_GT:
        data_item = reference_dataset[inference_cfg.TEST_DATA_ITEM]
        data_seq = data_item['trg']
        data_seq *= reference_dataset.dataset_std
        data_seq += reference_dataset.dataset_mean
        data_seq = data_seq.squeeze(1)
        # thlog.debug(f"ground truth trg seq\n\n{data_seq}\n")

        # queue up this inference-and-viz execution in the REPL so the user
        # can step through them one-by-one at their command
        inference_repl.queue_up_inference_and_viz_action(inference_cfg, lambda:\
            run_inference_and_viz(model, reference_dataset, None, opt.loop_repr_type
            , using_the_sequence=data_seq, accept_any_sequence_length=True
            , save_filename_suffix=(f"gt{inference_cfg.TEST_DATA_ITEM}" \
                if inference_cfg.SAVE_CONTOUR_AND_OBJ_FILE else None)))
        
    elif inference_cfg.JUST_VIEW_THIS_SEQUENCE_DATA is not None:
        data_seq = inference_cfg.JUST_VIEW_THIS_SEQUENCE_DATA
        if (len(data_seq.shape) != 2) or (data_seq.shape[1] != reference_dataset.n_input_features):
            thlog.err(f"Invalid shape for the sequence data-viewing configuration; must be (n_loops, {reference_dataset.n_input_features})")
            return
        inference_repl.queue_up_inference_and_viz_action(inference_cfg, lambda:\
            run_inference_and_viz(model, reference_dataset, None, opt.loop_repr_type
            , using_the_sequence=data_seq, accept_any_sequence_length=True
            , save_filename_suffix=("seqviz" \
                if inference_cfg.SAVE_CONTOUR_AND_OBJ_FILE else None)))

    elif inference_cfg.DOING_RECO_TEST:
        # do this first (run thru encoder) to get the corresponding latent vector
        # (and if we choose to  not use autoregressive decoding, this will also give back
        # the teacher-forced decoded sequence)
        teacher_forced_pred_seq, reco_latent_z = \
            do_reconstruction_test(model, reference_dataset, 
            inference_cfg.TEST_DATA_ITEM, inference_cfg.RECO_WITH_NO_SAMPLING_USING_SIGMA)

        if inference_cfg.RECO_TEST_AUTOREGRESSIVE:
            using_the_sequence = None
            #if GROUND_TRUTH_BINARY_FLAGS:
            # # note this is actually wrong, so I've commented this option out for now...
            # # (as written, this is taking the binary flags from the teacher-forced reco, not the
            # # actual ground truth!)
            #    using_the_sequence = data_seq[:, -1]
        else:
            using_the_sequence = teacher_forced_pred_seq

        # queue up this inference-and-viz execution in the REPL so the user
        # can step through them one-by-one at their command
        inference_repl.queue_up_inference_and_viz_action(inference_cfg, lambda:\
            run_inference_and_viz(model, reference_dataset, reco_latent_z, opt.loop_repr_type
            , using_the_sequence=using_the_sequence
            , save_filename_suffix=(f"reco{inference_cfg.TEST_DATA_ITEM}" \
                if inference_cfg.SAVE_CONTOUR_AND_OBJ_FILE else None)
            ))
    else:
        
        # 2022-09-15 ==========================================================
        # testing 'intervention functions' executed/injected during
        # autoregressive loop-sequence generation
        def __test_intervention_fn__with_state(reference_dataset: lmodels.LoopSeqDataset, ts_i_state: utils.StatefulValue, ts):
            """ this is intended to be partially applied ('curried'); the 
            inference routine will only see and provide the 'ts'.
            """
            return inference_repl.loop_generation_intervention(reference_dataset, ts_i_state, ts)
        
        def __reset_test_intervention_state(ts_i_state):
            ts_i_state.put(0)
            
        __test_intervention_fn_state = utils.StatefulValue(0)
        __test_intervention_fn_state.store("latent_size", opt.latent_size)
        # partially apply the dataset and stateful value, so that
        # the intervention function only needs to take one argument, as needed
        __test_intervention_fn = partial(__test_intervention_fn__with_state, reference_dataset, __test_intervention_fn_state)
        
        # set to None to not intervene at all
        if not inference_cfg.DO_LOOP_INTERVENTION_EXPERIMENT:
            __test_intervention_fn = None 
        # ====================================================================== 

        # running list of closest-match GT data items, as a diversity measure
        inference_repl.variable_bind("closest_data_matches", [])
        latent_zs_to_test = [np.random.normal(0.0, 1.0, size=opt.latent_size) 
                            for _ in range(inference_cfg.N_SAMPLES)]
        if inference_cfg.DOING_LERP_SAMPLING_TEST:
            latent_zs_to_test = np.linspace(
                inference_cfg.LERP_SAMPLING_START_Z,
                inference_cfg.LERP_SAMPLING_END_Z, 
                inference_cfg.N_SAMPLES)
        for viz_i, latent_z_to_test in enumerate(latent_zs_to_test):

            def __planned_inference_and_viz(my_viz_i, my_latent_z_to_test):
                __test_intervention_fn_state.store("latent_z", my_latent_z_to_test)
                pred_seq, latent_z = run_inference_and_viz(model, reference_dataset, 
                    my_latent_z_to_test,
                    opt.loop_repr_type,
                    save_filename_suffix = (str(my_viz_i) if inference_cfg.SAVE_CONTOUR_AND_OBJ_FILE else None),
                    fn_after_each_timestep=__test_intervention_fn 
                    )
                
                # (also part of 2022-09-15 loop intervention tests)
                # in our case we want to reset the intervention function's state
                # each new mesh, since this state counts up the indices of loops
                # within each sequence; it should not have anything to do with
                # other/prev meshes
                __reset_test_intervention_state(__test_intervention_fn_state)

                if inference_cfg.SAVE_CONTOUR_AND_OBJ_FILE:
                    # save the latent too while we're at it
                    np.savetxt(os.path.join(model.save_dir, "inference",
                            f"inference-{my_viz_i}-latent.txt"), my_latent_z_to_test)

                
                closest_data_matches = inference_repl.variable_get("closest_data_matches")
                closest_data_matches.append(reference_dataset.find_closest_data_item(pred_seq))

                closest_data_matches_as_set = set(map(lambda t: t[1], 
                    closest_data_matches))
                inference_repl.variable_bind("diversity_stats", 
                    f"Closest data item matches: {closest_data_matches}, mean distance {np.mean(list(map(lambda t: t[0], closest_data_matches)))}\n"
                    f"Unique data items: {closest_data_matches_as_set}, "
                    f"representing {100*len(closest_data_matches_as_set)/len(closest_data_matches)}% of all matches."
                )
                
                return pred_seq, latent_z
            
            # queue up this inference-and-viz execution in the REPL so the user
            # can step through them one-by-one at their command
            inference_repl.queue_up_inference_and_viz_action(inference_cfg,
                partial(__planned_inference_and_viz, viz_i, latent_z_to_test))

        # diversity stats are now calculated on the fly and stored in the REPL's
        # diversity_stats variable.


# interactive inference repl 
def inference_main():
    np.random.seed()
    torch.set_printoptions(precision=12)
    thlog.init_polyscope()
    
    opt = loptions.LoopSeqOptions().parse_cmdline()
    opt.batch_size = 1
    # this will get the mean and std of whichever dataroot subfolder is
    # specified by opt / by the override_mode kwarg. In inference we would
    # like to test reconstructions of data from the train set hence override
    reference_dataset = lmodels.LoopSeqDataset(opt, override_mode='train')
    # however, if we DO want to view reconstructions/gt from the test set, the
    # --load_test_set_for_inference option is available:
    if opt.load_test_set_for_inference:
        has_test_dataset = os.path.isdir(os.path.join(opt.dataroot, 'test'))
        if not has_test_dataset:
            thlog.err("Specified --load_test_set_for_inference but there is no test directory in --dataroot. Falling back to train set instead.")
        else:
            # This assumes that the train set's seq len is BIGGER than that
            # of the test set. This may not always be true! but it is true for
            # all the datasets I'm using so far (11-08) so we'll work with this
            # for now. (This assumption does not exist in loop_models_main.py so
            # you could still train with such a pair of train-test sets.)
            reference_dataset = lmodels.LoopSeqDataset(opt, override_mode='test'
                , pad_to_max_sequence_length=reference_dataset.n_steps
                , override_mean_std=(reference_dataset.dataset_mean, reference_dataset.dataset_std))
            thlog.info("Loaded test set instead, since --load_test_set_for_inference is specified.")

    inference_repl = linferconfig.InferenceREPL(opt)

    model = lmodels.LoopSeqEncoderDecoderModel(opt, 
    reference_dataset.n_input_features, reference_dataset.n_steps)

    model.print_detailed_network_stats()
    
    thlog.info("Loaded model. Starting interactive inference REPL :)")
    while True:
        try:
            repl_line = input(inference_repl.prompt)
            inference_cfg = inference_repl.parse_inference_repr_line(repl_line)
            if inference_cfg is not None:
                inference_sub_main(opt, model, reference_dataset, inference_cfg, inference_repl)
        except Exception as e:
            if isinstance(e, EOFError) or isinstance(e, KeyboardInterrupt):
                inference_repl.repl_print("\nQuitting inference console.", is_error=True)
                break
            else:
                inference_repl.repl_print(f"caught an error while running the inference task:\n {repr(e)}", is_error=True)
                raise e
                # continue

if __name__ == "__main__":
    inference_main()