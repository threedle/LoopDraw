import os
import importlib
import numpy as np
import loop_inference_intervention_func


class LoopInferenceSettings:
    def __init__(self):
        self.JUST_VIEW_GT = False 
        """ 
        specify True to override all other inference options and just view the
        ground truth data item with index specified in TEST_DATA_ITEM below.
        Overrides all the other options except for TEST_DATA_ITEM which is used.
        """

        self.JUST_VIEW_THIS_SEQUENCE_DATA = None
        """
        specify an np.array of shape (n_loops, n_features) describing a single loop
        in order to view it in isolation, with no reconstruction etc. Overrides
        all the other options, but lower priority and overridden by JUST_VIEW_GT
        if that's set to True.
        """

        # inference options;
        self.DOING_RECO_TEST = False
        """
        Set to True in order to run reconstruction tests; if this is True, must
        provide values for RECO_TEST_AUTOREGRESSIVE and TEST_DATA_ITEM.
        (TEST_DATA_ITEM must be in the range of the dataset class when assuming
        a batch size of 1). Reconstruction involves passing
        dataset[TEST_DATA_ITEM] into the encoder, sampling a latent vector using
        these predictions (as opposed to sampling directly from the Normal(0,1)
        prior, which is what would happen with DOING_RECO_TEST = False), then
        running the decoder using this latent vector. if
        RECO_TEST_AUTOREGRESSIVE is True, the decoder must predict all time
        steps using its own outputs from previous time steps; else, predictions
        for time step i are made using the ground-truth timestep i-1.

        If set to True this will disregard all the latent decoding options
        (including DOING_LERP_SAMPLING_TEST, LERP_SAMPLING_START_Z,
        LERP_SAMPLING_END_Z, DECODE_CUSTOM_LATENT)
        """
        # reconstruction options

        self.RECO_WITH_NO_SAMPLING_USING_SIGMA = True
        """
        if RECO_WITH_NO_SAMPLING_USING_SIGMA is True, then only the encoder's
        predicted mu will be returned as the latent vector for use by the
        decoder. This is to check whether the sigma (sampling variance) is
        what's messing up reconstruction each new run.
        """

        self.RECO_TEST_AUTOREGRESSIVE = True
        """ 
        If this is False, the model will only ever predict 1 step ahead; each
        previous step is provided by ground truth. Only meaningful if
        DOING_RECO_TEST is True.
        """


        # non-reco/sampling options
        self.DOING_LERP_SAMPLING_TEST = False
        """
        Set to True in order to see how the model decodes a set of latents that
        lerps from one random vector to another in Normal(0,1). Only works if
        DOING_RECO_TEST is False.
        """

        # these are only relevant if DOING_LERP_SAMPLING_TEST is True.
        self.LERP_SAMPLING_START_Z = None
        self.LERP_SAMPLING_END_Z = None


        self.DECODE_CUSTOM_LATENT = None
        """
        If this is not None, the latent z specified here (of shape latent_size)
        will be input as a custom latent code for the model to decode.
        """

        self.N_SAMPLES = 25
        """
        For when DOING_RECO_TEST is False; number of samples to draw from prior
        or interpolate between two random latent codes
        """

        #151 #50 is the nice connected handle vase.. # let's use 67
        self.TEST_DATA_ITEM = 0   
        """
        Dataset index for the test item to reconstruct. (Index with respect to a
        dataset with batch size of 1.)
        """


        self.DO_LOOP_INTERVENTION_EXPERIMENT = False
        """
        Specify True in order to apply the 'loop generation intervention
        function' (during autoregressive generation. (see
        loop_models_inference docs for more on this)
        (this is an 'optional keyword argument' in the 'submit' command.)
        """

        self.SAVE_CONTOUR_AND_OBJ_FILE = False
        """ save as .contour file and meshlab poisson-reconstructed .obj file
        """

        # this is not a config option used by run_inference_and_viz, 
        # only used by InferenceREPL for recordkeeping in the queue
        self.invoking_command = ""
    
    def __str__(self):
        r = ""
        if self.JUST_VIEW_GT:
            r += f"[ground-truth viewing] of data item {self.TEST_DATA_ITEM} from the dataset. "
        elif self.DOING_RECO_TEST:
            r += "[reconstruction], "\
                f"{'autoregressive' if self.RECO_TEST_AUTOREGRESSIVE else 'teacher-forced'}, "\
                f"of data item {self.TEST_DATA_ITEM} of the dataset; invoked from `{self.invoking_command}`. "
        else:
            r += f"[decoding] invoked from `{self.invoking_command}`. "
        if self.DO_LOOP_INTERVENTION_EXPERIMENT:
            r += "Loop intervention function will be applied. "
        if self.SAVE_CONTOUR_AND_OBJ_FILE:
            r += "Output .obj, latent.txt, .contour will be saved. "
        
        return r
        

class InferenceREPL:
    def __init__(self, opt):
        # REPL state:
        self.__repl_variable_binds = dict()
        self.__inference_and_viz_queue = []

        # since we'll be dealing with latents a lot in this class
        self.latent_size = opt.latent_size
        self.save_dir = opt.save_dir
        
        # this function is intended to be hot reloadable, so we save a reference
        # to the currently-loaded function
        self.loop_generation_intervention = \
            loop_inference_intervention_func.loop_generation_intervention

        self.prompt = '\033[96m' + '\033[4m' + 'inference' + '\033[0m' + '> '
    
    def repl_print(self, msg, is_error=False):
        OKBLUE = '\033[94m'
        WARNING = '\033[93m'
        ENDC = '\033[0m'
        print((OKBLUE if not is_error else WARNING) + str(msg) + ENDC)

    def variable_bind(self, var_name, var_value):
        self.__repl_variable_binds[var_name] = var_value
    
    def variable_get(self, var_name):
        return self.__repl_variable_binds.get(var_name)
    
    def pop_inference_and_viz_queue(self):
        """ Binds the latest pred_seq and latent_z to the variable names 
        "last_pred_seq" and "last_z"
        """
        if not self.__inference_and_viz_queue:
            self.repl_print("The task queue is currently empty.")
            return

        cfg, action = self.__inference_and_viz_queue.pop(0)
        if cfg.DO_LOOP_INTERVENTION_EXPERIMENT:
            # hot-reload the intervention function from its file
            self.repl_print("Hot-reloading the intervention function!")
            try:
                __reloaded_module = importlib.reload(loop_inference_intervention_func)
                # update our saved reference to the reloaded function. This will
                # be used inside inference.py for its needs.
                self.loop_generation_intervention = \
                    __reloaded_module.loop_generation_intervention
            except Exception as e:
                self.repl_print("Hot-reload failed!! Still using last/currently successfully loaded function.", is_error=True)
                self.repl_print(repr(e))
            

        pred_seq, latent_z = action()
        if latent_z is not None:
            self.variable_bind("last_z", latent_z)
        else:
            self.repl_print("no latent_z reported by latest run, won't be saved into $last_z")
        self.variable_bind("last_pred_seq", pred_seq)


    def queue_up_inference_and_viz_action(self, inference_cfg: LoopInferenceSettings, planned_function):
        self.__inference_and_viz_queue.append((inference_cfg, planned_function))

    

    def parse_inference_repr_line(self, line: str):
        """
        Guide to the interactive inference REPL :-)
        
        Synopsis:
        - use a "submit" command first to submit a job to the queue.
        - use a "run" command to execute items on the queue.

        Commands: submit, run, print, queue, assign, variables, quit
        "submit": submit an inference run to be queued. Usage:
            submit gt <data-item-index-in-dataset>
            submit auto-reco <data-item-index-to-reconstruct>
            submit tf-reco <data-item-index-to-reconstruct>
            submit sample <number of samples to draw>
            submit interp <number of samples to interp> <start z> <end z>
            submit decode <z>
            submit seq-viz <pred-sequence> [start-index] [end-index]

            Latent vector arguments (the <z>s) in the above can be specified in
            4 ways, indicated by 4 different prefix sequences: 
            - $<variable_name> : get the latent vector stored under variable_name
            - ?                : draw a random latent code from standard Normal
            - |<filepath>      : read the latent code from a text file. 
                                 The format is one array entry per line, like
                                 1D arrays saved with np.savetxt. This path is
                                 relative to the current working directory.
            - ||<filepath>     : like single-pipe |<filename> but here the path
                                 is relative to the model's save directory, as
                                 specified in the --save_dir option.
            (you can't hand-write or paste a latent vector directly into the 
            REPL command line.)

            The <pred-sequence> argument can take values from the $last_pred_seq
            variable (i.e. of shape (max_n_timesteps, n_features)). 
            The seq-viz subcommand if [start-index] AND [end-index] are
            specified will only use pred_sequence[start_index:end_index] for
            reconstruction and visualization  (useful for saving out a portion
            of the generated loops for transplanting into other generation runs
            via the loop gen intervention function!)

            Note that the submit command can take additional arguments prefixed
            with -- (argparse/GNU-style). Right now the following such arguments
            are available:
                --intervene: enables application of the 'loop intervention
                function' (see docs in loop_models_inference for more on that;
                the actual intervention function is defined in
                loop_inference_intervention_func.py, which is hot-reloaded every
                time an --intervene run is triggered.)
                --save: save the resulting files (.obj, latent.txt, .contour) to
                disk (to the save_dir of the model and inside an 'inference'
                subdirectory.)
                --use_enc_sigma: by default, the autoregressive reco
                test does NOT use the sigma value predicted by the encoder, only
                using the mean (mu) as the resulting latent vector. 
                This option enables sampling using that predicted sigma too.
                (however, this is irrelevant/unused when the model is a "NoKL" 
                model, i.e. --enc_kl_weight 0. The mu is always the only encoder
                output used when testing auto-reco with those models.)

        "run": pops queued inference tasks and runs them, including viz. Usage:
            run: runs 1 queued task and removes it from the queue.
            run all: go through the entire queue.
            run <number>: runs <number> items from the top of the queue
        
        "print": print a bound variable. Usage:
            print varname [filename]
            
            The optional [filename] argument if present will be where the
            variable contents are saved to. [filename] can be a path relative to
            the current directory, or if prefixed with ||, a path relative to
            the model's checkpoint dir.

            ** Tip: Here is a list of built-in/automatically populated variables
            for you to print/assign:
                - last_z: stores the latent code used in the most recent 
                    sample/interp/decode job
                - last_pred_seq: stores the predicted sequence from the most
                    recent sample/interp/decode job
                - diversity_stats: after each sample/interp/decode job, this
                    variable updates with information on how diverse the meshes
                    generated so far have been, measured as "how many unique
                    dataset meshes are represented in these sampled ones". This
                    variable is reset and reflects the most recently run
                    batch of sample/interp/decode jobs.
        
        "queue": manage the queue. Usage:
            queue       : see an overview of tasks in the queue
            queue pop   : remove a task from the queue without running it
            queue clear : clear the queue
        
        "assign": set a variable's value to a latent vector. Usage:
            assign var_name_destination <z>
            This <z> format is as in the "submit" command: either $<var_name>, 
            or ?, or |<relative_filepath>, or ||<save_dir_relative_filepath>.

        "quit": quit the REPL

        """
        # Returns: the configuration object for the desired run submitted, so
        # that the caller can use it to execute run_inference_and_viz. May also
        # return None, in which case the caller shouldn't do anything.

        if not line:
            return None
        tokens = [tok for tok in line.split(" ") if tok]
        command = tokens[0]
        
        def __read_nparray_option_token(tok: str, should_be_latent_size=True):
            if tok.startswith("$"):
                arr = self.variable_get(tok[1:])
                if arr is None:
                    self.repl_print(f"!! Unknown variable '{tok[1:]}'", is_error=True)
                elif (not isinstance(arr, np.ndarray)) \
                    or (isinstance(arr, np.ndarray) and \
                        (should_be_latent_size and arr.shape[0] != self.latent_size)):
                    self.repl_print(f"!! Value stored as ${tok[1:]} is not a valid latent vector.\n"
                        "(It might be a special builtin variable; see help text)", is_error=True)
                    arr = None
            elif tok == "?":
                arr = np.random.normal(0.0, 1.0, self.latent_size)
            elif tok.startswith("|"):
                fname = os.path.join(self.save_dir, tok[2:]) \
                    if tok.startswith("||") else tok[1:]
                try:
                    arr = np.loadtxt(fname)
                    if should_be_latent_size and (arr.shape[0] != self.latent_size):
                        self.repl_print(f"!! Array loaded from {fname} has "
                            f"the wrong size {arr.shape} while the model's latent size is {self.latent_size}. "
                            "(this might be a special builtin variable; see help text)", is_error=True)
                except FileNotFoundError:
                    self.repl_print(f"!! File '{fname}' not found.", is_error=True)
                    arr = None
            else:
                arr = None
            if arr is None:
                self.repl_print(f"!! Unable to parse latent code/ array from argument '{tok}'", is_error=True)
            return arr

        
        if command == "submit":
            # subcommands: gt <idx>, auto-reco <idx>, tf-reco <idx>, 
            # sample <count>, interp <count> <latent0-path-or-varname> <latent1-path-or-varname>, 
            # decode <latent-path-or-varname>

            if len(tokens) > 1:
                subcommand = tokens[1]
            else:
                self.repl_print(f"!! subcommand needed for command `submit`.", is_error=True)
                return None 
            
            settings_obj = LoopInferenceSettings()
            settings_obj.invoking_command = line
            optional_dashdash_args_start_at = None
            if subcommand == "gt":
                optional_dashdash_args_start_at = 3
                settings_obj.N_SAMPLES = 1
                settings_obj.JUST_VIEW_GT = True
                try:
                    settings_obj.TEST_DATA_ITEM = int(tokens[2])
                except:
                    self.repl_print("!! Unable to parse integer at 2nd argument. Using a default, which is 0", is_error=True)
                    settings_obj.TEST_DATA_ITEM = 0
                
            elif subcommand == "auto-reco" or subcommand == "tf-reco":
                optional_dashdash_args_start_at = 3
                settings_obj.DOING_RECO_TEST = True
                settings_obj.N_SAMPLES = 1
                settings_obj.RECO_TEST_AUTOREGRESSIVE = (subcommand == "auto-reco")
                try:
                    settings_obj.TEST_DATA_ITEM = int(tokens[2])
                except:
                    self.repl_print("!! Unable to parse integer at 2nd argument. Using a default, which is 0", is_error=True)
                    settings_obj.TEST_DATA_ITEM = 0
            
            elif subcommand == "sample":
                optional_dashdash_args_start_at = 3
                settings_obj.DOING_RECO_TEST = False
                settings_obj.DOING_LERP_SAMPLING_TEST = False
                settings_obj.RECO_TEST_AUTOREGRESSIVE = True
                try:
                    settings_obj.N_SAMPLES = int(tokens[2])
                except:
                    self.repl_print("!! Unable to parse integer at 2nd argument. Using a default, which is 25", is_error=True)
                    settings_obj.N_SAMPLES = 25
                            
            elif subcommand == "interp":
                optional_dashdash_args_start_at = 5
                settings_obj.DOING_RECO_TEST = False
                settings_obj.DOING_LERP_SAMPLING_TEST = True
                try:
                    settings_obj.N_SAMPLES = int(tokens[2])
                except:
                    self.repl_print("!! Unable to parse integer at 2nd argument. Using a default, which is 25", is_error=True)
                    settings_obj.N_SAMPLES = 25
                settings_obj.LERP_SAMPLING_START_Z = __read_nparray_option_token(tokens[3])
                settings_obj.LERP_SAMPLING_END_Z = __read_nparray_option_token(tokens[4])
                if (settings_obj.LERP_SAMPLING_START_Z is None or settings_obj.LERP_SAMPLING_END_Z is None):
                    settings_obj = None
                
                
            elif subcommand == "decode":
                optional_dashdash_args_start_at = 3
                settings_obj.DOING_RECO_TEST = False
                settings_obj.DOING_LERP_SAMPLING_TEST = True
                settings_obj.N_SAMPLES = 1
                the_custom_latent = __read_nparray_option_token(tokens[2])
                settings_obj.LERP_SAMPLING_START_Z = the_custom_latent
                settings_obj.LERP_SAMPLING_END_Z = the_custom_latent
                if settings_obj.LERP_SAMPLING_START_Z is None:
                    settings_obj = None
            
            elif subcommand == "seq-viz":
                optional_dashdash_args_start_at = 3
                settings_obj.RECO_TEST_AUTOREGRESSIVE = False
                seq_data_to_view = __read_nparray_option_token(tokens[2], should_be_latent_size=False)
                if seq_data_to_view is None:
                    settings_obj = None
                    return None
                if len(tokens) > 3:
                    try:
                        loop_seq_data_indices = (int(tokens[3]), int(tokens[4]))
                    except:
                        self.repl_print("!! Unable to parse integers at 3rd and 4th arguments. "
                        "This argument pair is optional, so assuming that it is unspecified.", is_error=True)
                        loop_seq_data_indices = None
                    
                    if loop_seq_data_indices is not None:
                        # shape (1, n_features)
                        start_i, end_i = loop_seq_data_indices
                        seq_data_to_view = seq_data_to_view[start_i : end_i]
                settings_obj.JUST_VIEW_THIS_SEQUENCE_DATA = seq_data_to_view
                settings_obj.N_SAMPLES = 1

            else:
                self.repl_print("unknown subcommand", is_error=True)
                settings_obj = None
            

            if settings_obj is not None:
                if settings_obj.N_SAMPLES > 1:
                    self.repl_print(f"{settings_obj.N_SAMPLES} tasks have been queued")
                else:
                    self.repl_print("One new task has been queued")
                
                # now parse the --arg arguments
                if optional_dashdash_args_start_at is not None and \
                    (len(tokens) > optional_dashdash_args_start_at):
                    remaining_tokens = tokens[optional_dashdash_args_start_at:]
                    if "--intervene" in remaining_tokens:
                        settings_obj.DO_LOOP_INTERVENTION_EXPERIMENT = True
                    if "--save" in remaining_tokens:
                        settings_obj.SAVE_CONTOUR_AND_OBJ_FILE = True
                    if "--use_enc_sigma" in remaining_tokens:
                        settings_obj.RECO_WITH_NO_SAMPLING_USING_SIGMA = False


            return settings_obj
        
        elif command == "run":
            if len(tokens) > 1:
                # handle subcommands 'all' and <number>
                if tokens[1] == "all":
                    n_remaining = len(self.__inference_and_viz_queue)
                else:  # tokens[1] is a number possibly
                    try:
                        n_remaining = int(tokens[1])
                    except:
                        self.repl_print("!! Unable to parse integer at argument 1. Using a default, which is 1", is_error=True)
                        n_remaining = 1
                while n_remaining > 0:
                    self.pop_inference_and_viz_queue()
                    n_remaining -= 1
            else:
                self.pop_inference_and_viz_queue()
            
        
        elif command == "print":
            var_value = self.variable_get(tokens[1])
            if var_value is not None:
                self.repl_print(var_value)
                if len(tokens) > 2:
                    fname = tokens[2]
                    if fname.startswith("||"):
                        fname = os.path.join(self.save_dir, fname[2:])
                    try:
                        # if np array then we use np.savetxt, else just use python builtin writes
                        if isinstance(var_value, np.ndarray):
                            np.savetxt(fname, var_value)
                            self.repl_print(f"Saved this array to {fname}")
                        else:
                            with open(fname, "w") as f:
                                f.write(str(var_value))
                            self.repl_print(f"Saved this non-ndarray value to {fname}")
                    except FileNotFoundError:
                        self.repl_print(f"!!directory of {fname} does not exist, file can't be written", is_error=True)
            else:
                self.repl_print(f"!! Unknown variable '{tokens[1]}'", is_error=True)

            
        
        elif command == "assign":
            value_to_store = __read_nparray_option_token(tokens[2], should_be_latent_size=False)
            # the should_be_latent_size = False because we can store whatever
            # size np array; the enforcement to be latent_size is up to those
            # who read the values of stored variables, not the assign function.
            if value_to_store is None:
                self.repl_print(f"!! could not read value '{tokens[2]}' to store as ${tokens[1]}", is_error=True)
            else:
                self.variable_bind(tokens[1], value_to_store)
        
        elif command == "variables":
            if not self.__repl_variable_binds:
                self.repl_print("No bound variables.")
            else:
                self.repl_print("Current bound variables:")
                for key_name in self.__repl_variable_binds.keys():
                    self.repl_print(f"- {key_name}")

        elif command == "queue":
            if len(tokens) > 1:
                subcommand = tokens[1]
                if subcommand == "pop":
                    # pop and don't run
                    if len(self.__inference_and_viz_queue) > 0:
                        self.__inference_and_viz_queue.pop(0)
                        self.repl_print("Popped one task off the queue.")
                    else:
                        self.repl_print("The queue is already empty.")
                elif subcommand == "clear":
                    self.__inference_and_viz_queue = []
                    self.repl_print("Cleared queue.")
                else:
                    self.repl_print(f"Unknown subcommand {subcommand}", is_error=True)
            else:
                # print the str() of the config of all the queued tasks
                self.repl_print(f"Tasks on the queue:")
                for i, (cfg, _) in enumerate(self.__inference_and_viz_queue):
                    self.repl_print(f"{i+1}:" + str(cfg))
                # how many tasks
                n_tasks = len(self.__inference_and_viz_queue)
                self.repl_print(f"The queue has {n_tasks} task{'s' if n_tasks != 1 else ''}.")

        elif command == "quit":
            raise EOFError()
        
        elif command == "help":
            self.repl_print(self.parse_inference_repr_line.__doc__)
        else:
            self.repl_print(f"!! Unknown command '{command}'", is_error=True)

