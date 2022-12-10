import os
from glob import glob
from collections import OrderedDict # for GenericFCNet

import numpy as np
import torch
import torch.nn as nn
import fvcore.nn

import loop_representations as loop_reprs
import loop_model_options as loop_opts
from utils import read_planespec_file, vstack_with_padding, pad
from thlog import *

thlog = Thlogger(LOG_INFO, VIZ_NONE, "model", imports=[loop_reprs.thlog])
np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)

## dataloading for loop representations


def ls_all_datafiles_under_dataroot(dir, ext, recursive=False):
    if recursive:
        original_fpaths = glob(os.path.join(dir, '**', '*{}'.format(ext)), recursive=True)
    else:
        original_fpaths = glob(os.path.join(dir, '*{}'.format(ext)), recursive=False)
    return original_fpaths

def validate_cache_repr_type(cache_npz, loop_repr_type_being_used) -> bool:
    """ returns True if the cache's repr type and our repr type match.
    if there is a 'loop_repr_type' field then match it against
    self.opt.loop_repr_type if there isn't, assume the cache is of
    loop_repr_type == 1 (ellipse-multiple)
    """
    
    loop_repr_type_of_cache = cache_npz.get('loop_repr_type')
    if loop_repr_type_of_cache is None:
        loop_repr_type_of_cache = loop_reprs.LOOP_REPR_ELLIPSE_MULTIPLE
    return loop_repr_type_of_cache == loop_repr_type_being_used

# this dataset class is actually general enough to be used with any sequence data
class LoopSeqDataset(torch.utils.data.Dataset):
    def __init__(self, opt, 
        override_mode = '', 
        override_paths = [], 
        override_mean_std: tuple = None,
        pad_to_max_sequence_length: int = None,
        regenerate_cache = False):
        """ A dataset class for feeding in loop sequence inputs to the model.
        Optional args
        - override_mode :: str. Either 'test' or 'train'; to override the 
            dataset mode despite what opt says
        - override_paths :: List[str]. Build a dataset from a custom list 
            of input files instead of what's in the --dataroot argument in opt.
        - override_mean_std :: Optional[Tuple[Optional[float], Optional[float]]].
            Skip calculating mean and std over the dataset 
            and use some precalculated values instead. This is useful for NOT
            calculating mean/std for the test set but directly using the 
            mean/std calculated from the train set. Specify (None, None)
            to disable normalization altogether.
        - pad_to_max_sequence_length :: Optional[int]:
            Pad all sequences to be the specified max length; must be larger
            than the max length of the actual sequences loaded in the dataset.
            Useful for when the model is trained on a set with max length N
            but the test set has a max length less than N (then we'd load the
            test dataset with pad_to_max_sequence_length=N so that the model
            can run on the shorter-sequence-set too.)
        - regenerate_cache: by default, this will cache the loaded data in an
            npz file in the dataroot/train or dataroot/test directory, for fast
            reloading on later runs/inference tests.
            Specify regenerate_cache=True to ignore this cache file and reload.
            Note that this dataset class saves partial cache checkpoints during
            loading; regenerate_cache will respect and resume from any partial
            cache found. Delete the partial cache file in the dataroot/train (or
            test) directory to make regenerate_cache start over from scratch.
        """
        self.opt = opt
        self.mode = self.opt.mode
        self.batch_size = opt.batch_size
        if override_mode:
            self.mode = override_mode
        self.dir = ""
        if self.mode in ('train', 'test'):
            # dir == "dataroot/train" or "dataroot/test"
            self.dir = os.path.join(self.opt.dataroot, self.mode)
            if override_paths:
                thlog.debug("overriding paths with files from "+override_paths)
                self.obj_paths = ls_all_datafiles_under_dataroot(override_paths, ".obj", recursive=True)
            else:
                self.obj_paths = ls_all_datafiles_under_dataroot(self.dir, ".obj", recursive=True)
        else:
            pass
        
        # check for cached planes + loaded slices
        cache_path = os.path.join(self.dir, "preprocessed_cache.npz")
        partial_cache_path = os.path.join(self.dir, "preprocessed_cache_PARTIAL.npz")
        if os.path.isfile(cache_path) and (not regenerate_cache):
            # COMPLETE cache file exists
            thlog.info(
                f"[DATA ] Cache exists, loading dataset from {cache_path}")
            cache_npz = np.load(cache_path)

            self.planes = cache_npz['planes']
            self.meshes_reprs = cache_npz['meshes_reprs']

            # NOTE versioning now added (08-03)
            if not (validate_cache_repr_type(cache_npz, self.opt.loop_repr_type)):
                thlog.err(f"cached npz file not of the same loop representation"
                            "as the one specified in this run's options!")
                raise ValueError("cached loop representation mismatch with specified representation")
        else:
            # COMPLETE cache doesn't exist or we've forced cache-regenerate
            next_index_to_load = 0
            meshes_reprs_load = None
            failed_meshes_indices = []
            n_skipped_meshes = 0
            self.planes = None

            # (new 2022-08-18) ... but if PARTIAL cache exists, we can resume.
            if os.path.isfile(partial_cache_path):
                partial_cache_npz = np.load(partial_cache_path)
                # first check repr type match
                if validate_cache_repr_type(partial_cache_npz, self.opt.loop_repr_type):
                    thlog.info("[DATA ] Partially-loaded dataset cache is available, resuming from that")
                    # then resume, loading the above state
                    next_index_to_load = partial_cache_npz["next_index_to_load"]
                    meshes_reprs_load = partial_cache_npz["meshes_reprs"]
                    self.planes = partial_cache_npz["planes"]
                    failed_meshes_indices = list(partial_cache_npz["failed_meshes_indices"])
                    n_skipped_meshes = len(failed_meshes_indices)



            # define slice planes for the loops here, if not already loaded
            if self.planes is None:
                planespec_scan_direction, planespec_num_slices, \
                    planespec_min_coord, planespec_max_coord, \
                        = read_planespec_file(os.path.join(self.dir, "planespec.txt"))
                thlog.info("[DATA ] Read planespec file: scan dir " 
                    f"{planespec_scan_direction}, with "
                    f"{planespec_num_slices} slices, "
                    f"min {planespec_min_coord} max {planespec_max_coord}")
                self.planes = loop_reprs.get_nice_set_of_slice_planes(
                    planespec_num_slices, planespec_min_coord, planespec_max_coord,
                    scan_in_direction_xyz=planespec_scan_direction)
             
            
            # (2022-08-18, also for the partial cache thing) 
            # iterate from the last index that has not been loaded
            # The idx_in_obj_paths is to save the meshes that fail and are skipped
            # (rather than having to save the whole full filenames)
            for idx_in_obj_paths, fname in list(enumerate(self.obj_paths))[next_index_to_load:]:
                mesh_name = os.path.basename(fname)
                thlog.info(f"[{next_index_to_load}] Loading mesh {mesh_name} into dataset")
                try:
                    repr_data, slice_points, slice_normals, repr_sampled_points = \
                        loop_reprs.load_loop_repr_from_mesh(fname, self.planes, 
                        opt.loop_repr_type, throw_on_bad_slice_plane=False, 
                        append_endofsequence_timestep=opt.use_eos_token)
                except Exception as e:
                    if isinstance(e, KeyboardInterrupt):
                        raise e
                    thlog.err(f"Warning: {mesh_name} failed preprocessing; skipping this mesh.")
                    thlog.err(f"Here was the error:\n {repr(e)}")
                    n_skipped_meshes += 1
                    failed_meshes_indices.append(idx_in_obj_paths)
                    continue
                repr_data = np.expand_dims(repr_data, axis=0)
                meshes_reprs_load = repr_data if meshes_reprs_load is None else\
                    vstack_with_padding((meshes_reprs_load, repr_data))
                next_index_to_load += 1
                # ^^ this value is an index in self.obj_paths.
                # new 2022-08-18: partial cache checkpoint: save a partial
                # cache of the meshes_reprs_load array along with this idx
                # to remember where to pick up next time loading is run
                __CACHE_CHECKPOINT_SAVE_INTERVAL = 25

                if (next_index_to_load % __CACHE_CHECKPOINT_SAVE_INTERVAL) == 0:
                    np.savez_compressed(partial_cache_path,
                        planes=self.planes, meshes_reprs=meshes_reprs_load, 
                        loop_repr_type=self.opt.loop_repr_type,
                        next_index_to_load=next_index_to_load,
                        failed_meshes_indices=np.asarray(failed_meshes_indices, dtype=int)
                        )

            # if code gets to this point, the loading loop has completed
            if meshes_reprs_load is None:
                raise ValueError("Dataset has loaded no items.")
            # we have been padding on the fly while appending to meshes_reprs_load,
            # so this should already be a valid np array 
            self.meshes_reprs = meshes_reprs_load
            thlog.info(f"[DATA ] Saving loaded dataset as cache file {cache_path}")
            np.savez_compressed(cache_path, 
                planes=self.planes, meshes_reprs=self.meshes_reprs, 
                loop_repr_type=self.opt.loop_repr_type)

            if n_skipped_meshes > 0: 
                thlog.info(f"[DATA ] note that {n_skipped_meshes} meshes were skipped due to failing loading")
                with open(os.path.join(self.opt.dataroot, 'failed_meshes.txt'),"w") as f:
                    f.writelines(map(lambda idx: self.obj_paths[idx] + "\n", failed_meshes_indices))
                thlog.info(f"[DATA ] saved a list of failed meshes into {os.path.join(self.opt.dataroot, 'failed_meshes.txt')}")

            # delete the partial cache (also new part of 2022-08-18)
            if os.path.isfile(partial_cache_path):
                os.remove(partial_cache_path)
        # shape[0] = number of meshes, shape[1] = number of slices/steps per mesh,
        # shape[2] = number of features in the loop representation
        thlog.info(f"Shape of {self.mode} dataset is (num meshes, num timesteps, features) {self.meshes_reprs.shape}")
        
        # new 2022-09-26: proper support for separate test sets. Different sets
        # may have different max lengths, hence this, in case we need to pad
        # either the test or train set to match the 'longer-seqs' one.
        if pad_to_max_sequence_length is not None and pad_to_max_sequence_length > self.meshes_reprs.shape[1]:
            thlog.info("However, it is specified that the dataset should "
            f"be padded, from the loaded max length of {self.meshes_reprs.shape[1]} up to "
            f"a new max length of {pad_to_max_sequence_length} timesteps, so doing "
            "this padding on the timestep dimension now.")

            # only do padding in this instance, don't re-save it to the cache 
            # npz file, which should always be 'tighest pad' for each dataset
            self.meshes_reprs = pad(self.meshes_reprs, pad_to_max_sequence_length, dim=1, pad_value=0)

        self.n_seqs = self.meshes_reprs.shape[0]
        self.n_steps = self.meshes_reprs.shape[1]
        self.n_input_features = self.meshes_reprs.shape[2] # for ellipse-single

        # calculate mean, std
        self.do_norm = True
        if override_mean_std is not None:
            self.dataset_mean, self.dataset_std = override_mean_std
            if self.dataset_mean is None and self.dataset_std is None:
                self.do_norm = False
        else: 
            if self.opt.data_norm_by == 'per_value':
                # compute per-feature mean and std for all "steps/slices" across
                # all batches.
                
                all_steps_all_batches = np.reshape(self.meshes_reprs, 
                    (self.n_seqs * self.n_steps, self.n_input_features))
                timesteps_that_are_nonzero = np.any(all_steps_all_batches != 0, axis=-1)
                self.dataset_mean = np.mean(all_steps_all_batches[timesteps_that_are_nonzero], axis=0)
                self.dataset_std = np.std(all_steps_all_batches[timesteps_that_are_nonzero], axis=0)
                
                if loop_reprs.loop_repr_uses_binary_levelup_flags(self.opt.loop_repr_type):
                    # NOTE important; we must remember to exclude the binary
                    # flag from this normalization, otherwise BCE loss won't
                    # work (and the binary flag 0. and 1. values will be messed
                    # up). The easy way to "turn off" normalization for a
                    # feature is to manually set that feature's mean to 0 and
                    # stddev to 1...
                    self.dataset_mean[-1] = 0
                    self.dataset_std[-1] = 1
                # mean and std have shape (n_features)
                
            elif self.opt.data_norm_by == 'whole_array': # probably not much use for this option...
                self.dataset_mean = np.mean(self.meshes_reprs) # fold all values into scalar mean and std
                self.dataset_std = np.std(self.meshes_reprss) 
            else:
                thlog.debug("[DATA ] Based on --data_norm_by argument, no normalization will be done.")
                self.do_norm = False
                self.dataset_mean = None
                self.dataset_std = None

        if self.do_norm and (
            (np.count_nonzero(self.dataset_std)) != np.size(self.dataset_std)):
            thlog.debug("[DATA ] Some features of the sequence have 0 standard deviation! No normalization will be done.")
            self.do_norm = False
        
        # calculate number of batches
        self.n_batches = int(np.ceil(self.n_seqs / self.batch_size))

        # a mapping between [0..n_seqs] and a shuffled [0..n_seqs], for 
        # per-epoch within-batch shuffling
        self.seqs_index_map = np.arange(self.n_seqs)


    def get_batch_nonorm(self, batch_i):
        """ Return a batch given  a batch index.
            Return shape: np array (timesteps, batch size, features)
        """
        if batch_i >= self.n_batches:
            raise IndexError(f"no batch index {batch_i}; there are {self.n_batches} batches")
        # this is of shape (Batch, Timesteps, Features)
        start_index = self.batch_size * batch_i
        indices_to_grab = self.seqs_index_map[
            np.arange(start_index, min(start_index + self.batch_size, self.n_seqs))]
        full_seqs = self.meshes_reprs[indices_to_grab]
        # turn into (Timesteps, Batch, Features)
        full_seqs = np.transpose(full_seqs, (1, 0, 2))
        # prepend with a dummy step, so that LSTM can predict the whole thing
        seqs_train = np.vstack((np.expand_dims(np.zeros_like(full_seqs[0]), axis=0), full_seqs))
        seqs_target = full_seqs
        return { 'inp': seqs_train, 'trg': seqs_target }

    def shuffle_batches(self):
        """ run this after each epoch / each complete runthrough of dataset """
        np.random.shuffle(self.seqs_index_map)

    def __len__(self):
        return self.n_batches
        
    def __getitem__(self, seq_i):
        data_item = self.get_batch_nonorm(seq_i)
        __inp = data_item['inp']
        inp_timesteps_that_are_zero = np.all(__inp == 0, axis=-1)
        inp_timesteps_that_are_zero[0] = False  # start-of-sequence timestep should never be padding
        __trg = data_item['trg']
        trg_timesteps_that_are_zero = np.all(__trg == 0, axis=-1)
        data_item['inp_bool_mask'] = inp_timesteps_that_are_zero
        # save a bool mask of shape (timesteps, batch) where True is an all-zero timestep        
        if self.do_norm:
            data_item['inp'] = (__inp - self.dataset_mean) / self.dataset_std
            data_item['inp'][inp_timesteps_that_are_zero] = 0  # don't normalize the padding, keep the padding zero
            
            data_item['trg'] = (__trg - self.dataset_mean) / self.dataset_std
            data_item['trg'][trg_timesteps_that_are_zero] = 0
        return data_item
    
    def find_closest_data_item(self, seq: np.ndarray) -> int:
        """ returns the index of the data item that is the closest by L2 distance
        to the given data item. seq is of shape (timesteps, features) """
        distances = np.mean((self.meshes_reprs - seq) ** 2, axis=(1,2)) # shape (len(meshes_reprs))
        return np.min(distances), np.argmin(distances)

        

class GenericFCNet(nn.Module):
    def __init__(self, 
        flattened_in_size: int, 
        hidden_layer_sizes: list, 
        out_size: int, 
        nonlinearity_between_hidden_layers: nn.Module = nn.ReLU(),
        nonlinearity_at_the_end: nn.Module = nn.Tanh()):
        """
        A generic fully-connected module with configurable n of multiple hidden
        layers. Automatically flattens the input (but respecting batching, which
        is taken to be the first dimension).

        Parameters:
        - flattened_in_size: int; size of the inputs after flattening (i.e. all
            dims except batchdim multiplied together )
        - hidden_fcs: a list of ints, denoting the sizes of the hidden layers
        - out_size: int; the number of output features
        - nonlinearity_between_hidden_layers: the nn.Module to apply to the 
            output of each linear layer in the FC network
        - nonlinearity_at_the_end: the nn.Module to apply to the final outputs.
        
        The two nonlinearity_* arguments HAVE to be nn.Modules and not just
        lambdas/functions in general because they need to be sequenced in a 
        nn.Sequential.
        """
        super(GenericFCNet, self).__init__()
        fc_in_size = flattened_in_size
        moduleseq = [('flattener', nn.Flatten())] # will flatten all dims after batch
        for i, fc_out_size in enumerate(hidden_layer_sizes):
            moduleseq.append(('fc{}'.format(i), nn.Linear(fc_in_size, fc_out_size)))
            moduleseq.append(('nonlin{}'.format(i), nonlinearity_between_hidden_layers))
            #moduleseq.append(('dropout{}'.format(i), nn.Dropout(0.2)))
            fc_in_size = fc_out_size # this layer's output size is next layer's input size
        
        moduleseq.append(('out', nn.Linear(fc_in_size, out_size)))

        if nonlinearity_at_the_end is not None:
            moduleseq.append(('nonlinEnd', nonlinearity_at_the_end)) 
        
        self.fcnet = nn.Sequential(OrderedDict(moduleseq))
    
    def forward(self, x):
        return self.fcnet(x)



class LoopSeqEncoderNet(nn.Module):
    # inspired by sketchrnn encoder
    def __init__(self, in_size: int, enc_hidden_size: int, latent_size: int,
            enc_fc_hidden_sizes: list, enc_bidirectional: bool = False):
        super().__init__()
        hidden_sz_multiplier = 2 if enc_bidirectional else 1

        self.lstm = nn.LSTM(in_size, enc_hidden_size, 1, 
            bidirectional=enc_bidirectional)
        self.latent_size = latent_size
        
        self.fc_lstm_hidden_to_mu = GenericFCNet(
            enc_hidden_size * hidden_sz_multiplier, enc_fc_hidden_sizes, latent_size, nonlinearity_at_the_end=None)
        
        self.fc_lstm_hidden_to_sigma = GenericFCNet(
            enc_hidden_size * hidden_sz_multiplier, enc_fc_hidden_sizes, latent_size, nonlinearity_at_the_end=None)
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x, None) # we never need to seed the encoder with some previous state, since we always feed whole ground truth seqs into the encoder
        # we don't actually need to use lstm_out for anything; all this needs as an encoder is
        # a sequence-aware fold over sequence tokens, which is what the result of h_n is 

        # torch LSTM retuurns h_n with shape (n_directions * n_layers, batch, hidden_size).
        # We want shape (batch, n_directions * n_layers * hidden_size), by concatenating all the different layers' h_n together (if there are >1 lstm layers)
        h_n = h_n.transpose(0,1).reshape(-1, h_n.shape[0] * h_n.shape[-1])
        
        pred_mu = self.fc_lstm_hidden_to_mu(h_n)
        pred_log_of_sigma_squared = self.fc_lstm_hidden_to_sigma(h_n)
        
        # what we predict is actually log of (sigma^2), hence sigma = e^(pred/2)
        sampling_sigma = torch.exp(pred_log_of_sigma_squared / 2)  
        # for sampling it should be 'real sigma'

        # sample a latent from these parameters
        sampled_from_N = torch.normal(torch.zeros(self.latent_size), torch.ones(self.latent_size)).to(x.device)
        sampled_z = pred_mu + sampling_sigma * sampled_from_N
        return sampled_z, pred_mu, pred_log_of_sigma_squared


class PlanePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_sequence_length: int, n=10000):
        super().__init__()
        # precompute the pos embed matrix
        positions = torch.arange(max_sequence_length + 1).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(n) / d_model))
        positional_embedding_additives = torch.zeros(max_sequence_length + 1, d_model)
        positional_embedding_additives[:, 0::2] = torch.sin(positions * div_term)
        positional_embedding_additives[:, 1::2] = torch.cos(positions * div_term)
        # shape (max_sequence_length + 1, d_model)
        self.register_buffer('positional_embedding_additives', positional_embedding_additives)
    
    def forward(self, x_original, x_projected_to_d_model, 
        first_timestep_is_not_a_loop=True, ignore_plane_flags=False):
        """ 
        Parameters:
        - x_original of shape (timesteps, batch, in_size) (i.e. containing
            the original plane-up flags.)
        - x_projected_to_d_model of shape (timesteps, batch, d_model) (i.e.
            x but projected up to d_model for the transformer via a Linear)
        - (optional) first_timestep_is_not_a_loop(=True): set this to True when
            the first time step in x_original is not actually a loop feature
            vector but some dummy start token embedding. This will make sure
            that any number in that first time step is NOT treated as plane-up
            set to 1, and that the positional embedding indices will not have 1 
            subtracted from them (since by convention the first loop always has
            a plane-up flag of 1.)
        - ignore_plane_flags (=False): specify True to ignore plane flags and
            just apply regular positional embedding (one time step = one new PE
            index).
        Returns x_projected_to_d_model with pos. embed. added to it. 
        (this positional embedding is plane-up-flag-aware, meaning
        loops on the same plane are assigned the same position embedding. The
        plane-up flag information is taken from x_original, which is otherwise
        untouched.)
        """
        if not ignore_plane_flags:
            levelup_flags = x_original[:, :, -1].clone() # shape (timesteps, batch)
            if first_timestep_is_not_a_loop:
                levelup_flags[0] = 0

            cumulative_levelups = torch.cumsum(levelup_flags.long(), dim=0) # shape (timesteps, batch)
            # then to index and get the appropriate pos-embed vector, we take ^ and
            # subtract 1 (since the cumsum vector should start at 1 because the
            # first loop by convention has a plane-up flag of 1).
            pos_embed_additive = self.positional_embedding_additives[
                cumulative_levelups - (0 if first_timestep_is_not_a_loop else 1)]
        else:
            pos_embed_additive = self.positional_embedding_additives[
                torch.tile(torch.arange(x_projected_to_d_model.shape[0]), (x_projected_to_d_model.shape[1], 1)).T]
        return x_projected_to_d_model + pos_embed_additive


class LoopSeqTransformerEncoderNet(nn.Module):
    def __init__(self, in_size: int, 
        enc_transformer_d_model: int,
        enc_transformer_n_heads: int, enc_transformer_n_layers: int, 
        enc_transformer_ffwd_size: int,
        latent_size: int, max_sequence_length: int, 
        enc_fc_hidden_sizes: list):

        """
        new 2022-08-20: transformer architecture
        Parameters:
        - enc_transformer_d_model: hidden size of the embeddings used throughout
            the transformer blocks
        - enc_transformer_n_heads: number of heads each transformer enc layer
        - enc_transformer_n_layers: number of transformer enc blocks stacked
        - enc_transformer_ffwd_size: hidden size of the internal feed-forward
            networks inside each transformer block
        - latent_size: size of latent vector produced from the encoder output
        - max_sequence_length: max timestep count of all sequences piped
            into this network (for the FC nets mapping encoder outputs to 
            latent vector sampling parameters)
        - enc_fc_hidden_sizes: hidden sizes of the FC nets mapping the encoder
            output to the latent vector('s sampling distribution parameters)

        """
        super().__init__()

        self.linear_in_to_enc = nn.Linear(in_size, enc_transformer_d_model)
        self.positional_embedding = PlanePositionalEncoding(enc_transformer_d_model, max_sequence_length)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=enc_transformer_d_model, nhead=enc_transformer_n_heads, dim_feedforward=enc_transformer_ffwd_size, dropout=0)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
            num_layers=enc_transformer_n_layers)
        self.fc_enc_out_to_mu = GenericFCNet(
            enc_transformer_d_model, enc_fc_hidden_sizes, latent_size, nonlinearity_at_the_end=None)
        self.fc_enc_out_to_sigma = GenericFCNet(
            enc_transformer_d_model, enc_fc_hidden_sizes, latent_size, nonlinearity_at_the_end=None)
        self.latent_size = latent_size

    
    @staticmethod
    def make_padding_mask(x, keep_start_token=False):
        """
        returns a boolean mask of shape (timesteps, batch) where the timesteps
        whose feature values are all zero (i.e. padding) correspond to a True;
        otherwise False.

        optional arg:
        - keep_start_token: (default=False). after creating the mask, manually
        set the first timestep to never be masked even if it is all zeroes.
        (This is useful because all-zeroes is the 'start of sequence' vector,
        present in the tgt sequence that the decoder sees.)
        """
        # a time step where every feature value is zero is a padding time step
        # (adapting code from reco_alongside_one_bit_flag_loss below)
        
        x_is_zero_at = (x == 0)  # shape (timesteps, batch, features)
        
        # array of bools, one bool for each time step with all-zero features; batched.
        timesteps_that_are_zero = torch.all(x_is_zero_at, dim=-1) # shape (timesteps, batch)
        
        # since nn.Transformer's key_padding_mask requires shape (timesteps[, batch])
        # and not (timesteps, batch, FEATURES), we can leave it at that!
        if keep_start_token:
            # if we want to treat the first timestep as the start token, then
            # set it to not-masked (False) even if it's all-zreoes (which is
            # what we use as our dummy CLS token in the encoder)
            timesteps_that_are_zero[0, :] = False

        # transpose is because for some reason, key_padding_mask arguments 
        # need the shape to be batch-first (batch, seq length), even though
        # the rest of the Transformer API defaults to (seq, batch)??!!!
        return timesteps_that_are_zero.transpose(0,1)
        

    def forward(self, x, key_padding_mask):
        """ x is of shape (timesteps, batch, features (=in_size))
        key_padding_mask is a bool tensor of shape (timesteps, batch) where
        True indicates a timestep that is padding
        """
        # insert a fake "CLS" time step at the beginning so that we can use it
        # to aggregate the encoder state at the output (ala BERT). This can be
        # all-zeroes.
        x = torch.cat((torch.zeros_like(x[0]).unsqueeze(0), x), dim=0)

        # padding_mask needs to have a slot for the fake CLS token at the start
        key_padding_mask = torch.cat((torch.zeros_like(key_padding_mask[0]).unsqueeze(0), key_padding_mask), dim=0)
        # and padding mask needs to be (batch, timesteps) not (timesteps, batch)
        key_padding_mask = key_padding_mask.T        
        
        # project to d_model
        x_projected_to_d_model = self.linear_in_to_enc(x)
        
        # add pos embed based on plane up flags (x will then have d_model features)
        x = self.positional_embedding(x, x_projected_to_d_model, first_timestep_is_not_a_loop=True, ignore_plane_flags=True)

        
        # now run thru encoder
        encoder_out = self.transformer_encoder(x, 
            src_key_padding_mask = key_padding_mask) # shape (timesteps, batch, d_model)
        
        encoder_accumulate = encoder_out[0]  
        # this corresponds to that fake CLS timestep we inserted earlier.
        # it has shape (batch, d_model)

        # now we pipe that accumulate result of the encoder through the FCs to
        # get mu and sigma
        pred_mu = self.fc_enc_out_to_mu(encoder_accumulate)
        pred_log_of_sigma_squared = self.fc_enc_out_to_sigma(encoder_accumulate)
        
        sampling_sigma = torch.exp(pred_log_of_sigma_squared / 2)  # what we predict is actually log of sigma^2... 
        # for sampling it should be 'real sigma'

        # sample a latent from these parameters
        sampled_from_N = torch.normal(torch.zeros(self.latent_size), torch.ones(self.latent_size)).to(x.device)
        sampled_z = pred_mu + sampling_sigma * sampled_from_N
        return sampled_z, pred_mu, pred_log_of_sigma_squared




def kl_loss(pred_mu, pred_log_of_sigma_squared, kl_min: float, kl_weight: float): 
    latent_size = pred_mu.shape[1]  
    # pred_mu and pred_log_of_sigma_squared have shape (batch, latent_size)
    kl = -0.5 * torch.sum(
        1 + pred_log_of_sigma_squared - pred_mu ** 2 - torch.exp(pred_log_of_sigma_squared)) * (1/latent_size) 
    # div by latent size to make the magnitude of KL not depend on latent size
    # (for better comparisons between latent sizes)
    return kl_weight * torch.max(kl, torch.tensor(kl_min).to(pred_mu.device))

def reco_alongside_one_bit_flag_loss(recoloss, binaryloss, 
    pred: torch.Tensor, target: torch.Tensor,
    recoweight: float = 1.0, binaryweight: float = 1.0,
    apply_sigmoid_to_last_feature=True):
    """ 
    pred is of shape (timesteps, batch, n_features), same for target. we take
    the last "feature" to be the binary bit flag, so an L2/other reconstruction
    loss will be taken on the features from 0 to n_features - 2, and the binary
    cross entropy loss will be taken on the feature n_features-1. 
    
    recoloss and binaryloss are available as parameters to specify the nn module
    to use as the loss calculators respectively.

    recoweight and binary weight are also parameters to specify the weights for
    each loss type. (these are optional)

    Note that since our model doesn't understand which flags must be binary,
    it is here that we apply the sigmoid activation to the place where we know
    there will be a binary flag. However, if this behavior is not needed (if
    the model already applied its own sigmoid) then specify
    apply_sigmoid_to_last_feature=False. (it is True by default)
    """
    
    
    target_is_zero_at = (target == 0)  # shape (timesteps, batch, features)
    # array of bools, one bool for each time step with all-zero features; batched.
    timesteps_that_are_zero = torch.all(target_is_zero_at, dim=-1) # shape (timesteps, batch)
    timesteps_that_are_nonzero = torch.logical_not(timesteps_that_are_zero) # (timesteps, batch)
    
    # then mask out the prediction in places that correspond to padding in 
    # the target tensor (i.e. any padding zero in target becomes corresponding zero in pred, too).
    # since each batch may have different padding lengths (in terms of timesteps),
    # this will actually flatten batchwise and return an array of timesteps/strokes/tokens
    # not organized by batch, but we don't care about that here because
    # as long as pred and target are organized in the same way we can take loss.
    pred_significants = pred[timesteps_that_are_nonzero] # (n_tokens_that_are_nonpadding, features)
    target_significants = target[timesteps_that_are_nonzero] # (n_tokens_that_are_nonpadding, features)

    pred_for_reco = pred_significants[:, :-1]     # shape (n_tokens_that_are_nonpadding, n_features-1)
    target_for_reco = target_significants[:, :-1]   # shape (n_tokens_that_are_nonpadding, n_features-1)
    
    pred_for_binary = torch.sigmoid(pred_significants[:, -1]) if apply_sigmoid_to_last_feature \
        else pred_significants[:, -1]  # shape (n_tokens_that_are_nonpadding,)
    target_for_binary = target_significants[:, -1] # shape (n_tokens_that_are_nonpadding,)
    
    reco_loss_val = recoloss(pred_for_reco, target_for_reco)
    binary_loss_val = binaryloss(pred_for_binary, target_for_binary)
    return recoweight * reco_loss_val + binaryweight * binary_loss_val




class LoopSeqDecoderFromLatentNet(nn.Module):
    def __init__(self, in_size: int, dec_hidden_size: int, n_layers: int, 
    latent_size: int, dec_z2lstmhidden_fc_hidden_sizes: list):
        """
        Parameters:
        - in_size: n of features of each time step
        - dec_hidden_size: size of the hidden state of the decoder LSTM
        - n_layers: number of LSTMs stacked on top of one another
        - latent_size: size of the latent vector to be piped into each time step
            and also to initialize the decoder LSTM(s)'s hidden state(s)
        - dec_z2lstmhidden_fc_hiddden_sizes: a list of ints denoting the sizes
            of the hidden layers (each entry is a layer) of the FC network
            transforming the latent vector into the initial hidden state for the
            decoder LSTM.
        """
        super().__init__()
        # doing what sketchrnn did with their latent code z: 
        # - initialize lstm hidden state with a h_0 = tanh(W_z * z + b_z)
        # - pipe the z into every lstm time step by concatenating with each x_t
        
        # OLD:
        # self.latent_to_lstm_hidden = nn.Linear(latent_size, dec_hidden_size)

        # NEW (2022/05/10): 
        # by default GenericFCNet has ReLU in between hidden layers and
        # Tanh at the end so no need the tanh in the call in forward() anymore
        self.latent_to_lstm_hidden = GenericFCNet( 
            latent_size, dec_z2lstmhidden_fc_hidden_sizes, dec_hidden_size)
        self.lstm = nn.LSTM(in_size + latent_size, dec_hidden_size, n_layers)
        self.fc = nn.Linear(dec_hidden_size, in_size) 
        #^uhh this can be whatever I guess; this maps LSTM predictions to actual
        #outputs
        
        # for making the zeros for init cell state
        self.init_c_state_size = (n_layers, dec_hidden_size)

    def forward(self, x, latent_z, init_hc_states):
        # latent_z is batched, i.e. shape (batch_size, latent_size)
        if init_hc_states is None:
            batch_size = x.shape[1]
            n_layers, dec_hidden_size = self.init_c_state_size
            init_h_state_lstm0 = self.latent_to_lstm_hidden(latent_z)
            init_h_state_lstm_others = torch.zeros(batch_size, dec_hidden_size).float().to(x.device)
            init_h_state = torch.stack([init_h_state_lstm0] + [init_h_state_lstm_others.detach().clone() for _ in range(n_layers-1)] )
            init_c_state = torch.zeros(n_layers, batch_size, dec_hidden_size).float().to(x.device)
            init_hc_states = (init_h_state, init_c_state)
        seq_len = x.shape[0]
        # create new view of latent_z that repeats latent_z as many times as the sequence length
        # (doesn't actually copy the tensor, but we won't write to tiled_z so this is fine)
        tiled_z = latent_z.expand((seq_len, -1, -1)) # shape (seq_len, batch_size, latent_size)
        x_with_z = torch.cat((x, tiled_z), dim=-1) # shape (seq_len, batch_size, in_size + latent_size)
        lstm_out, (h_n, c_n) = self.lstm(x_with_z, init_hc_states)
        return self.fc(lstm_out), (h_n, c_n)


class LoopSeqTransformerDecoderNet(nn.Module):
    def __init__(self, transformer_arch_version: int, # 09-16: versioning added
        in_size: int,
        dec_transformer_d_model: int,  
        dec_transformer_n_heads: int, dec_transformer_n_layers: int, 
        dec_transformer_ffwd_size: int, 
        latent_size: int, max_sequence_length: int, 
        dec_z2memory_fc_hidden_sizes: list, 
        dropout: float = 0):

        super().__init__()

        self.transformer_arch_version = transformer_arch_version

        self.linear_in_to_dec = nn.Linear(in_size, dec_transformer_d_model)
        self.positional_embedding = PlanePositionalEncoding(dec_transformer_d_model, max_sequence_length)
        
        # we don't use any cross-attention so our 'decoder'  actually uses the
        # pytorch TransformerEncoder module (which uses pure self attention)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=dec_transformer_d_model, nhead=dec_transformer_n_heads, 
            dim_feedforward=dec_transformer_ffwd_size, dropout=dropout)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, 
            num_layers=dec_transformer_n_layers)

        self.linear_dec_to_out = nn.Linear(dec_transformer_d_model, in_size)

        self.in_size = in_size
        self.max_sequence_length = max_sequence_length

        if self.transformer_arch_version == 0:
            # learnable start loop embedding, and injection via an extra CLS timestep.
            # this is part of v0 transformers
            self.fc_latent_to_dec_cls = GenericFCNet( 
            latent_size, dec_z2memory_fc_hidden_sizes, dec_transformer_d_model)
            self.sequence_start_embedding = nn.Parameter(torch.rand(in_size), requires_grad=True)

        elif self.transformer_arch_version in (1, 2):
            # start embedding learned from latents
            self.fc_latent_to_start_embedding = GenericFCNet(
                latent_size, dec_z2memory_fc_hidden_sizes, in_size)
            
            # latent to the array of all binary flags to prevent latent-forgetting of topology
            if self.transformer_arch_version == 2:
                self.fc_latent_to_binary_flags = GenericFCNet(
                    latent_size, dec_z2memory_fc_hidden_sizes, max_sequence_length)
            
        
    
    def inject_latent_z_into_transformer_input(self, latent_z, x_projected_to_d_model):
        """ x_projected_to_d_model has shape (timesteps, batch, d_model). 
        maps latent_z into a CLS token/aggregate embedding and injects it into
        the start of x_projected_to_d_model; returns the result of shape 
        (timesteps + 1, batch, d_model)

        NOTE (2022-09-16) this is only applicable for version 0 of the 
        transformer architecture.
        """
        # latent_z is of shape (batch, latent_size)
        
        fake_CLS_embedding = self.fc_latent_to_dec_cls(latent_z).unsqueeze(0) # (1, batch, d_model)
        x_with_cls = torch.cat((fake_CLS_embedding, x_projected_to_d_model), dim=0)
        return x_with_cls
    
    def forward(self, x, latent_z, key_padding_mask):
        if self.transformer_arch_version == 0:
            return self.forward_v0(x, latent_z, key_padding_mask)
        elif self.transformer_arch_version == 1:
            return self.forward_v1(x, latent_z, key_padding_mask)
        elif self.transformer_arch_version == 2:
            return self.forward_v2(x, latent_z, key_padding_mask)
        else:
            raise NotImplementedError("unknown transformer arch version")

    def forward_v2(self, x, latent_z, key_padding_mask):
        """ version2 differs from version 1 in that there is now an additional
        'latent to binary flag array' FC that then overwrites the binary flag
        portion of the forward_v1 output. This is supposed to prevent forgetting
        of the planeup configuration by leaving it out of the  autoregressive
        training part and have it be directly given from latent. """
        out, _ = self.forward_v1(x, latent_z, key_padding_mask)
        binary_flags_sequence = self.fc_latent_to_binary_flags(latent_z)
        # ^^ this has shape (batch, self.max_sequence_length)
        
        # modify the transformer's binary flag output with this. 
        # (though it needs to first be (max_sequence_length, batch))
        out[:, :, -1] += binary_flags_sequence.T
        return out, (None, None)


    def forward_v1(self, x, latent_z, key_padding_mask):
        """ version1 differs from the below forward_v0 in that there is no 
        start-embedding, and instead that start embedding is predicted directly
        from the latent (using the latent-to-cls layer) """
        seq_len = x.shape[0]
        x = x.clone()
        
        x[0, :, :] = self.fc_latent_to_start_embedding(latent_z)
        
        x_projected_to_d_model = self.linear_in_to_dec(x)
        x = self.positional_embedding(x, x_projected_to_d_model, first_timestep_is_not_a_loop=True, ignore_plane_flags=True)
        key_padding_mask = key_padding_mask.T
        decoder_out = self.transformer_decoder(x, 
            mask=nn.Transformer.generate_square_subsequent_mask(
                self.max_sequence_length).to(x.device),
            src_key_padding_mask = key_padding_mask 
            )
        
        # the None is to match the return tuple format of 
        # LoopSeqDecoderFromLatentNet
        out = self.linear_dec_to_out(decoder_out)
        return out, (None, None)
    

    def forward_v0(self, x, latent_z, key_padding_mask):
        """ should have an identical call interface to that of 
            LoopSeqDecoderFromLatentNet.forward. The third argument is the
            key_padding_mask (not a 'prev hidden states' like with the LSTM models).
            This is a bool tensor of shape (timesteps, batch) where True values
            indicate timesteps that are padding.

            x is of shape (timesteps, batch, in_size)
            this timesteps == self.max_sequence_length (and is the data sequence
            shifted right so that the first 'token'/loop is a start-of-sequence
            marker. For this module it will be replaced with the learned
            start-of-sequence embedding.)

            returns tensor (shape (timesteps, batch, in_size)), (None, None)
        """
        
        seq_len = x.shape[0]
        
        # put into x the learned start embedding. 
        x = x.clone()
        x[0, :, :] = self.sequence_start_embedding

        # make padding mask (before pos enc is added, so we can still know 
        # which time step is padding (all-zeroes)). keep_start_token is still
        # True in case our learned start embedding happens to be all zeroes.

        # we need shape (batch, timesteps) because torch transformers expect
        # this shape for the key padding mask (for some reason...)
        key_padding_mask = key_padding_mask.T
        # since we concat the CLS token in inject_latent_z_into_transformer_input,
        # we must also add that extra time step in the pad mask
        key_padding_mask = torch.cat((torch.zeros_like(key_padding_mask[:, 0]).unsqueeze(1), key_padding_mask), dim=1)

        x_projected_to_d_model = self.linear_in_to_dec(x)
        x_projected_to_d_model = self.inject_latent_z_into_transformer_input(latent_z, x_projected_to_d_model)
        x = self.positional_embedding(x, x_projected_to_d_model, first_timestep_is_not_a_loop=True, ignore_plane_flags=True)
               
        decoder_out = self.transformer_decoder(x, 
            mask=nn.Transformer.generate_square_subsequent_mask(
                self.max_sequence_length + 1).to(x.device),
            src_key_padding_mask = key_padding_mask 
            )
        
        # the None is to match the return tuple format of 
        # LoopSeqDecoderFromLatentNet
        return self.linear_dec_to_out(decoder_out)[1:], (None, None)
    
    @torch.no_grad()
    def inference(self, latent_z, n_time_steps: int, fn_after_each_timestep=None):
        """ perform autoregressive inference for n_time_steps starting from
        the learned start vector, conditioned by latent_z.
        (n_time_steps should also count the start vector!)

        optional arg: fn_after_each_timestep (a callable):
            a function to apply to each newly generated timestep before it is
            fed back in to generate the next timestep.

        output: shape (n_time_steps, in_size)
        """
        # tile the start_vector n_time_steps across; unsqueeze for batch size=1
        # this has shape (n_time_steps, 1, in_size)
        assert n_time_steps-1 <= self.max_sequence_length, \
            f"n_time_steps ({n_time_steps}) for inference must be <= than"\
            f" the dataset max_sequence_length ({self.max_sequence_length}) plus 1"
        output = torch.tile(torch.zeros(self.in_size, device=latent_z.device), (n_time_steps-1, 1)).unsqueeze(1)
        # no time step is padding in this inference context, so all False
        key_padding_mask = torch.zeros(self.max_sequence_length, 1, 
            dtype=bool, device=latent_z.device)
        latent_z = latent_z.unsqueeze(0)
        for timestep_i in range(1, n_time_steps):
            decoder_output, _ = self.forward(output, latent_z, key_padding_mask)
            if timestep_i == n_time_steps - 1:
                # last iter: since the feed-in into the network must be
                # max_sequence_length only, 'output' cannot contain both the
                # start token AND the last generated one, so we must manually
                # add the last generated token here. This is fine since after
                # this iteration `output` is no longer run through the model
                output = torch.cat((output, torch.zeros_like(output[0]).unsqueeze(1)), dim=0)
            
            # decoder_output's index 0 is the first meaningful timestep;
            # output's index 1 is the first meaningful timestep;
            output[timestep_i] = decoder_output[timestep_i - 1]
            if callable(fn_after_each_timestep):
                fn_after_each_timestep__output = fn_after_each_timestep(output[timestep_i])
                if isinstance(fn_after_each_timestep__output, torch.Tensor):
                    output[timestep_i] = fn_after_each_timestep__output
                elif isinstance(fn_after_each_timestep__output, tuple):
                    modified_timestep, latent_modification_function = fn_after_each_timestep__output
                    # just like the case where the intervention function's
                    # output is just a torch.tensor...
                    output[timestep_i] = modified_timestep  
                    # but with the addition of modifying the latent vector
                    # for all subsequent time steps!!
                    if callable(latent_modification_function):
                        if self.transformer_arch_version == 0:
                            thlog.error("transformer architecture v0 does not "
                            "support latent-conditioned start embedding and thus"
                            " does not support dynamic latent swapping during autoregressive generation.")
                        else:
                            # we need the fc_latent_to_start_embedding,
                            # so transformer arch v0 is out...
                            thlog.debug("intervention function is updating the latent z for the next timesteps!")
                            latent_z = latent_modification_function(latent_z)
                            output[0, :, :] = self.fc_latent_to_start_embedding(latent_z)

        return output.squeeze(1)

########## wrapper Model definitions and related helpers ##############

def init_net_for_gpu(opt, net):
    if opt.gpu_ids:
        assert(torch.cuda.is_available())
        net.cuda(opt.gpu_ids[0])
        net = net.cuda()
        net = torch.nn.DataParallel(net, opt.gpu_ids)
        return net
    return net


class LoopSeqEncoderDecoderModel():
    """ Skeleton for final model consisting of an encoder (or set of encoders) and a decoder """
    # we won't be using the MyGenericModel superclass here because we have two nn modules
    def __init__(self, opt, num_features:int, max_sequence_length: int, dry_run=False):

        self.NUM_INPUT_FEATURES = num_features
        self.LATENT_SIZE = opt.latent_size
        self.MAX_SEQUENCE_LENGTH = max_sequence_length

        self.opt = opt
        self.dry_run = dry_run
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if \
            self.gpu_ids else torch.device('cpu')
        self.save_dir = opt.save_dir
        self.is_train = opt.is_train

        if opt.architecture == "lstm":
            self.encoder = init_net_for_gpu(opt, 
                LoopSeqEncoderNet(
                    self.NUM_INPUT_FEATURES
                    , opt.enc_lstm_hidden_size 
                    , self.LATENT_SIZE
                    , opt.enc_fc_hidden_sizes
                    , opt.enc_bidirectional))

            self.decoder = init_net_for_gpu(opt, 
                LoopSeqDecoderFromLatentNet(
                    self.NUM_INPUT_FEATURES
                    , opt.lstm_hidden_size
                    , opt.lstm_n_layers
                    , self.LATENT_SIZE
                    , opt.dec_fc_hidden_sizes))

        elif opt.architecture == "transformer" or opt.architecture == "lstm+transformer":
            assert loop_reprs.loop_repr_uses_binary_levelup_flags(opt.loop_repr_type), \
                "transformer architectures only implemented for loop representations"\
                "that use binary plane-up flags"

            self.encoder = init_net_for_gpu(opt,
                LoopSeqTransformerEncoderNet(
                    self.NUM_INPUT_FEATURES
                    , opt.enc_transformer_d_model
                    , opt.enc_transformer_n_heads
                    , opt.enc_transformer_n_layers
                    , opt.enc_transformer_ffwd_size
                    , self.LATENT_SIZE
                    , self.MAX_SEQUENCE_LENGTH
                    , opt.enc_fc_hidden_sizes))
            if opt.architecture == "lstm+transformer":
                # hybrid architecture, uses an LSTM decoder instead of a
                # transformer decoder!!!!
                self.decoder = init_net_for_gpu(opt, 
                    LoopSeqDecoderFromLatentNet(
                        self.NUM_INPUT_FEATURES
                        , opt.lstm_hidden_size
                        , opt.lstm_n_layers
                        , self.LATENT_SIZE
                        , opt.dec_fc_hidden_sizes))
            elif opt.architecture == "transformer":
                # pure transformer decoder...
                self.decoder = init_net_for_gpu(opt, 
                    LoopSeqTransformerDecoderNet(
                          opt.transformer_arch_version
                        , self.NUM_INPUT_FEATURES
                        , opt.dec_transformer_d_model
                        , opt.dec_transformer_n_heads
                        , opt.dec_transformer_n_layers
                        , opt.dec_transformer_ffwd_size
                        , self.LATENT_SIZE
                        , self.MAX_SEQUENCE_LENGTH
                        , opt.dec_fc_hidden_sizes
                        , opt.dec_transformer_dropout))
        else:
            raise ValueError("invalid model architecture type")
        # set nn.Module training/eval mode (for dropout layers etc)
        self.encoder.train(self.is_train)
        self.decoder.train(self.is_train)

        if opt.reco_loss_type == 'l2':
            thlog.info("using L2 loss")
            recoloss = nn.MSELoss()
        elif opt.reco_loss_type == 'l1':
            thlog.info("using L1 loss")
            recoloss = nn.L1Loss()
        else:
            raise NotImplementedError(f"unknown loss type {opt.reco_loss_type}")

        loop_repr_type_being_used = opt.loop_repr_type
        if loop_reprs.loop_repr_uses_binary_levelup_flags(loop_repr_type_being_used):
            binaryloss = nn.BCELoss()

            self.decoder_recoloss = lambda p, t:\
                reco_alongside_one_bit_flag_loss(recoloss, binaryloss, p, t,
                recoweight=1.0, binaryweight=1.0)

        elif loop_repr_type_being_used == loop_reprs.LOOP_REPR_ELLIPSE_SINGLE:
            self.decoder_recoloss = recoloss
        

        self.kl_annealing_curr_iter = 0  # for cyclic annealing schedule. an iter is a single run of forward_and_backward (on a single data item/batch)
        self.kl_annealing_cycle = opt.enc_kl_anneal_cycle
        _kl_annealing_formula_cyclic = (lambda _: 1.0) if opt.enc_kl_weight == 0.0 else \
            (lambda curr_iter: (curr_iter % self.kl_annealing_cycle) / self.kl_annealing_cycle) # cyclic annealing rule, with linear ramping
        _kl_annealing_R = 0.9999
        _kl_annealing_formula_sketchrnn = (lambda _: 1.0) if ((opt.enc_kl_weight == 0.0) or (self.kl_annealing_cycle < 0)) else \
            (lambda curr_iter: 1.0-(1.0-opt.enc_kl_min)*(_kl_annealing_R ** curr_iter))
        self.kl_annealing_formula = _kl_annealing_formula_sketchrnn

        # initialize optimizer first, if training:
        if self.is_train:
            DEFAULT_BETA1 = 0.9
            self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(),
                lr=opt.lr, betas=(DEFAULT_BETA1, 0.999))
            self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(),
                lr=opt.lr, betas=(DEFAULT_BETA1, 0.999))
        
        # then load network states and optimizer states, if load_epoch is specified
        if loop_opts.has_field(self.opt, 'load_epoch'):
            self.load_networks(self.opt.load_epoch, self.opt.load_epoch) # TODO add ability to load different enc_epoch, dec_epoch

        # then create lr scheduler and init the last_epoch to the desired opt.count_from_epoch. 
        # we need to have loaded the optimizer first because if last_epoch is not -1 it will check
        # if the optimizer is freshly initialized or not (and if fresh it will raise an error).
        # NOTE that the scheduler's epoch state is the only place in this model class where
        # epoch number is kept as state. All the other methods that use epoch numbers will
        # have the epoch num as an argument.
        if self.is_train:
            def lambda_rule(epoch):
                start_epoch_num = 0
                lr_l = 1.0 - max(0, epoch + 1 + start_epoch_num - opt.niter) / float(opt.niter_decay + 1)
                return lr_l
            scheduler_last_epoch = self.opt.count_from_epoch if loop_opts.has_field(self.opt, 'count_from_epoch') else (-1)
            self.encoder_scheduler = torch.optim.lr_scheduler.LambdaLR(self.encoder_optimizer, lr_lambda=lambda_rule, last_epoch=scheduler_last_epoch)
            self.decoder_scheduler = torch.optim.lr_scheduler.LambdaLR(self.decoder_optimizer, lr_lambda=lambda_rule, last_epoch=scheduler_last_epoch)
            #self.encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.encoder_optimizer, factor=0.8, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
            #self.decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer, factor=0.8, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
        
        # info print
        n_enc_params = 0
        n_dec_params = 0
        for param in self.encoder.parameters():
            n_enc_params += param.numel()
        for param in self.decoder.parameters():
            n_dec_params += param.numel()
        thlog.info(f"Encoder has {n_enc_params} parameters; Decoder has {n_dec_params} parameters")
    
    def print_detailed_network_stats(self):
        out_encoder = fvcore.nn.parameter_count_table(self.encoder)
        out_decoder = fvcore.nn.parameter_count_table(self.decoder)
        thlog.info(f"ENCODER:\n" + out_encoder + "\nDECODER:\n"+out_decoder)

    def save_networks(self, which_epoch):
        """save encoder and decoder nets separately to disk"""
        if self.dry_run:
            thlog.info("dry run, not saving")
            return
        
        for (curr_module, fname_format, is_nn_module) in \
            ((self.encoder, 'enc{}_net.pth', True), (self.decoder, 'dec{}_net.pth', True), 
             (self.encoder_optimizer, 'enc{}_optimizer.pth', False), (self.decoder_optimizer, 'dec{}_optimizer.pth', False)):
            save_filename = fname_format.format(which_epoch)
            save_path = os.path.join(self.save_dir, save_filename)
            if self.gpu_ids and torch.cuda.is_available() and is_nn_module:
                torch.save(curr_module.module.cpu().state_dict(), save_path)
                curr_module.cuda(self.gpu_ids[0])
            else:
                if is_nn_module:
                    torch.save(curr_module.cpu().state_dict(), save_path)
                else:
                    # optimizers don't have a .cpu() method 
                    torch.save(curr_module.state_dict(), save_path)
    
    def save_network(self, which_epoch):
        # alias for save_networks
        self.save_networks(which_epoch)

    def load_networks(self, enc_epoch, dec_epoch):
        """ load encoder and decoder nets from disk; can specify different
        epochs to load for each.
        If optimizer states are also found in the directory, then also load those.
        """
        for fnameprefix in ('enc', 'dec'):
            which_epoch = enc_epoch if fnameprefix == 'enc' else dec_epoch
            which_epoch = which_epoch if which_epoch != -1 else 'latest'
            
            # first load the enc/dec nn.module state_dict
            save_filename = f'{fnameprefix}{which_epoch}_net.pth'
            load_path = os.path.join(self.save_dir, save_filename)
            net = self.encoder if fnameprefix == 'enc' else self.decoder
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            thlog.info(f'loading the {fnameprefix} network from {load_path}')
            state_dict = torch.load(load_path, map_location=str(self.device))
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            net.load_state_dict(state_dict)
            # if we're not in self.is_train, optimizers will not even be
            # defined in __init__ so we can return out here...
            if not self.is_train:
                continue

            # otherwise, check if optimizer saved state for this enc/dec exists
            # and load that too if it does
            optimizer_fname = f"{fnameprefix}{which_epoch}_optimizer.pth"
            optimizer_load_path = os.path.join(self.save_dir, optimizer_fname)
            if os.path.isfile(optimizer_load_path):
                optimizer = self.encoder_optimizer if fnameprefix == 'enc' else self.decoder_optimizer
                optimizer_state_dict = torch.load(optimizer_load_path, map_location=str(self.device))
                optimizer.load_state_dict(optimizer_state_dict)
                thlog.info(f"{fnameprefix} optimizer state also exists: loaded with LR {optimizer.param_groups[0]['lr']}")

    
    def write_loss_log(self, epoch_num:int, loss_vals:tuple, is_test:bool):
        """ Save multiple sublosses and total loss"""
        if self.dry_run:
            return
        if is_test:
            save_filename = 'test_log.txt'
        else:
            save_filename = 'loss_log.txt'
        save_path = os.path.join(self.save_dir, save_filename)
        with open(save_path, 'a') as f:  
            # csv format: epoch,reco loss,klloss(,and more if there are any)
            f.write(f'{epoch_num},{",".join(map(str,loss_vals))}\n')

    def update_learning_rate(self, losses:tuple=None):
        """update learning rate (called once every epoch)"""
        if isinstance(self.encoder_scheduler, 
            torch.optim.lr_scheduler.ReduceLROnPlateau):
            # loss tuple is (decoder reco loss, encoder KL loss)
            self.encoder_scheduler.step(losses[1])
            self.decoder_scheduler.step(losses[0])
        else:    
            self.encoder_scheduler.step()
            self.decoder_scheduler.step()
        enclr = self.encoder_optimizer.param_groups[0]['lr']
        declr = self.decoder_optimizer.param_groups[0]['lr']
        thlog.info('encoder LR set to {:5.7f}, decoder LR set to {:5.7f}'\
                .format(enclr, declr))

    def to_gpu_tensor(self, np_arr):
        return torch.from_numpy(np_arr).float().to(self.device)

    def save_options(self):
        if self.dry_run:
            return
        opt_fname = os.path.join(self.save_dir, 'opt.txt')
        with open(opt_fname, 'wt') as opt_file:
            opt_file.write('MODEL HYPERPARAMS & TRAINING OPTIONS\n')
            for (k, v) in self.opt.__dict__.items():
                opt_file.write(f"{k}: {v}\n")
            


    def run_decoder_one_step(self, data_item: dict, prev_states=None):
        """ (only valid if the architecture is `lstm`)
            For sampling a sequence from the trained decoder by feeding in a
            custom z. Should have an identical interface to the
            forward_and_backward method of the MeshSeqManualLatentDecoderModel
            class. Assumes that the input data (input sequence, latent z) are
            not batched, ie shape (timesteps, features) """
        if self.opt.architecture not in ("lstm", "lstm+transformer"):
            raise ValueError("attempting to call an lstm decoder one-step " 
            "function when the architecture is currently not set to `lstm`.")
        input_seq = self.to_gpu_tensor(data_item['inp']).unsqueeze(1)
        latent_z = self.to_gpu_tensor(data_item['z']).float().unsqueeze(0)
        pred_seq, state_tuple = self.decoder(input_seq, latent_z, prev_states)
        return pred_seq, state_tuple, (None, None)
    
    def run_transformer_inference(self, latent_z, n_time_steps, fn_after_each_timestep=None):
        """ (only valid if the architecture is `transformer`; this is kind of 
        used in a similar context to run_decoder_one_step except this is for
        when the architecture is transformer.)
        
        fn_after_each_timestep (default=None) if not None should be a callable
        that takes 1 argument: the timestep data of shape (1, features), and
        returns a (modified) timestep of the same shape.
        """
        if self.opt.architecture != "transformer":
            raise ValueError("attempting to call a transformer inference function"
            "but the architecture is not set to `transformer`.")
        
        # each ts is of shape (batch, features); we want to sigmoid the last
        # feature, because our model doesn't do that, the loss does...
        # we need this to keep the numbers looking like how they did when the
        # model trained on it
        def __sigmoid_last_feature(ts):
            ts[:, -1] = torch.sigmoid(ts[:, -1])
            return ts
        sigmoiding_after_each_timestep = __sigmoid_last_feature if\
            loop_reprs.loop_repr_uses_binary_levelup_flags(self.opt.loop_repr_type) else (lambda u: u)
        
        __maybe_compose = lambda f, g: \
            ((lambda x: f(g(x))) if callable(f) else (lambda x: g(x)))

        if isinstance(self.decoder, torch.nn.DataParallel):
            __accessor = self.decoder.module
        else:
            __accessor = self.decoder
        return __accessor.inference(
            self.to_gpu_tensor(latent_z), 
            n_time_steps, 
            fn_after_each_timestep=__maybe_compose(
                fn_after_each_timestep, sigmoiding_after_each_timestep))
            
        

    def forward_and_backward(self, data_item: dict, is_train=True, prev_decoder_states=None, return_latent=False, no_random_sampling=False):
        input_seq = self.to_gpu_tensor(data_item['inp'])
        input_mask_timesteps_that_are_zero = torch.from_numpy(data_item['inp_bool_mask']).to(self.device) # shape (timesteps, batch)
        decoder_target = data_item.get('trg')
        decoder_target = self.to_gpu_tensor(decoder_target) if \
            decoder_target is not None else None
        # first pipe input seq througuh the encoder to get a sampled z and the
        # predicted mu, sig
        sampled_z, pred_mu, pred_log_of_sigma_squared = self.encoder(input_seq[1:], input_mask_timesteps_that_are_zero[1:])
        # ^^ that is [1:] because we can throw away the dummy all-zeroes
        # operation at the start, since the encoder doesn't have to give
        # predictions of the next token
        if (self.opt.enc_kl_weight == 0.0) or no_random_sampling:
            # no random sampling if we're not going to train KL; turns this into
            # a normal autoencoder. # uncomment this to print KL stats regardless,
            # since sigma seems to be optimized as mu is trained as well, even if
            # kl isn't explicitly trained.  (2022/06/14)

            # with torch.no_grad():
            #     thlog.info(f"(no KL training, reporting for info) mu={pred_mu}, sigm={pred_log_of_sigma_squared}, klraw={kl_loss(pred_mu, pred_log_of_sigma_squared, 0, 1)}")
            sampled_z = pred_mu 
        # then make a prediction / reconstruction with the decoder
        
        if self.opt.architecture == "lstm" or self.opt.architecture == "lstm+transformer":
            third_decoder_argument = prev_decoder_states
        elif self.opt.architecture == "transformer":
            third_decoder_argument = input_mask_timesteps_that_are_zero[:-1]

        # NOTE input_seq[-1] because now the dataset will provide
        # the entire sequence from the 0-dummy-step to the last timestep, rather
        # than chopping off the last time step. This makes sure that the encoder
        # sees the whole thing all the way through.
        pred_seq, state_tuple = self.decoder(input_seq[:-1], sampled_z, third_decoder_argument)
        
        
        if decoder_target is not None:
            decoder_reco_loss = self.decoder_recoloss(pred_seq, decoder_target)

            encoder_kl_annealing_eta = self.kl_annealing_formula(
                self.kl_annealing_curr_iter) if is_train else 1.0
            encoder_kl_loss = kl_loss(
                pred_mu, pred_log_of_sigma_squared, 
                self.opt.enc_kl_min, self.opt.enc_kl_weight * encoder_kl_annealing_eta)
            loss = decoder_reco_loss + encoder_kl_loss
            # for better numerical stability; because currently the scale of
            # values is way too low and we may run into precision loss
            loss = loss * 100 
        
            if is_train:
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                loss.backward()
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()
                # 'step' the kl annealing schedule as well
                self.kl_annealing_curr_iter += 1
                # if (self.kl_annealing_cycle != 0) and (self.kl_annealing_curr_iter % self.kl_annealing_cycle == 0):
                #     thlog.info("One KL annealing cycle complete, resetting to KL multiplier = 0")
            # NOTE we are reporting the true losses rather than the 10*-scaled loss as seen above.
            if return_latent:
                thlog.trace("Encoder predicted mu and sigma for this input: ")
                thlog.trace(f"mu: {pred_mu}")
                thlog.trace(f"sigma: {pred_log_of_sigma_squared}")
                return pred_seq, state_tuple, (decoder_reco_loss.item(), encoder_kl_loss.item()), sampled_z
            else:
                return pred_seq, state_tuple, (decoder_reco_loss.item(), encoder_kl_loss.item())
        if return_latent:
            return pred_seq, state_tuple, sampled_z
        else:
            return pred_seq, state_tuple, (None, None)
    
    def finalize_after_training(self):
        thlog.debug("dummy finalizing function")


def dataset_cache_gen_main():
    thlog.info("loop_models running as main, will be doing DATASET CACHE GENERATION.")
    opt = loop_opts.LoopSeqOptions().parse_cmdline()
    assert opt.mode in ('train', 'test')
    train_dataset = LoopSeqDataset(opt, override_mode=opt.mode, regenerate_cache=True)
    thlog.info("Dataset cache generation done.")

if __name__ == "__main__":
    dataset_cache_gen_main()
