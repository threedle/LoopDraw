import argparse
import os
import sys
import shlex

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def has_field(opt, argname):
    """ checks whether opt contains the field "argname" and it is not None.
    the Argparse namespace opject leaves a field as None if it's not specified in the command args
    and there is no default value."""
    return (hasattr(opt, argname) and getattr(opt, argname) is not None)

class LoopSeqOptions:
    def __init__(self):
        # 
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--architecture', type=str, default='lstm', choices=['lstm', 'transformer', 'lstm+transformer', 'pointnet+transformer'], help=
            "choose the model architecture to use. default: lstm; options are [lstm, transformer, lstm+transformer]. "
            "If `lstm` is chosen, then the options enc_lstm_hidden_size, enc_bidirectional, lstm_hidden_size, lstm_n_layers are applicable."
            "If `transformer` is chosen, then the options "
            "enc_transformer_n_layers, enc_transformer_n_heads, enc_transformer_ffwd_size, "
            "dec_transformer_n_layers, dec_transformer_n_heads, dec_transformer_ffwd_size "
            "are available. "
            "If `lstm+transformer` is chosen, then the enc_transformer_* options"
            "are available, plus enc_lstm_hidden_size, and lstm_*. "
            "The pointnet+transformer architecture is a pointnet encoder + a transformer decoder; "
            "all dec_transformer* arguments apply, in addition to pointnet_hidden_size"
            )
        
        # fourier map options 
        self.parser.add_argument('--fourier_map_size', type=int, default=None, help="fourier feature map size before projecting to transformer d model")
        self.parser.add_argument('--fourier_map_sigma', type=float, default=4.0, help="fourier feature map sigma value")

        # LSTM options
        self.parser.add_argument('--enc_lstm_hidden_size', type=int, default=64, help='encoder lstm hidden layer size')
        # self.parser.add_argument('--enc_lstm_n_layers', type=int, default=1, help='WARNING not used!!! encoder lstm number of layers stacked on each other')
        self.parser.add_argument('--enc_bidirectional', action='store_true', help='whether the encoder LSTM is bidirectional')
        self.parser.add_argument('--lstm_hidden_size', type=int, default=64, help='decoder lstm hidden layer size')
        self.parser.add_argument('--lstm_n_layers', type=int, default=1, help='number of lstm layers to stack')

        # PointNet options, for architecture "pointnet+transformer"
        self.parser.add_argument('--pointnet_hidden_size', type=int, default=128, help="hidden feature size for pointnet encoder for --architecture pointnet+transformer and others that use a pointnet")
        # Transformer options

        # I didn't have this option in the first version (v0) 
        # so the default is 0 for the runs that don't have this in their command args.
        # Version changelog:
        # - v1 bypasses the extra CLS token and directly predicts the
        # start-embedding from the latent. 
        # additionally, the key_padding_mask is no longer wrong: the first
        # timestep is now not considered padding anymore.
        self.parser.add_argument('--transformer_arch_version', type=int, default=0, help="versioning for the transformer architecture. ")

        self.parser.add_argument('--enc_transformer_d_model', type=int, default=128, help="embedding dim size used internally in the transformers")
        self.parser.add_argument('--enc_transformer_n_layers', type=int, default=4, help="number of layers of transformer decoder blocks")
        self.parser.add_argument('--enc_transformer_n_heads', type=int, default=1, help="number of attention heads per decoder block")
        self.parser.add_argument('--enc_transformer_ffwd_size', type=int, default=256, help="hidden size of the internal feed-forward network in each transformer encoder block")

        self.parser.add_argument('--dec_transformer_d_model', type=int, default=128, help="embedding dim size used internally in the transformers")
        self.parser.add_argument('--dec_transformer_n_layers', type=int, default=4, help="number of layers of transformer decoder blocks")
        self.parser.add_argument('--dec_transformer_n_heads', type=int, default=1, help="number of attention heads per decoder block")
        self.parser.add_argument('--dec_transformer_ffwd_size', type=int, default=256, help="hidden size of the internal feed-forward network in each transformer decoder block")
        self.parser.add_argument('--dec_transformer_dropout', type=float, default=0, help="dropout rate on the decoder side")

        # VAE training options
        self.parser.add_argument('--enc_kl_min', type=float, default=0.05, help='minimum KL loss, to prevent overemphasizing optimizing the KL div during training')
        self.parser.add_argument('--enc_kl_weight', type=float, default=0.5, help='weight applied to KL divergence loss; also to control how important it is to enforce the latent distribution')
        self.parser.add_argument('--enc_kl_anneal_cycle', type=int, default=100, help='cycle length (in number of iterations, not epochs) for the cyclic KL annealing rule. If (-1) then no KL annealing is done (useful for resuming from a previous epoch at an already-annealed KL stage.')
        self.parser.add_argument('--enc_kl_anneal_formula', type=str, default='ramp', choices=['ramp', 'cyclic'], help='type of KL annealing formula. Use either "ramp" or "cyclic".')
        self.parser.add_argument('--latent_size', type=int, default=8, help='number of dimensions for latent vectors')
        self.parser.add_argument('--enc_fc_hidden_sizes', type=int, nargs='*', default=[64,64], help='sizes of each hidden layer in the FCs that map the encoder final hidden state to sampling parameters for the latent vector')
        self.parser.add_argument('--dec_fc_hidden_sizes', type=int, nargs='*', default=[64,64], help='sizes of each hidden layer in the FC net that maps the latent vector to the initial decoder hidden state')

        # loss options
        self.parser.add_argument('--reco_loss_type', type=str, default='l2', choices=['l1', 'l2'], help='reconstruction loss metric; choose either l1 or l2.')
        self.parser.add_argument('--binary_flag_loss_weight', type=float, default=1.0, help="weight for the binary crossentropy loss term, default 1.0")

        # learning setup and checkpointing options
        self.parser.add_argument('--gpu_ids', type=int, default=[0], nargs='+', help='gpu ids space separated. use -1 for CPU')
        self.parser.add_argument('--save_dir', type=str, help='directory in which to store model checkpoints')
        self.parser.add_argument('--load_epoch', type=int, help='which epoch to load for test/continuing training')
        self.parser.add_argument('--count_from_epoch', type=int, help="which epoch to count from within the specified niter+niter_decay plan, for LR scheduling and loss/checkpoint logging purposes. If unspecified, defaults to 0 (and the LR scheduler will start its epoch count from scratch)")
        self.parser.add_argument('--niter', type=int, default=20, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=20, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
        self.parser.add_argument('--optimizer', type=str, choices=["adam", "adamw"], default="adam", help="choose an optimizer; default: 'adam'")

        # dataset options
        self.parser.add_argument('--data_norm_by', type=str, choices=['per_value', 'whole_array', 'none'], default='per_value', help='per_value normalizes the sequence dataset by coordinate; whole_array calculates a single mean/std scalar using every element in all sequences')
        self.parser.add_argument('--loop_repr_type', type=str, choices=['ellipse-single', 'ellipse-multiple', 'fixed-res-polyline'], default='ellipse-multiple', help='choose the loop representation to use. "ellipse-single" allows fitting one ellipse per plane level; "ellipse-multiple" allows multiple ellipses per level, with a binary flag each timestep to determine levelup')
        self.parser.add_argument('--use_eos_token', type=bool, default=False, help='in generating data, append a special EOS vector at the end of each loop sequence (defined as all-zeros params but with a 1 levelup flag); in inference, detect this EOS embedding to stop the sequence gen early.')
        self.parser.add_argument('--batch_size', type=int, default=4, help="batch size, in number of meshes/sequences-of-slices per batch")
        self.parser.add_argument('--dataroot', help='path to meshes (should have subfolders train, test)')
        self.parser.add_argument('--mode', type=str, choices=['test', 'train'], help='either test|train. ')
        self.parser.add_argument('--optfile', type=str, default='', help='specify a file with all these arguments rather than parsing them from the command line')
        self.parser.add_argument('--load_test_set_for_inference', action='store_true', help='whether to load the test set for inference.py rather than the train set (which is the default)')

        self.opt = None
    def parse_cmdline(self, line=None):
        self.opt, unknown = self.parser.parse_known_args(None if line is None else line.split(' '))
        if self.opt.optfile:
            return self.parse_from_file(self.opt.optfile)
        return self.after_parse()

    def parse_from_file(self, fname):
        print('[IO   ] Overriding command line args and getting arguments from file')
        with open(fname, 'r') as f:
            argv = filter(lambda line: line != '', map(lambda line: line.strip(), f))
            argv = filter(lambda line: line[0] != '#',argv)
            argv = map(lambda line: line.strip('\n'), argv)
            argv = ' '.join(argv).split(' ')
        self.opt, unknown = self.parser.parse_known_args(argv)
        print(f"[IO   ] WARNING: unknown arguments provided: {unknown}. "
        "Check if you're not mistakenly setting an argument with a typo/wrong name.")
        return self.after_parse()
    
    def after_parse(self):
        # sanity checks after parsing arguments
        def __enforce_arg(argname):
            assert has_field(self.opt, argname), f'needs argument {argname}'
        __enforce_arg('mode')
        __enforce_arg('save_dir')
        assert self.opt
        self.opt.is_train = self.opt.mode == 'train'
        if self.opt.mode == 'npz':
            assert has_field(self.opt, 'single_npz'), "No npz file specified"
            assert has_field(self.opt, 'load_epoch'), "No epoch specified to load"
        elif self.opt.mode == 'test':
            assert has_field(self.opt, 'load_epoch'), "No epoch specified to load"
            # make sure that in this case save_dir exists
            assert os.path.isdir(self.opt.save_dir), \
                "--load_epoch is specified but specified --save_dir is not a valid directory"
        if self.opt.mode == 'train' or self.opt.mode == 'test':
            __enforce_arg('dataroot')
        
        if self.opt.mode == 'train':
            mkdir(self.opt.save_dir)
        print("========= OPTIONS for run =========")
        for (k, v) in self.opt.__dict__.items():
            print(f"{k}: {v}")
        print("========= end  OPTIONS    =========")

        # postprocess the loop_repr_type option...
        # we need to convert that to an int 
        if self.opt.loop_repr_type == "ellipse-single":
            self.opt.loop_repr_type = 0
        elif self.opt.loop_repr_type == "ellipse-multiple":
            self.opt.loop_repr_type = 1
        elif self.opt.loop_repr_type == "fixed-res-polyline":
            self.opt.loop_repr_type = 3
        else:
            raise ValueError("Unknown/unimplemented choice for --loop_repr_type")
        
        # the gpu_ids option: if it contains -1, then we clear the list because
        # the other modules expect an empty gpu_ids list in case of no-gpu
        if self.opt.gpu_ids and self.opt.gpu_ids[0] == -1:
            self.opt.gpu_ids = None
        
        # save the actual ccommand used to invoke
        
        cmdline = " ".join(map(shlex.quote, sys.argv[1:]))
        self.opt.original_command = cmdline

        return self.opt


