from typing import Union, Tuple
import sys
import numpy as np
import torch
import loop_models as lmodels
import loop_model_options as loptions
from thlog import *

thlog = Thlogger(LOG_DEBUG, VIZ_NONE, "train", imports=[lmodels.thlog])

DRY_RUN = False

SAVE_INTERVAL = 10

########## Training and testing code ##############

def loss_averager_multiloss(loss_list: list) -> Union[float, Tuple[float, ...]]:
    # calculates average loss(es) from a list of iterables of loss metrics;
    # used for models which report multiple sub-losses to add together (in our case, KL and reco)
    if isinstance(loss_list[0], float):
        return float(np.mean(loss_list))
    return tuple(np.mean(np.array(list(map(list, loss_list))), axis=0))

def calc_summary_loss(loss_tuple_or_val):
    return sum(loss_tuple_or_val) if isinstance(loss_tuple_or_val, tuple) else loss_tuple_or_val

def run_training_loop(opt, model, train_dataset: lmodels.LoopSeqDataset, test_callback=None, save_by_best_loss=False):
    min_train_loss_so_far = 9999
    min_test_loss_so_far = 9999

    save_interval_counter = 0
    model.save_options()
    count_from_epoch = opt.count_from_epoch if loptions.has_field(opt, 'count_from_epoch') else 0
    assert count_from_epoch >= 0, "--count_from_epoch must be a nonnegative integer"
    for epoch_num in range(count_from_epoch, opt.niter + opt.niter_decay):
        thlog.info("")
        thlog.info(f"epoch {epoch_num} start")
        losses = []
        
        
        data_indices = list(range(len(train_dataset)))
        train_dataset.shuffle_batches() 
        # better shuffling, shuffles the items within each batch, rather than
        # just shuffling the order of predetermined fixed batches

        for data_i in data_indices:
            data_item = train_dataset[data_i]
            # add code to retrieve the manual latent stuff here if we want
            out, _, loss_val = model.forward_and_backward(data_item)
            losses.append(loss_val)
            #print(input_seq, target_seq)
        
        average_loss = loss_averager_multiloss(losses)  # this could be a tuple of sublosses or just a single loss
        summary_loss = calc_summary_loss(average_loss)  # returns the loss value itself if just a single value; otherwise take sum of the sublosses
        if isinstance(average_loss, tuple):
            thlog.info('epoch {} done; avg losses {} = {:5.8f}'.format(epoch_num, average_loss, summary_loss))
        else:
            thlog.info('epoch {} done average loss: {:5.8f}'.format(epoch_num, average_loss))

        model.write_loss_log(epoch_num, average_loss, False)
        sys.stdout.flush() # so that the stdout log is continually updated...
        # save a rolling copy of the latest net, in case we don't finish til the
        # end 2022-06-09: don't save every epoch, that taxes the hard drive way
        # too much; save every interval ish
        if save_interval_counter % SAVE_INTERVAL == 0:
            model.save_network("latest")
        save_interval_counter += 1
        
        epoch_already_saved = False
        
        if callable(test_callback):
            test_loss = calc_summary_loss(test_callback(epoch_num))
            if test_loss < min_test_loss_so_far:
                min_test_loss_so_far = test_loss
                if save_by_best_loss:
                    thlog.info('[TEST] epoch {} test loss better than previous best test loss; saving checkpoint'.format(epoch_num))
                    model.save_network(epoch_num)
                    epoch_already_saved = True
        model.update_learning_rate(average_loss)
        if summary_loss < min_train_loss_so_far:
            min_train_loss_so_far = summary_loss
            extra_text = "; saving checkpoint" if not epoch_already_saved else " as well"
            if save_by_best_loss:
                thlog.info('epoch {} loss better than previous best train loss{}'.format(epoch_num, extra_text))
                if not epoch_already_saved:
                    model.save_network(epoch_num)
                    epoch_already_saved = True
        if epoch_num == opt.niter_decay + opt.niter - 1:
            thlog.info('epoch {} saving due to being the last epoch'.format(epoch_num))
            if not epoch_already_saved:
                model.save_network(epoch_num)
            model.finalize_after_training() # just bookkeeping printing stuff at the end; for 2021/12/26 experiment, we print the latents
    thlog.info('done')

def run_tests(opt, model, test_dataset: lmodels.LoopSeqDataset):
    with torch.no_grad():
        losses = []
        outs = []
        for data_item in test_dataset:
            out, _, loss_val = model.forward_and_backward(data_item, is_train=False)
            losses.append(loss_val)
            outs.append(out)
        if losses[0] is not None: # if None then there's no target, i.e. doing split inference
            average_loss = loss_averager_multiloss(losses)
            thlog.info('[TEST] average test loss: {} = {:5.8f}'.format(average_loss, calc_summary_loss(average_loss)))
    return outs, losses


if __name__ == "__main__":
    if DRY_RUN:
        thlog.info("Currently in a DRY-RUN, no checkpoints will be saved!")
    opt = loptions.LoopSeqOptions().parse_cmdline()
    assert opt
    assert opt.mode in ('train', 'test')

    has_test_dataset = os.path.isdir(os.path.join(opt.dataroot, 'test'))
    
    # if we are training, the main set to take these values from is 'train'
    if opt.mode == 'train':
        train_dataset = lmodels.LoopSeqDataset(opt, override_mode='train')
        dataset_n_input_features = train_dataset.n_input_features
        dataset_n_steps = train_dataset.n_steps
        if has_test_dataset:
            test_dataset = lmodels.LoopSeqDataset(opt, override_mode='test')
            # there are separate train and test sets, make sure they match in max timestep length,
            # padding and reloading if necesary.
            if train_dataset.n_steps != test_dataset.n_steps:
                __max_dataset_n_steps = max(train_dataset.n_steps, test_dataset.n_steps)
                dataset_n_steps = __max_dataset_n_steps
                if train_dataset.n_steps != __max_dataset_n_steps:
                    # reload with padding to this max seq length value
                    thlog.info(f"[DATA] Reloading & padding train set to max length of {__max_dataset_n_steps}")
                    train_dataset = lmodels.LoopSeqDataset(opt, override_mode='train', pad_to_max_sequence_length=__max_dataset_n_steps)
                elif test_dataset.n_steps != __max_dataset_n_steps:
                    # reload with padding to this max seq length value
                    thlog.info(f"[DATA ] Reloading & padding test set to max length of {__max_dataset_n_steps}")
                    test_dataset = lmodels.LoopSeqDataset(opt, override_mode='test', pad_to_max_sequence_length=__max_dataset_n_steps)
            
        else:
            test_dataset = train_dataset
    elif opt.mode == 'test':
        # the train dir should always be present
        if has_test_dataset:
            thlog.info("Loading the two sets first to get the max sequence length used in training")
            train_dataset = lmodels.LoopSeqDataset(opt, override_mode='train')
            test_dataset = lmodels.LoopSeqDataset(opt, override_mode='test')
            trained_on_max_seq_len = max(train_dataset.n_steps, test_dataset.n_steps)
            thlog.info("Ok now loading the test set to run whole-test-set tests.")
            test_dataset = lmodels.LoopSeqDataset(opt, override_mode='test', pad_to_max_sequence_length=trained_on_max_seq_len)
        else:
            train_dataset = None
            test_dataset = lmodels.LoopSeqDataset(opt, override_mode='train')
        dataset_n_input_features = test_dataset.n_input_features
        dataset_n_steps = test_dataset.n_steps
    else:
        raise ValueError("unknown mode, use either 'train' or 'test'")
        
            
    thlog.info(f"number of time steps per sequence is {dataset_n_steps}")
    opt.lstm_n_steps = dataset_n_steps # this is just so that n_steps is printed in opt.txt
    model = lmodels.LoopSeqEncoderDecoderModel(opt, dataset_n_input_features, dataset_n_steps, dry_run=DRY_RUN)
    model.print_detailed_network_stats()
    if opt.mode == 'test':
        thlog.info("running only in TEST mode")
        run_tests(opt, model, test_dataset)
    elif opt.mode == 'train':
        thlog.info("running in TRAINING mode")
        def __test_callback(epoch_num):
            thlog.info(f'[TEST] epoch {epoch_num} running test')
            outs, losses = run_tests(opt, model, test_dataset)
            average_test_loss = loss_averager_multiloss(losses)
            if not isinstance(average_test_loss, tuple):
                average_test_loss = (float(average_test_loss),)
            model.write_loss_log(epoch_num, average_test_loss, True)
            return average_test_loss
        
        __do_nothing = None
        assert train_dataset, "train dataset must be present in train mode"
        run_training_loop(opt, model, train_dataset, test_callback=__do_nothing, save_by_best_loss=False)
