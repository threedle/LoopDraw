# LoopDraw: a Loop-Based Autoregressive Model for Shape Synthesis and Editing
*Nam Anh Dinh, Haochen Wang, Greg Shakhnarovich, Rana Hanocka*

**[arXiv](https://arxiv.org/abs/2212.04981) | [project page](https://threedle.github.io/LoopDraw/)**

![teaser](https://threedle.github.io/LoopDraw/l_teaser.png)


## Setup

### Conda environment
```
conda env create -f environment.yml
conda activate loopdraw
```

### Datasets and models

#### **Datasets**

The datasets featured in the experiments in the paper can be downloaded [from this Drive folder](https://drive.google.com/drive/folders/1ClVknlE3xf24X3T-2Nt8dVHfQ3XGdpQu?usp=share_link). Each `.tgz` file contains at its top level a `train` and (for the sofas set) a `test` directory.

Given a `datasets/` directory, extract as follows. For example, after downloading `shapenet-sofas.tgz`:
```
mkdir datasets/shapenet-sofas
cd datasets/shapenet-sofas
tar xvzf <path to your shapenet-sofas.tgz>
```

The paths should look like this:

```
datasets/
|- shapenet-sofas/
   |- train/
      |- (all .obj files; optional if preprocessed_cache.npz is present)
      |- planespec.txt
      |- preprocessed_cache.npz
   |- test/
      |- (all .obj files; optional if preprocessed_cache.npz is present)
      |- planespec.txt
      |- preprocessed_cache.npz
```

In general, a dataset folder (a dataroot) must be a folder that contains a `train` and optionally a `test` subdirectory. 
In the `train/` (and `test/` if present) directory, there must be a `planespec.txt` file that describes the slice planes used for the dataset. If there also exists a file `preprocessed_cache.npz`, this will be loaded as the dataset. Otherwise, all `.obj` mesh files in `train/` (or `test/`) will be gathered, sliced into loops according to `planespec.txt`, and stored as `preprocessed_cache.npz`.
The provided datasets already contain `preprocessed_cache.npz` for each train and test directory.


#### **Model checkpoints**

The checkpoints featured in the experiments in the paper can be downloaded from [this Drive folder](https://drive.google.com/drive/folders/1SK4Gy_gwoVXVzcokiTb1h5m729y6jsZw?usp=share_link). Each `.tgz` file contains at the top level a folder with the model name, and under that, two files `enclatest_net.pth` and `declatest_net.pth`, corresponding to the encoder and decoder.

Given a `checkpoints/` directory, extract as follows. For example, after downloading `shapenet-sofas-0.02KL-10-24.tgz`:
```
cd checkpoints
tar xvzf <path to your shapenet-sofas-0.02KL-10-24.tgz>
```
And the paths should look like
```
checkpoints/
|- shapenet-sofas-0.02KL-10-24/
   |- enclatest_net.pth
   |- declatest_net.pth
```
A checkpoint save directory must contain those two `.pth` files. When saving files from inference runs, output files will be saved in a folder `inference/` at the same directory level as the `.pth` files.

## Running the code

### Inference
LoopDraw offers an interactive REPL (console) interface for running and interacting with inference experiments.

Using the provided options files, run the code as follows. (Check the optfiles first to make sure paths are correct for your setup, and more.)

```
python loop_models_inference.py --optfile vases-optfile.txt
```
```
python loop_models_inference.py --optfile sofas-optfile.txt
```

Entering the interactive inference REPL, you can type commands and execute them with Return. The `help` command gives a manual on how to use the REPL. (Also see the doc comments in `loop_inference_config_and_repl.py`). 

The highlights:
- submit runs with `submit`. For instance, to see 10 random samples from the model, type `submit sample 10`. This puts 10 actions onto the queue.
- run queued actions with `run all`, or just the latest queued action with `run`.
- `submit interp 10 ||latent-a.txt ||latent-b.txt` will load two latent vectors from arrays saved in the two text files *relative to the model's `--save_dir`*, then queue 10 decoding tasks that decode 10 shapes interpolating between the two latents.
- `submit decode ||latent-a.txt` will queue one task that decodes the given latent.
- Anywhere a latent code is needed as an argument, you can specify one of the following
  -  `?` to use a random latent vector sampled from standard Gaussian
  -  `$varname` to use a latent vector saved under the name `varname` in the REPL
  -  `|latent-a.txt` to use a latent vector saved in a file `latent-a.txt` relative to the repo/python working directory.
  -  `||latent-a.txt` to use a latent vector saved in a file `latent-a.txt` relative to the model's `--save_dir`.
- For `submit decode`, `submit sample`, and `submit interp`, you may also specify `--intervene` to apply the "loop intervention function". When tasks queued with `--intervene` are executed, the file `loop_inference_intervention_func.py` will be hot-reloaded, and the function within it will be applied on the fly on each new generated timestep during inference. This is how we do loop editing / intervention experiments. See the comments in that file for more details.
- You may also specify the argument `--save` on all `submit` commands to save the run's output to the model's `--save_dir` in a subfolder called `inference`. The saved outputs consist of an `inference-#.obj` file, an `inference-#-slices.npz` file (for visualizing loops), and an `inference-#-latent.txt` for loading
with the `||latent.txt` or `|latent.txt` syntax above. (They are the same format as files saved with
`np.savetxt` if you'd like to save from your own code.)

Inference will invoke a Polyscope 3D viewer window upon running each action. Polyscope requires a display and an OpenGL (or other 3D) device; to run this on a pure CLI with no display, specify the environment variable `NO_POLYSCOPE=1`. (If the environment also has no GPU, add the option `--gpu_ids -1` into the optfile/command line arguments to use CPU only.)



Because this REPL uses stdin, you can also automate inference tasks by piping in a string of commands separated by newlines. For instance, running inference to get 10 random samples, without showing any Polyscope windows, can be done in one line with
```
printf "submit sample 10 --save\nrun all" | NO_POLYSCOPE=1 python loop_models_inference.py --optfile vases-optfile.txt
```

### Training
The training entrypoint is `loop_models_main.py`. To run training, you can use the same optfiles provided, but change `--mode test` to `--mode train`. 
Then, simply do (for example)
```
python loop_models_main.py --optfile vases-optfile.txt
```

Note that if the dataset does not already contain a `preprocessed_cache.npz`, running any of these scripts will trigger preprocessing (i.e. slicing and caching) to create the `preprocessed_cache.npz` file. This can take quite a long time if there are many `.obj` files in your dataset. To run just dataset generation alone (without continuing to inference or training), the entrypoint is `loop_models.py`. 

```
python loop_models.py --optfile vases-optfile.txt
```
(Dataset generation is robust to interruptions from i.e. running out of cluster job time. The code will save a `preprocessed_cache_PARTIAL.npz` in the same directory as the `.obj` files every 25 new meshes loaded, and can automatically resume from that checkpoint the next time this is run. These `preprocesed_cache_PARTIAL.npz` files can themselves be used as a fully-fledged dataset cache file like `preprocessed_cache.npz` if you'd like to go straight to training on what has already been preprocessed. Simply rename them to `preprocessed_cache.npz`.)

By default, training will save a checkpoint every 10 epochs. To resume training, specify `--load_epoch -1` and `--count_from_epoch <latest epoch num that was logged>`, and optionally `--enc_kl_annealing_cycle -1` to pin the KL weight to `--enc_kl_weight` if you would like to not restart the KL ramp on each resumption.

Check out `loop_models_options.py` (or run any of the above with `--help`) for more information on the commands available. Note that any of these scripts can be run with an `--optfile optfile.txt` (as in the above examples; this will disregard other command line arguments) or with written-out command line arguments in python argparse style.

### Visualizing results

When inference actions are run with `--save`, the output will consist of an `inference-#.obj` file as well as an `inference-#-slices.npz` file. These can be visualized with the script `polyscope_screenies.py`, which takes as arguments paths to .obj files. (Loops will also be visualized if a `meshname.obj` also has `meshname-slices.npz` in the same location.) 

See the top-level variables in that script to customize the look of rendered images.

