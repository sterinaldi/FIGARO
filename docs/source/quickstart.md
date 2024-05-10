# Quick start
You can use FIGARO either using the provided CLI or including it directly in your `.py` script. 
In this page we will describe how to use the CLI (the simplest way to use FIGARO), pointing the users interested in writing their own scripts to the relevant documentation page.

FIGARO comes with two main CLI:
 * `figaro-density`: reconstructs a probability density given a set of samples;
 * `figaro-hierarchical`: performs a hierarchical inference given different probability densities (each represented by a set of samples).
 
 Both CLI are automatically installed with FIGARO. You can check it by running `figaro-density -h` and `figaro-hierarchical -h`: this will print the help pages for the scripts.

## `figaro-density`

The `figaro-density` CLI reconstructs a probability density given a set of samples. Let's assume to have a folder structure as this:
```
my_folder
└── events
    ├─ event_1.txt
    ├─ event_2.txt
    └─ event_3.txt
```
We want to reconstruct `event_1.txt`, which contains a set of 2-dimensional samples and is structured as follows:
```
# X Y Z
x1 y1 z1
x2 y2 z2
x3 y3 z3
x4 y4 z4
...
```
The only other required thing, other than the samples, are the minimum and maximum allowed values for our samples to take, say Xmin, Xmax, Ymin, Ymax, Zmin, Zmax. Please note that these **must not** be the smallest and largest samples, otherwise FIGARO will raise an error.
From `my_folder`, the minimal instruction to run is 

```
figaro-density -i events/event_1.txt -b "[[Xmin, Xmax],[Ymin, Ymax],[Zmin, Zmax]]"
```

This will draw 100 realisations of the DPGMM distributed around the true underlying distribution. As soon as the run is finished (depending on the number of available samples and the dimensionality of the problem, the runtime may vary from tens of seconds upwards), the folder will look something like this:

```
my_folder
├── events
│   ├─ event_1.txt
│   ├─ event_2.txt
│   └─ event_3.txt
├── draws_event_1.json
├── event_1.pdf
├── log_event_1.pdf  (only if the distribution is 1D)
├── prob_event_1.txt (only if the distribution is 1D)
└── options.ini
```

`event_1.pdf` and `log_event_1.pdf` (this is produced only if the samples are one-dimensional) show the reconstructed probability density, whereas `draws_event_1.json` contains the individual draws that has been produced by FIGARO (see below for how to use them). `prob_event_1.txt` contains the probability values used to produce `event_1.pdf`.
`options.ini` contains a summary of all the options (provided and default) for the run. It can be used both as a log file and to reproduce the run with the same settings via

```
figaro-density --config options.ini
```

An example of options file with some suggestions on how to customise it can be found [here](https://github.com/sterinaldi/FIGARO/blob/main/options_example.ini).

If instead of a single file we point `figaro-density` to a folder with multiple files, e.g. with

```
figaro-density -i events -b "[[Xmin, Xmax],[Ymin, Ymax],[Zmin, Zmax]]"
```

The CLI will gather all the suitable files in the folder and will produce a reconstruction per file. Eventually, the folder will look like this:
```
my_folder
├── events
│   ├─ event_1.txt
│   ├─ event_2.txt
│   └─ event_3.txt
├── draws
│   ├─ draws_event_1.json
│   ├─ draws_event_2.json
│   └─ draws_event_3.json
├── density
│   ├─ event_1.pdf
│   ├─ event_2.pdf
│   └─ event_3.pdf
├── log_density
│   ├─ log_event_1.pdf
│   ├─ log_event_2.pdf
│   └─ log_event_3.pdf
├── txt
│   ├─ prob_event_1.txt
│   ├─ prob_event_2.txt
│   └─ prob_event_3.txt
└── options.ini
```
Keep in mind that this is **not** a hierarchical analysis.

### Options

Several options are available to further customise the behaviour of `figaro-density`. Below you find a complete list of all the available options.

* `-i FILE, --input=FILE`: file or folder with the samples to analyse. **REQUIRED!**;
* `-b BOUNDS, --bounds=BOUNDS`: minimum and maximum allowed values for our samples to take per dimension. Must be formatted as "[[Xmin, Xmax], [Ymin, Ymax],...]". For 1D distributions use "[Xmin, Xmax]". **REQUIRED and quotation marks are mandatory!**
* `-o OUTPUT_FOLDER, --output=OUTPUT_FOLDER`: folder where to put the output files. Default values is the parent directory of the samples directory;
* `--ext=EXTENSION`: format of the output file. Can either be `json` (default, recommended) or `pkl`;
* `--inj_density=INJECTED_FILE`: if you know the true underlying distribution for whathever reason (simulations or other analyses), you may want to include it in your plots. You just have to prepare a `.py` file that includes a method called `density(x)` (the name is important and it must take only one parameter, possibly a vector, as input) with your pdf. This distribution will be shown in the produced plots. This option is available only for 1-dimensional distributions;
* `--selfunc=SELECTION_FUNCTION_FILE`: if your data are affected by selection bias but you have a model for the selection function, you can prepare a `.py` file including a method called `selection_function(x)` with the same prescriptions as above. FIGARO will deconvolve the selection effects from your data (only 1-dimensional distribution). Making use of this option will produce three additional files: `true_event_1.pdf`, `log_true_event_1.pdf` and `prob_true_event_1.txt`. WARNING: check that your selection function does not return zeros;
* `--parameter=PAR`: GW parameter(s) to use for posterior probability density reconstruction (available only for [LVK data release files](https://gwosc.org/eventapi/html/GWTC/) – this option will be ignored for other files). For a list of available GW parameters, run `python -c 'from figaro.load import available_gw_pars; available_gw_pars()'`;
* `--waveform=WF`: waveform family used to generate the posterior samples. Default is *combined*, but *imr* and *seob* are also available. To be used together with `--parameter` only. If you're not familiar with different waveform families, you probably won't need this option;
* `-p, --postprocess`: produce the final plots only without re-running the whole analysis. Useful if you want to change only the output options;
* `--symbol=SYMBOL`: LaTeX-style quantity symbol, for plotting purposes. Dollar signs before and after ($$) are not needed and you'll probably need to doube all the backslashes. For the example presented here, we could use `--symbol \\mathrm{X},\\mathrm{Y},\\mathrm{Z}` (no spaces). If no symbols are provided, FIGARO will use x1,x2,...xn as placeholders;
* `--unit=UNIT`: LaTeX-style quantity unit, for plotting purposes. Same prescriptions as above applies. If not provided, no units will be plotted. Leave blank these parameters that does not have units: `--units \\mathrm{cm},,\\mathrm{s}`;
* `--draws=N_DRAWS`: number of draws, default 100. The more, the better (you don't say?);
* `--n_samples_dsp=N_SAMPLES`: number of samples to use for the analysis. Default is to use all the samples, but you can specify a smaller subset. Samples are randomly drawn;
* `--exclude_points`: FIGARO, by default, performs a completely user-transparent coordinate change from the rectangle defined by the bounds to :math:`\mathbb{R}^N` (see Section 3.1 of [Rinaldi & Del Pozzo (2022)](https://ui.adsabs.harvard.edu/abs/2022MNRAS.509.5454R/abstract) for the definition). Since the transformation is not defined on or outside the boundaries, all the samples must lie within these, and this option prunes all the samples that are outside the specified bounds;
* `--cosmology=h,om,ol`: cosmological parameters to map luminosity distance samples into redshift samples (again, if you don't know what we're talking about, you probably won't need this option);
* `--sigma_prior=SIGMA_PRIOR`: expected width of the features to be found in the reconstructed distribution. By default, this is estimated using the available samples using the `fraction` option instead of this one. It can either be a single value (valid for all the dimensions) or a N-uple of values, one per dimension;
* `--fraction=FRACTION`: Estimates the expected width of the features as `std(samples)/fraction`. Default is 5. This option is overrided by `sigma_prior`, if both are provided;
* `--snr_threshold=TH`: signal-to-noise threshold to filter simulated GW catalogs (most likely not needed);
* `--far_threshold=TH`: false alarm rate threshold to filter simulated GW catalogs (as above, most likely not needed);
* `--no_probit`: in some cases, the underlying distribution is defined over the whole real line (or :math:`\mathbb{R}^N`), therefore the coordinate change might not be needed. This option disables it, and in this case no exception will be thrown if some samples are outside the bounds;
* `--config=CONFIG_FILE`: all the above options can be stored in a single `.ini` file under the `[DEFAULT]` section and passed to `figaro-density` with this option. The config file (see [here](https://github.com/sterinaldi/FIGARO/blob/main/options_example.ini) for an example) does not need to include all the options – default values will be used for all unprovided options – and command-line options take precedence over the ones provided via config file.

A more fine-tuned analysis for the example above could be:

```
figaro-density -i events -b "[[Xmin, Xmax],[Ymin, Ymax],[Zmin, Zmax]]" --symbol \\mathrm{X},\\mathrm{Y},\\mathrm{Z} --unit \\mathrm{cm},,\\mathrm{s} --exclude_points --draws 1000 --fraction 10 --n_samples_dsp 3000
```

## `figaro-hierarchical`

Let us consider now the situation in which we have a set of posterior distributions, described by different sets of posterior samples, and we want to infer the distribution from which the observations are drawn. The `figaro-hierarchical` CLI performs the hierarchical inference, first reconstructing the individual posterior distributions and then combining them together. Let's assume, once again, to have a folder structure as this:
```
some_cool_data
└── events
    ├─ event_1.txt
    ├─ event_2.txt
    └─ event_3.txt
```
Each of these files contains, again, a set of three-dimensional samples:
```
# X Y Z
x1 y1 z1
x2 y2 z2
x3 y3 z3
x4 y4 z4
...
```
This CLI, from the user's perspective, is very similar to `figaro-density`. The main difference is that now the user must point it to a folder storing all the posterior samples sets. The minimal command to run the hierarchical analysis is

```
figaro-hierarchical -i events -b "[[Xmin, Xmax],[Ymin, Ymax],[Zmin, Zmax]]"
```

At the end of the run – beware that this analysis will take longer, both because it needs to reconstruct N posterior probability densities and because they are then combined together in the hierarchical step – the folder will look like this:

```
some_cool_data
├── events
│   ├─ event_1.txt
│   ├─ event_2.txt
│   └─ event_3.txt
├── draws
│   ├─ draws_event_1.json
│   ├─ draws_event_2.json
│   ├─ draws_event_3.json
│   ├─ posteriors_single_event.json
│   └─ draws_some_cool_data.json
├── plots
│   ├── density (only if -s is used, see below)
│   │   ├─ event_1.pdf
│   │   ├─ event_2.pdf
│   │   └─ event_3.pdf
│   ├── log_density (only if -s is used, see below)
│   │   ├─ log_event_1.pdf
│   │   ├─ log_event_2.pdf
│   │   └─ log_event_3.pdf
│   ├── txt (only if -s is used, see below)
│   │   ├─ prob_event_1.txt
│   │   ├─ prob_event_2.txt
│   │   └─ prob_event_3.txt
│   ├─ some_cool_data.pdf
│   ├─ log_some_cool_data.pdf  (only if the distribution is 1D)
│   └─ prob_some_cool_data.txt (only if the distribution is 1D)
└── options.ini
```

`some_cool_data.pdf` shows the inferred hierarchical distribution, and the relative draws are stored in `draws/draws_some_cool_data.json` (by default, the inference is named after the parent directory).

### Options

`figaro-hierarchical` has all the options and functionalities of `figaro-density`, plus some dedicated options. Below you find all the options that are not already mentioned in the previous list;
* `--name=NAME`: name for the hierarchical output files. By default, they are named after the parent folder.
* `-s, --save_se`: by default, plots for individual posterior distributions are not produced (to save both memory and computational resources). This option allow you to produce them;
* `--hier_samples=HIER_SAMPLES`: if you're running FIGARO on a set of simulated data and you know the true value for the parameters of each individual observation, you can collect them into a single `.txt` file and pass it to FIGARO. These values will be plot along the inferred distribution;
* `--se_draws=N_SE_DRAWS`: in principle there is no reason why you might want to have the same number of draws both for the individual events and for the hierarchical distribution. The `--draws` option controls the number of realisation of the hierarchical distribution, whereas the `--se_draws` option fixes the number of realisations for each single event distribution. If not provided. `--se_draws = --draws`;
* `-e, --events`: if you already have the single-event reconstructions, you can skip that part of the analysis with this option;
* `--se_sigma_prior=SE_SIGMA_PRIOR`: as above, controls the expected width of the features for the individual events. If not provided, this is estimated using the samples (each event its own set);
* `--mc_draws=MC_DRAWS`: `figaro-hierarchical` evaluates a Monte Carlo integral. This option controls the number of samples used for the integral. You have no idea of what we're talking about? Most likely you won't need this!  

An example might be

```
figaro-hierarchical -i events -b "[[Xmin, Xmax],[Ymin, Ymax],[Zmin, Zmax]]" --symbol \\mathrm{X},\\mathrm{Y},\\mathrm{Z} --unit \\mathrm{cm},,\\mathrm{s} --no_probit --se_draws 100 --draws 1000 -s
```

## Parallelised inference

In certain circumstances, e.g. when we have a large number of samples or events or if we have access to a HPC, it might be useful to take advantage of parallel computing. FIGARO comes with a parallelised version of both the scripts described above, `figaro-par-density` and `figaro-par-hierarchical`. The parallelisation is almost user-transparent and it is made using [RAY](https://www.ray.io). These two CLI works exactly as the two described above, with the same options and outputs. The only additional setting to provide is the number of parallel threads with the option `--n_parallel` (valid for both, default is 2), and RAY will take care of the rest.
