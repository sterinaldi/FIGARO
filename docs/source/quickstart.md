# Quick start
You can use FIGARO either using the provided CLI or including it directly in your `.py` script. 
In this page we will describe how to use the CLI (the simplest way to use FIGARO), pointing the users interested in writing their own scripts to [this jupyter notebook](https://github.com/sterinaldi/FIGARO/blob/main/introductive_guide.ipynb).

FIGARO comes with two main CLI:
 * `figaro-density`: reconstructs a probability density given a set of samples;
 * `figaro-hierarchical`: performs a hierarchical inference given different probability densities (each represented by a set of samples).
 
 Both CLI are automatically installed with FIGARO. You can check it by running `figaro-density -h` and `figaro-hierarchical -h`: this will print the help pages for the scripts.

## figaro-density

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
# X Y
x1 y1
x2 y2
x3 y3
x4 y4
...
```
The only other required thing, other than the samples, are the minimum and maximum allowed values for our samples to take, say Xmin, Xmax, Ymin, Ymax. Please note that these **must not** be the smallest and largest samples, otherwise FIGARO will raise an error.
From `my_folder`, the minimal instruction to run is 

```figaro-density -i events/event_1.txt -b "[[Xmin, Xmax],[Ymin, Ymax]]"```

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

`event_1.pdf` and `log_event_1.pdf` (this is produced only if the samples are one-dimensional) show the reconstructed probability density, whereas `draws_event_1.json` contains the individual draws that has been produced by FIGARO (see below for how to use them).
`options.ini` contains a summary of all the options (provided and default) for the run. It can be used both as a log file and to reproduce the run with the same settings via
```figaro-density --config options.ini```
An example of options file with some suggestions on how to customise it can be found [here](https://github.com/sterinaldi/FIGARO/blob/main/options_example.ini).

If instead of a single file we point `figaro-density` to a folder with multiple files, e.g. with
```figaro-density -i events -b "[[Xmin, Xmax],[Ymin, Ymax]]"```
the CLI will gather all the suitable files in the folder and will produce a reconstruction per file. Eventually, the folder will look like this:
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


## figaro-hierarchical

## Parallelised inference

## Using FIGARO reconstructions
