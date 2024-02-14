# Quick start
You can use FIGARO either using the provided CLI or including it directly in your `.py` script. 
In this page we will describe how to use the CLI (the simplest way to use FIGARO), pointing the users interested in writing their own scripts to [this jupyter notebook](https://github.com/sterinaldi/FIGARO/blob/main/introductive_guide.ipynb).

FIGARO comes with two main CLI:
 * `figaro-density`: reconstructs a probability density given a set of samples;
 * `figaro-hierarchical`: performs a hierarchical inference given different probability densities (each represented by a set of samples).
 
 Both CLI are automatically installed with FIGARO. You can check it by running `figaro-density -h` and `figaro-hierarchical -h`: this will print the help pages for the scripts.

## `figaro-density`

The `figaro-density` CLI reconstructs a probability density given a set of samples. Let's assume to have a folder structure as this:
```
my_folder
└── events
    ├─ event_1.txt
    ├─ event_2.txt
    └─ event_3.txt
```
We want to reconstruct `event_1.txt`, and we are in `my_folder`. 
