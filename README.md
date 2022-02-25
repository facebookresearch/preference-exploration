# Preference Exploration
Code for replicating experiments from the paper, Preference Exploration for Efficient Bayesian Optimization with Multiple Outcomes, published in AISTATS 2022.

## Recreaing figures in the paper
Folder `plots` contains all figures we used in the paper generated using data under `data/processed`.
`notebooks/illustrative_plot.ipynb` creates Figure 1.
`notebooks/plot.ipynb` creates all other figures.

## Rerunning the experiments
We have provided processed data for plotting and analysis under folder `data/processed`. However, if you wish to re-run the experiments, please follow the instructions below.
Results will be saved under folder `data/sim_results`.
`notebooks/clean_sim_data.ipynb` turns raw data under `data/sim_results` into the processed format. 


#### Identifying high utility designs with PE 
To re-run the experiment in Section 5.1 *Identifying High Utility Designs with PE*, run the following command:
```
./run_within_sim.sh -g[gpu index or "cpu"] -b[begin of random seeds] -e[end of random seeds] -c[comparison noise type]
```

Args:
- `g`: indicate whether you want to run the simulation on a gpu or cpu.
- `b`: Begin of random seeds used, inclusive. This experiment will be run using a range of random seeds between 0 and 255. `e - b + 1` will be the total number of replications we run using random seed b, b+1, ..., e.
- `e`: End of random seeds used, inclusive.
- `c`: comparison noise type. It should be either "constant" or "probit"

Example:
`./run_within_sim.sh -gcpu -b0 -e99 -cconstant`


#### BOPE with a single or multiple PE stages
To re-run the experiment in Section 5.2 and 5.3, run the following command:
```
./run_multi_sim.sh -g[gpu index or "cpu"] -b[begin of random seeds] -e[end of random seeds] -c[comparison noise type]
```
The arguments follow the same pattern as above.

Example:
`./run_multi_sim.sh -gcpu -b0 -e29 -cconstant`

## Reference
Lin, Zhiyuan Jerry, Raul Astudillo, Peter I. Frazier, and Eytan Bakshy. "Preference Exploration for Efficient Bayesian Optimization with Multiple Outcomes" International Conference on Artificial Intelligence and Statistics, 2022.

#### Bibtex
```
@inproceedings{lin2022preference,
 author = {Lin, Zhiyuan Jerry and Astudillo, Raul and Frazier, Peter I. and Bakshy, Eytan},
 booktitle = {International Conference on Artificial Intelligence and Statistics},
 title = {Preference Exploration for Efficient Bayesian Optimization with Multiple Outcomes},
 year = {2022}
}
```

## License
This code repo is MIT licensed, as found in the LICENSE file.
