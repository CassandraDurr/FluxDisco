# Paper results

This folder contains the code required to reproduce the results from our FluxDisco paper. To keep the repository size manageable, we have only included scripts for running the experiments and the simulated datasets as opposed to providing the full set of results.

## Requirements ⚙️
To run this code, you will need to install the FluxDisco package and its dependencies via:
```
pip install git+https://github.com/CassandraDurr/FluxDisco.git
```
The repository currently supports Python 3.14. The code requires a multi-core CPU machine, but does not require any GPUs.

## Reproducing FluxDisco experiments 💻️
The main scripts for running the paper's case studies are:
- **Base experiments**: `run_base.py`
- **Ablation with no imposed stoichiometry**: `run_no_stoichiometry.py`

This directory also includes a utilities file (`utils.py`) that stores helper functions used to run the main scripts.

To run an experiment, execute
```
python run_base.py --system <system_name> --results_dir <path/to/results>
```
where `system_name` can be:
- `sir`
- `lotka-volterra`
- `brusselator`
- `fairen-velarde`
- `all`

*Note*: Running either of the experiment scripts will output the search's results to the directory specified by the `results_dir` argument.

## Additional ablation studies 📈️
The main text and supplementary material also reference two additional ablation studies. These experiments can be reproduced by making small changes to the base experiments script (`run_base.py`):
### Ablation: No state merging
To determine the effect of state merging, the boolean parameter `prevent_merging` can be toggled on and off. The default behaviour is to allow state merging and utilise a graphical search structure. To consider a tree search structure instead, set:
```
search_params = {
    # ... other parameters ...
    "prevent_merging": True
}
```

### Ablation: Grammar rule exclusions
To encode known domain knowledge, some grammar rules are excluded from sampling by setting their corresponding probabilities to zero via the `flux_priors`. These probabilities can be changed from `0` to `1` to determine the effect of removing these exclusions.

## Simulated data 📂️
The data required for these models is simulated when running the scripts above.
However, we also provide static versions of the simulated data for all case studies in the `data` directory. This simulated data is utilised in the benchmark comparison study.

## Benchmark comparison study 📊️
The directory `benchmark-study` contains code for reproducing the results from our comparative study with:
- Symbolic Physics Learner (`spl_comp.py`)
- SINDy (`sindy_comp.py`)
- PySR (`pysr.py`)
- ODEFormer and ODEFormer (Opt) (`odeformer_comp.py`)
- ProGed (`proged_comp.py`)

The various benchmarked methods have conflicting package requirements, therefore a requirements file is available per method in `benchmark-study/requirements`. All methods can be run on the same python version though, Python `3.11.15`. To install the method-specific packages, run:
```
conda create -n <method>_env python=3.11.15 -y
conda activate <method>_env
pip install -r requirements/requirements-<method>.txt
```
where `method` is one of `spl | sindy | pysr | odeformer | proged`.

The benchmarked methods can be run with
```
python <method>_comp.py
```
from the `benchmark-study` directory. Like the base FluxDisco code, the benchmark code does not require GPUs to run.

## References 📚️
- **Symbolic Physics Learner**: Fangzheng Sun et al. “Symbolic Physics Learner: Discovering governing equations via Monte Carlo
tree search”. In: *arXiv preprint arXiv:2205.13134* (2022)
- **SINDy**: Steven L Brunton, Joshua L Proctor, and J Nathan Kutz. “Discovering governing equations from
data: Sparse identification of nonlinear dynamical systems”. In: *Proceedings of the National
Academy of Sciences* 113.15 (2016), pp. 3932–3937.
- **PySR**: Miles Cranmer. “Interpretable Machine Learning for Science with PySR and
SymbolicRegression.jl”. In: *arXiv preprint arXiv:2305.01582* (2023).
- **ODEFormer**: Stéphane d’Ascoli et al. “ODEFormer: Symbolic Regression of Dynamical Systems with Transformers”. In: *arXiv preprint arXiv:2310.05573v1* (2023)
- **ProGED**: Nina Omejc et al. “Probabilistic grammars for modeling dynamical systems from coarse, noisy,
and partial data”. In: *Machine Learning* 113.10 (2024), pp. 7689–7721.
