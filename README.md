# FluxDisco: Symbolic Regression for Stoichiometric Dynamical Systems via Monte Carlo Graph Search 🪩

## Installation ⚡
This codebase currently supports Python 3.14. It can be installed directly via `pip` (recommended for users) or by cloning the repository.

### Installation via `pip`
To install with `pip`, run the following command in terminal:
```
pip install git+https://github.com/CassandraDurr/FluxDisco.git
```

### Installation via Git Clone
To clone the repository and install it locally, run the following commands in terminal:
```
git clone https://github.com/CassandraDurr/FluxDisco.git
cd FluxDisco
pip install -e .
```

*(Optional)* If you plan to contribute to this package, you should also install additional development dependencies:
```
pip install -r requirements/requirements-dev.txt
```
These include tools for testing, code formatting and quality checking.

## Running FluxDisco 🚀
Within the `fluxdisco/example` folder, there is a notebook `example.ipynb` that shows how to run FluxDisco for the coupled Lotka-Volterra ODE system.

**Expected Output:** Running FluxDisco will output the search's results to a specified directory. For example:
```
top_equations, results_df = run_fluxdisco(
    data=lv_data,
    system_config=system_config,
    search_params=search_params,
    experiment_name="lotka_volterra_example",
    save_dir="example_results",
)
```
will output the results to `example_results/lotka_volterra_example/` (which can be seen in the `fluxdisco/example` folder). The code also outputs the top equations and full set of results for direct usage.

Code for reproducing the paper results can be found in the `fluxdisco/paper-results` directory along with simulated datasets for all case studies considered in the paper. A separate `README` is available in the `fluxdisco/paper-results` directory describing how to reproduce the experimental results from the paper.

## License 📄
This project is licensed under the MIT License - see the `LICENSE` file for more details.
