#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 15:25:37 2022

@author: kate
"""
import multiprocessing as mp
import numpy as np
import pandas as pd
import lmfit
from lmfit import Model as Model_lmfit
from lmfit import Parameters as Parameters_lmfit
from typing import Tuple
from models.set_model import model
from modules.solve_single import solve_single_parameter_set
from config.settings import settings, parameter_estimation_problem_definition
from config.experimental_data import ExperimentalData
from plots.plots_training_data import plot_training_data
from plots.plots_parameter_estimation import plot_parameter_distributions_after_optimization, plot_chi_sq_trajectory

def generalize_parameter_labels(free_parameter_labels) -> Tuple[list, list]:
    """Generates generalized parameter labels (p_1, p_2, etc.) for use in optimization functions

    Parameters
    ----------
    free_parameter_labels
        a lists of strings containing the bounds for the free parameters

    Returns
    -------
    general_parameter_labels
        list of floats defining the generalized parameter labels

    general_parameter_labels_free
        list of floats defining which of the generalized parameter labels are free

    """
    free_parameter_indicies = []
    for i, value in enumerate(settings["parameters"]):
        label = settings["parameter_labels"][i]
        if label in free_parameter_labels:
            free_parameter_indicies.append(i)
    
    general_parameter_labels = ["p_" + str(i + 1) for i in range(0, len(settings["parameters"]))]
    general_parameter_labels_free = [
        "p_" + str(i + 1) for i in free_parameter_indicies
    ]
    
    return general_parameter_labels, general_parameter_labels_free

def define_best_optimization_results(
    df_optimization_results: pd.DataFrame,
run_type: str) -> Tuple[float, float, list]:
    """Prints best parameter set from optimization to
    console and plots the best fit to training data

    Parameters
    ----------
    df_optimization_results
        df containing the results of all optimization runs
        
    run_type
        a string defining the run_type ('default' or 'PPL')

    Returns
    -------
    r_sq_opt
        a float defining the best case r_sq value

    chi_sq_opt_min
        a float defining the best case chi2_sq value

    best_case_parameters
        a list of floats defining the best case parameters

    """

    best_case_parameters = []
    for i in range(0, len(settings["parameters"])):
        col_name = settings["parameter_labels"][i] + "*"
        val = df_optimization_results[col_name].iloc[0]
        best_case_parameters.append(val)
    best_case_parameters = [np.round(parameter, 5) for parameter in best_case_parameters]
    solutions_norm = df_optimization_results["Simulation results"].iloc[0]
    filename = "BEST FIT TO TRAINING DATA"

    r_sq_opt = round(df_optimization_results["r_sq"].iloc[0], 3)
    chi_sq_opt_min = round(df_optimization_results["chi_sq"].iloc[0], 3)
    
    if run_type == "default":
        plot_training_data(
                ExperimentalData.x,
                solutions_norm,
                ExperimentalData.exp_data,
                ExperimentalData.exp_error,
                filename
        )
        plot_chi_sq_trajectory(list(df_optimization_results["chi_sq_list"].iloc[0]))
    
        print("*************************")
        print("Calibrated parameters: ")
        for i, label in enumerate(settings["parameter_labels"]):
            print(label + " = " + str(best_case_parameters[i]))
        print("")
        
        print("Metrics:")
        print("R_sq = " + str(r_sq_opt))
        print("chi_sq = " + str(chi_sq_opt_min))
        print("*************************")

    return r_sq_opt, chi_sq_opt_min, best_case_parameters

def optimize_all(
    df_global_search_results: pd.DataFrame, run_type: str = "default", problem: dict = parameter_estimation_problem_definition
) -> Tuple[float, float, pd.DataFrame]:
    """Runs optimization for each intitial guess and saves results

    Parameters
    ----------
    df_global_search_results
        df containing the results of the global search

    run_type
        a string defining the run_type ('default' or 'PPL' or 'PPL threshold')
        
    problem
        a dictionary containing the parameter estimation problem - 
        must be provided for PPL simulations only, in which the free parameters 
        change depending on which parameter's PPL is being calculated

    Returns
    -------
    r_sq_opt
        a float defining the best case r_sq value

    chi_sq_opt_min
        a float defining the  best case chi2_sq value

    df_optimization_results
        a dataframe containing the optimization results

    best_case_parameters
        a list of floats defining the best case parameters

    """
   
    df_global_search_results['problem'] = [problem] * len(df_global_search_results.index)
    if run_type != "PPL threshold":
        df_global_search_results = df_global_search_results.sort_values(by=["chi_sq"])
        df_global_search_results = df_global_search_results.reset_index(drop=True)
        df_global_search_results = df_global_search_results.drop(
            df_global_search_results.index[settings["num_parameter_sets_optimization"] :]
        )
    
    all_opt_results = []
    if settings["parallelization"] == "no":
        for row in df_global_search_results.itertuples(name=None):
            results_row, results_row_labels = optimize_single_initial_guess(row)
            all_opt_results.append(results_row)

    elif settings["parallelization"] == "yes":
        with mp.Pool(settings["num_cores"]) as pool:
            result = pool.imap(
                optimize_single_initial_guess,
                df_global_search_results.itertuples(name=None),
            )
            pool.close()
            pool.join()
            output = [[list(x[0]), list(x[1])] for x in result]
        for i in range(0, len(output)):
            all_opt_results.append(output[i][0])
        results_row_labels = output[0][1]

    print("Optimization complete.")
    df_optimization_results = pd.DataFrame(all_opt_results, columns=results_row_labels)
    df_optimization_results = df_optimization_results.sort_values(by=["chi_sq"], ascending=True)
    df_optimization_results = df_optimization_results.reset_index(drop=True)

    r_sq_opt, chi_sq_opt_min, best_case_parameters = define_best_optimization_results(df_optimization_results, run_type)
    if run_type == "default":
        plot_parameter_distributions_after_optimization(df_optimization_results)
    df_optimization_results.to_csv("optimization results.csv")
    
    return r_sq_opt, chi_sq_opt_min, df_optimization_results, best_case_parameters

def define_parameters_for_opt(initial_parameters: list, free_parameter_labels: list, free_parameter_bounds: list) -> list:
    """Defines parameters for optimization with structure necessary for LMFit optimization code

    Parameters
    ----------
    initial_parameters
        a list of floats containing the initial guesses for each parameter
        
    free_parameter_labels
        a lists of strings containing the bounds for the free parameters
    
    free_parameter_bounds
        a lists of lists containing the bounds for the free parameters
    

    Returns
    -------
    params_for_opt
        an object defining the parameters and bounds for optimization
        (in structure necessary for LMFit optimization code)

    """
    (
        general_parameter_labels,
        general_parameter_labels_free,
    ) = generalize_parameter_labels(free_parameter_labels)

    # Set default values
    num_parameters = len(settings["parameters"])
    bound_min_list = [0] * num_parameters
    bound_max_list = [np.inf] * num_parameters
    vary_list = [False] * num_parameters

    # Set min and max bounds and vary_index by comparing free parameters
    # lists with list of all parameters
    for param_index, param_label in enumerate(general_parameter_labels):
        for free_param_index, free_param_label in enumerate(general_parameter_labels_free):

            # if param is free param, change vary to True and update bounds
            if param_label == free_param_label:
                vary_list[param_index] = True
                bound_min_list[param_index] = 10 ** free_parameter_bounds[free_param_index][0]
                bound_max_list[param_index] = 10 ** free_parameter_bounds[free_param_index][1]

    # Add parameters to the parameters class
    params_for_opt = Parameters_lmfit()
    for index_param, param_label in enumerate(general_parameter_labels):
        params_for_opt.add(
            param_label,
            value=initial_parameters[index_param],
            vary=vary_list[index_param],
            min=bound_min_list[index_param],
            max=bound_max_list[index_param],
        )

    for null_index in range(len(general_parameter_labels), 10):
        params_for_opt.add("p_" + str(null_index + 1), value=0, vary=False, min=0, max=1)

    return params_for_opt

def define_results_row(
    results: lmfit.model.ModelResult, initial_parameters: list, chi_sq_list: float
) -> Tuple[list, list]:
    """Defines results for each optimization run

    Parameters
    ----------
    results
        a results class containing results of the given optimization run

    initial_parameters
        a list of floats containing the initial guesses for each parameter

    chi_sq_list
        a list of floats containing the chi_sq values for each function evaluation


    Returns
    -------
    results_row
        a list of floats and lists containing the results for the given optimization run

    results_row_labels
        a list of strings defining the labels for each item in results_row

    """
    # add initial parameters to row for saving
    results_row = []
    results_row_labels = []
    for i, val in enumerate(initial_parameters):
        results_row.append(val)
        results_row_labels.append(settings["parameter_labels"][i])

    # Solve ODEs with final optimized parameters
    model.parameters = list(results.params.valuesdict().values())[: len(initial_parameters)]
    solutions_norm, chi_sq, r_sq = solve_single_parameter_set()

    # append best fit parameters to results_row for saving
    for index, best_val in enumerate(model.parameters):
        results_row.append(best_val)
        label = settings["parameter_labels"][index] + "*"
        results_row_labels.append(label)

    # Define other conditions and result metrics and add to result_row for saving
    # Results.redchi is the chi2 value directly from LMFit and should match the
    # chi2 calculated in this code if the same cost function as LMFit is used
    items = [chi_sq, r_sq, results.redchi, results.success, model, chi_sq_list, solutions_norm]
    item_labels = [
        "chi_sq",
        "r_sq",
        "redchi2",
        "success",
        "model",
        "chi_sq_list",
        "Simulation results",
    ]

    for i, item in enumerate(items):
        results_row.append(item)
        results_row_labels.append(item_labels[i])

    return results_row, results_row_labels


def optimize_single_initial_guess(row: tuple) -> Tuple[list, list]:
    """Runs optimization for a single initial guess

    Parameters
    ----------
    row
        a tuple of floats containing the initial guesses for each parameter
        (represents a row of the global search results)

    Returns
    -------
    results_row
        a list of floats and lists containing the results for the given optimization run

    results_row_labels
        a list of strings defining the labels for each item in results_row"""

    initial_parameters = list(row[1 : len(settings["parameters"]) + 1])
    count = row[0] + 1
    ExperimentalData.exp_data = row[-2]
    problem = row[-1]
    free_parameter_bounds = problem['bounds']
    free_parameter_labels = problem['names']
    
    chi_sq_list = []
    def solve_for_opt(x, p_1=0, p_2=0, p_3=0, p_4=0, p_5=0, p_6=0, p_7=0, p_8=0, p_9=0, p_10=0):
        p_opt = [p_1, p_2, p_3, p_4, p_5, p_6, p_7, p_8, p_9, p_10]
        model.parameters = p_opt[: len(settings["parameters"])]
        solutions_norm, chi_sq, _ = solve_single_parameter_set()
        chi_sq_list.append(chi_sq)
        return np.array(solutions_norm)

    params_for_opt = define_parameters_for_opt(initial_parameters, free_parameter_labels, free_parameter_bounds)
    model_lmfit = Model_lmfit(solve_for_opt, nan_policy="propagate")

    if settings["weight_by_error"] == "no":
        weights_ = [1] * len(ExperimentalData.exp_error)
    else:
        weights_ = [1 / i for i in ExperimentalData.exp_error]
 
    results = model_lmfit.fit(
        ExperimentalData.exp_data,
        params_for_opt,
        method="leastsq",
        x=ExperimentalData.x,
        weights=weights_,
    )

    print("Optimization round " + str(count) + " complete.")

    results_row, results_row_labels = define_results_row(
        results, initial_parameters, chi_sq_list
    )

    return results_row, results_row_labels