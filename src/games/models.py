#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 09:11:06 2022

@author: kate
"""
import math
import os
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from SALib.sample import latin
import pandas as pd
from lmfit import Model as Model_lmfit
from lmfit import Parameters as Parameters_lmfit

from config import Settings, ExperimentalData, Context, Save
from analysis import Plots, CalculateMetrics

plt.style.use("./paper.mplstyle.py")

class Modules:
    
    def test_single_parameter_set():
        """Solves model for a single parameter set using dataID defined in Settings.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None 
        
        """
        sub_folder_name = 'TEST SINGLE PARAMETER SET'
        path = Save.createFolder(sub_folder_name)
        os.chdir(path)
        Save.saveConditions()
        model.parameters = Settings.parameters
        solutions_norm, chi_sq, r_sq = GeneralModel.solve_single_parameter_set()
        filename = 'FIT TO TRAINING DATA'
        Plots.plot_x_y(ExperimentalData.x, solutions_norm, ExperimentalData.exp_data, ExperimentalData.exp_error, ExperimentalData.x_label, ExperimentalData.y_label, filename, ExperimentalData.x_scale)
        return solutions_norm, chi_sq, r_sq
    
    def estimate_parameters():
        """Runs parameter estimation method (multi-start optimization)
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None 
        
        """
        sub_folder_name = 'MODULE 2 - FIT TO EXPERIMENTAL DATA'
        path = Save.createFolder(sub_folder_name)
        os.chdir(path)
        Save.saveConditions()
        
        print('Starting global search...')
        df_parameters = GlobalSearch.generate_parameter_sets(Settings.parameter_estimation_problem_definition)
        df_global_search_results = GlobalSearch.solve_global_search(df_parameters)
        print('Global search complete.')
        
        print('Starting optimization...')
        Optimization.optimize_all(df_global_search_results)
        
    def generate_parameter_estimation_method_evaluation_data():
        pass
    
    def evaluate_parameter_estimation_method():
        pass
    
    def calculate_profile_likelihood():
        pass

class Optimization(Modules):
    
    def generalize_parameter_labels():
        """Generates generalized parameter labels (p1, p2, etc.) for use in optimization functions
        
        Parameters
        ----------
        None
            
        Returns
        -------
        general_parameter_labels
            list of floats defining the generalized parameter labels 
        
        general_parameter_labels_free 
            list of floats defining which of the generalized parameter labels are free
        
        """
        general_parameter_labels = ['p' + str(i+1) for i in range(0, len(Settings.parameters))]
        general_parameter_labels_free = ['p' + str(i+1) for i in Settings.free_parameter_indicies]
        return general_parameter_labels, general_parameter_labels_free 
    
    def define_best_optimization_results(df_optimization_results):
        """Prints best parameter set from optimization to console and plots the best fit to training data
        
        Parameters
        ----------
        df_optimization_results
            df containing the results of all optimization runs
            
        Returns
        -------
        None
        
        """
        
        df = df_optimization_results
        best_case_parameters = []
        for i in range(0, len(Settings.parameters)):
            col_name = Settings.parameter_labels[i] + '*'
            val = df[col_name].iloc[0]
            best_case_parameters.append(val)
        best_case_parameters = [np.round(parameter, 5) for parameter in best_case_parameters]
        solutions_norm = df['Simulation results'].iloc[0]
        filename = 'BEST FIT TO TRAINING DATA'
        Plots.plot_x_y(ExperimentalData.x, solutions_norm, ExperimentalData.exp_data, ExperimentalData.exp_error, ExperimentalData.x_label, ExperimentalData.y_label, filename, ExperimentalData.x_scale)
   
        filename = 'COST FUNCTION TRAJECTORY'
        y = list(df['chi_sq_list'].iloc[0])
        Plots.plot_x_y(range(0, len(y)), y, 'None', 'None', 'function evaluation', 'chi_sq', filename)
   
        print('*************************')
        print('Calibrated parameters: ')
        for i, label in enumerate(Settings.parameter_labels):
            print(label + ' = ' + str(best_case_parameters[i]))
        print('')
        
        r_sq_opt = round(df['r_sq'].iloc[0], 3)
        chi_sq_opt_min = round(df['chi_sq'].iloc[0], 3)
        
        print('Metrics:')
        print('R_sq = ' + str(r_sq_opt))
        print('chi_sq = ' + str(chi_sq_opt_min))
        print('*************************')
   
 
    def optimize_all(df_global_search_results):
        """Runs optimization for each intitial guess and saves results
        
        Parameters
        ----------
        df_global_search_results
            df containing the results of the global search
            
        Returns
        -------
        None
        
        """
        df = df_global_search_results.sort_values(by=['chi_sq'])
      
        results_row_list = []
        for i, row in enumerate(df.itertuples(name = None)):
            if i < Settings.num_parameter_sets_optimization:
                initial_parameters = list(row[1:len(Settings.parameters) + 1])
                results_row, results_row_labels = Optimization.optimize_single_initial_guess(initial_parameters, i)
                results_row_list.append(results_row)
        
        print('Optimization complete.')
        df_optimization_results = pd.DataFrame(results_row_list, columns = results_row_labels)
        df_optimization_results = df_optimization_results.sort_values(by=['chi_sq'], ascending = True)
        Optimization.define_best_optimization_results(df_optimization_results)
        
        with pd.ExcelWriter('./OPTIMIZATION RESULTS.xlsx') as writer:  
            df_optimization_results.to_excel(writer, sheet_name='opt')

            
    def define_parameters_for_opt(initial_parameters):
        """Defines parameters for optimization with structure necessary for LMFit optimization code
        
        Parameters
        ----------
        initial_parameters
            a list of floats containing the initial guesses for each parameter 
            
        Returns
        -------
        params_for_opt
            an object defining the parameters and bounds for optimization (in structure necessary for LMFit optimization code)
        
        """
        general_parameter_labels, general_parameter_labels_free = Optimization.generalize_parameter_labels()
        
        #Set default values
        bounds = Settings.parameter_estimation_problem_definition['bounds']
        num_parameters = len(Settings.parameters)
        bound_min_list = [0] * num_parameters
        bound_max_list = [np.inf] * num_parameters
        vary_list = [False] * num_parameters
    
        #Set min and max bounds and vary_index by comparing free parameters lists with list of all parameters
        for param_index in range(0, len(general_parameter_labels)): #for each parameter
            for free_param_index in range(0, len(general_parameter_labels_free)): #for each free parameter
            
                #if param is free param, change vary to True and update bounds
                if general_parameter_labels[param_index] == general_parameter_labels_free[free_param_index]: 
                    vary_list[param_index] = True
                    bound_min_list[param_index] = 10 ** bounds[free_param_index][0]
                    bound_max_list[param_index] = 10 ** bounds[free_param_index][1]          
       
        #Add parameters to the parameters class
        params_for_opt = Parameters_lmfit()
        for index_param in range(0, len(general_parameter_labels)):
            params_for_opt.add(general_parameter_labels[index_param], value=initial_parameters[index_param], 
                        vary = vary_list[index_param], min = bound_min_list[index_param], 
                        max = bound_max_list[index_param])
            
        for null_index in range(len(general_parameter_labels), 10):
            params_for_opt.add('p' + str(null_index + 1), value=0, 
                        vary = False, min = 0, 
                        max = 1)
    
        return params_for_opt
    
    def define_results_row(initial_parameters, results, chi_sq_list):
        """Defines results for each optimization run
        
        Parameters
        ----------
        initial_parameters
            a list of floats containing the initial guesses for each parameter 
            
        results
            an object containing results of the given optimization run
            
        chi_sq_list
            a list of floats containing the chi_sq values for each function evaluation
            
        Returns
        -------
        results_row
            a list of floats and lists containing the results for the given optimization run
            
        results_row_labels 
            a list of strings defining the labels for each item in results_row
        
        """
        #add initial parameters to row for saving 
        results_row = []
        results_row_labels = []
        for i, val in enumerate(initial_parameters):
            results_row.append(val)
            results_row_labels.append(Settings.parameter_labels[i])
        
        #Define best fit parameters (results of optimization run)   
        best_fit_params = list(results.params.valuesdict().values())[:len(initial_parameters)]
      
        #Solve ODEs with final optimized parameters, calculate chi2, and add to result_row for saving
        model.parameters = best_fit_params
        solutions_norm, chi_sq, r_sq = GeneralModel.solve_single_parameter_set()
        
        results_row.append(chi_sq)
        results_row_labels.append('chi_sq')
        
        results_row.append(r_sq)
        results_row_labels.append('r_sq')
        
        #append best fit parameters to results_row for saving (with an * at the end of each parameter name
        #to denote that these are the optimized values)
        for index in range(0, len(best_fit_params)):
            results_row.append(best_fit_params[index])
            label = Settings.parameter_labels[index] + '*'
            results_row_labels.append(label)
        
        #Define other conditions and result metrics and add to result_row for saving
        #Note that results.redchi is the chi2 value directly from LMFit and should match the chi2
        #calculated in this code if the same cost function as LMFit is used
        items = [results.redchi, results.success, model, 
                  chi_sq_list, solutions_norm]
        item_labels = ['redchi2',  'success', 'model', 'chi_sq_list', 
                        'Simulation results']
      
        for i in range(0, len(items)):
            results_row.append(items[i])
            results_row_labels.append(item_labels[i])
      
        return results_row, results_row_labels 

    def optimize_single_initial_guess(initial_parameters, i): 
        """Runs optimization for a single initial guess
        
        Parameters
        ----------
        initial_parameters
            a list of floats containing the initial guesses for each parameter 
            
        i
            an integer defining the optimization run number 
            
        Returns
        -------
        results_row
            a list of floats and lists containing the results for the given optimization run
            
        results_row_labels 
            a list of strings defining the labels for each item in results_row """
            
        count = i + 1
        chi_sq_list = []
        
        def solve_for_opt(x, p1 = 0, p2 = 0, p3 = 0, p4 = 0, p5 = 0, p6 = 0, p7 = 0, p8 = 0, p9 = 0, p10 = 0):
            p = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]
            model.parameters = p[:len(Settings.parameters)]
            solutions_norm, chi_sq, r_sq = GeneralModel.solve_single_parameter_set()
            chi_sq_list.append(chi_sq)
            return np.array(solutions_norm)
            
        params_for_opt = Optimization.define_parameters_for_opt(initial_parameters)
        model_ = Model_lmfit(solve_for_opt, nan_policy='propagate')
        
        if Settings.weight_by_error == 'no':
            weights_ = [1] * len(ExperimentalData.exp_error)
        else:
            weights_ = [1/i for i in ExperimentalData.exp_error] 
        results = model_.fit(ExperimentalData.exp_data, params_for_opt, method = 'leastsq', x = ExperimentalData.x, weights = weights_)
        print('Optimization round ' + str(count) + ' complete.')
        
        results_row, results_row_labels = Optimization.define_results_row(initial_parameters, results, chi_sq_list)
     
        return results_row, results_row_labels
    
    
class GlobalSearch:
    def generate_parameter_sets(problem):
        """
        Generate parameter sets for global search 
        
        Parameters
        ----------
        problem
            a dictionary including the number, labels, and bounds for the free parameters 
     
            
        Returns
        -------
        df_parameters
            df with columns defining to parameter identities  and rows defining the paramter values for each set in the sweep
        """
        
      
        fit_params = problem['names'] #only the currently free parameters (as set in settings)
        num_params = problem['num_vars']
        all_params = Settings.parameter_labels #all params that are potentially free 
        n_search = Settings.num_parameter_sets_global_search
        
        #Create an empty dataframe to store results of the parameter sweep
        df_parameters = pd.DataFrame()
        
        #Fill each column of the dataframe with the intial values set in Settings. 
        for item in range(0, len(all_params)):
            param = all_params[item]
            param_array = np.full((1, n_search), Settings.parameters[item])
            param_array = param_array.tolist()
            df_parameters[param] = param_array[0]
    
        #Perform LHS
        param_values = latin.sample(problem, n_search, seed=456767)
    
        #Transform back to to linear scale
        params_converted = []
        for item in param_values:
            params_converted.append([10**(val) for val in item])
        params_converted = np.asarray(params_converted)
            
        #Replace the column of each fit parameter with the list of parameters from the sweep
        for item in range(0, num_params):   
            for name in fit_params:
                if fit_params[item] == name:
                    df_parameters[name] = params_converted[:,item]
       
        with pd.ExcelWriter('./PARAMETER SWEEP.xlsx') as writer:  
            df_parameters.to_excel(writer, sheet_name='GS parameters')
        
        return df_parameters
    
    def solve_global_search(df_parameters):
        """
        Generate parameter sets for global search 
        
        Parameters
        ----------
        df_parameters
            df with columns defining to parameter identities  and rows defining the paramter values for each set in the sweep
     
            
        Returns
        -------
        df_global_search_results
            df that contains the information in df_parameters, along with an extra column defining the cost function for each parameter set
      
        """

        chi_sq_list = []
        for row in df_parameters.itertuples(name = None):
            model.parameters = list(row[1:])
            solutions_norm, chi_sq, r_sq = GeneralModel.solve_single_parameter_set()
            chi_sq_list.append(chi_sq)
        
        df_global_search_results = df_parameters
        df_global_search_results['chi_sq'] = chi_sq_list
        
        with pd.ExcelWriter('GLOBAL SEARCH RESULTS.xlsx') as writer: 
             df_global_search_results.to_excel(writer, sheet_name='GS results')
        
        return df_global_search_results

class GeneralModel:
    """
    General methods for any ODE model
    

    """
    def solve_single_parameter_set():
        """
        Solves model for a single parameter set
        
        Parameters
        ----------
        None
     
        Returns
        -------
        solutions_norm
            a list of floats containing the normalized simulation values corresponding to the dataID defined in Settings
        
        chi_sq
            a float defining the value of the cost function
        
        r_sq 
            a float defining the value of the correlation coefficient (r_sq)
        """
        
        if Settings.modelID == 'synTF_chem':
            if Settings.dataID == 'ligand dose response':
                solutions = model.solve_ligand_sweep(ExperimentalData.x)
               
        elif Settings.modelID == 'synTF':
            if Settings.dataID == 'synTF dose response':
                solutions = model.solve_synTF_sweep(ExperimentalData.x)
                
        solutions_norm = [i/max(solutions) for i in solutions]
        chi_sq = CalculateMetrics.calc_chi_sq(ExperimentalData.exp_data, solutions_norm, ExperimentalData.exp_error)
        r_sq = CalculateMetrics.calc_r_sq(ExperimentalData.exp_data, solutions_norm)
        
        return solutions_norm, chi_sq, r_sq 
        

    def set_general_parameters(self):
        """Defines general parameters that may be used in any model.
  
        Parameters
        ----------
        None 
        
        Returns
        -------
        None
        
        """
        k_txn = 1
        k_trans = 1
        kdeg_rna = 2.7
        kdeg_protein = 0.35
        kdeg_reporter = 0.029
        k_deg_ligand = 0.01
        self.general_parameters = np.array(
            [k_txn, k_trans, kdeg_rna, kdeg_protein, kdeg_reporter, k_deg_ligand]
        )
     

class synTF_chem(GeneralModel, Modules):
    """
    Representation of synTF_Chem model - synTFChem

    """

    def __init__(self, parameters = [1, 1, 1, 1, 1, 1], inputs = [50, 50], input_ligand = 1000):
        """Initializes synTF_Chem model.
        
        Parameters
        ----------
        parameters
            List of floats defining the parame ters
        
        inputs
            List of floats defining the inputs 
        
        input_ligand
            Float defining the input ligand concentration
            
        Returns
        -------
        None 
        
        """
        self.number_of_states = 8
        
        self.free_parameters = np.array(parameters)
        self.inputs = np.array(inputs)
        self.input_ligand = input_ligand
        
        GeneralModel.set_general_parameters(self)
        
        y_init = np.zeros(self.number_of_states)
        self.initial_conditions = y_init

    def solve_single(self):
        """Solves synTF_Chem model for a single set of parameters and inputs, including 2 steps
           1) Time from transfection to ligand addition
           2) Time from ligand addition to measurement via flow cytometry
        
        Parameters
        ----------
        None
            
        Returns
        -------
        solution
            An array of ODE solutions (rows are timepoints and columns are model states)
            
        t
            A 1D array of time values corresponding to the rows in solution
        
        """
        # solve before ligand addition
        timesteps = 100
        end_time = 18
        tspace_before_ligand_addition = np.linspace(0, end_time, timesteps)
        self.t = tspace_before_ligand_addition
        solution_before_ligand_addition = odeint(
            self.gradient,
            self.initial_conditions,
            self.t,
            args=(
                self.parameters,
                self.inputs,
                self.general_parameters,
            ),
        )

        # solve after ligand addition
        for i, label in enumerate(Settings.parameter_labels):
            if label == 'e':
                input_ligand_transformed = self.input_ligand * model.parameters[i]
                
        end_time = 24
        tspace_after_ligand_addition = np.linspace(0, end_time, timesteps)
        self.t = tspace_after_ligand_addition
        initial_conditions_after_ligand_addition = np.array(solution_before_ligand_addition[-1, :])
        initial_conditions_after_ligand_addition[4] = self.input_ligand
        self.initial_conditions_after_ligand_addition = initial_conditions_after_ligand_addition
        solution_after_ligand_addition = odeint(
            self.gradient,
            self.initial_conditions_after_ligand_addition,
            self.t,
            args=(
                self.parameters,
                self.inputs,
                self.general_parameters,
            ),
        )
        self.solution = solution_after_ligand_addition

        return self.solution, self.t

    def gradient(self, y, t, parameters, inputs, general_parameters):
        """Defines the gradient for synTF_Chem model.
        
        Parameters
        ----------
        parameters
            List of floats defining the parameters
        
        inputs
            List of floats defining the inputs 
            
        general_parameters
            List of floats defining the general parameters that may be used
                k_txn, k_trans, kdeg_rna, kdeg_protein, kdeg_reporter, k_deg_ligand
        
            
        Returns
        -------
        dydt
            An list of floats corresponding to the gradient of each model state at time t
        
        """
        y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8 = y
        [e, b, k_bind, m, km, n] = parameters
        [dose_a, dose_b] = inputs
        k_txn, k_trans, kdeg_rna, kdeg_protein, kdeg_reporter, k_deg_ligand = general_parameters

        f_num = b + m * (y_6 / km) ** n
        f_denom = 1 + (y_6 / km) ** n + (y_2 / km) ** n
        f = f_num / f_denom

        if math.isnan(f):
            f = 0

        u_1 = k_txn * dose_a - kdeg_rna * y_1  # y1 A mRNA
        u_2 = k_trans * y_1 - kdeg_protein * y_2 - k_bind * y_2 * y_4 * y_5  # y2 A protein
        u_3 = k_txn * dose_b - kdeg_rna * y_3  # y3 B mRNA
        u_4 = k_trans * y_3 - kdeg_protein * y_4 - k_bind * y_2 * y_4 * y_5  # y4 B protein
        u_5 = -k_bind * y_2 * y_4 * y_5 - y_5 * k_deg_ligand  # y5 Ligand
        u_6 = k_bind * y_2 * y_4 * y_5 - kdeg_protein * y_6  # y6 Activator
        u_7 = k_txn * f - kdeg_rna * y_7  # y7 Reporter mRNA
        u_8 = k_trans * y_7 - kdeg_reporter * y_8  # y8 Reporter protein
        dydt = np.array([u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8])

        return dydt
    
    def solve_ligand_sweep(self, x_ligand):
        """Solve synTF_Chem model for a list of ligand values.
        
        Parameters
        ----------
        x_ligand
            A list of integers containing the ligand amounts to sweep over
        
            
        Returns
        -------
        solutions
            A list of floats containing the value of the reporter protein 
            at the final timepoint for each ligand amount
        
        """
        
        solutions = []
        for ligand in x_ligand:
            self.input_ligand = ligand * model.parameters[0]
            sol, t = model.solve_single()
            solutions.append(sol[-1, -1])
            
        return solutions    


class synTF(GeneralModel, Modules):
    """
    Representation of synTF model - synTF only

    """

    def __init__(self, parameters = [1, 1], inputs = [50]):
        """Initializes synTF model.
        
        Parameters
        ----------
        free_parameters
            List of floats defining the free parameters
        
        inputs
            List of floats defining the inputs 
            
        Returns
        -------
        None 
        
        """
     
        self.number_of_states = 4
        self.parameters = np.array(parameters)
        self.inputs = np.array(inputs)
        GeneralModel.set_general_parameters(self)
        y_init = np.zeros(self.number_of_states)
        self.initial_conditions = y_init
  
    def solve_single(self):
        """Solves synTF model for a single set of parameters and inputs
        
        Parameters
        ----------
        None
            
        Returns
        -------
        solution
            An array of ODE solutions (rows are timepoints and columns are model states)
            
        t
            A 1D array of time values corresponding to the rows in solution
            
        """
        
        timesteps = 100
        end_time = 42
        tspace = np.linspace(0, end_time, timesteps)
        self.t = tspace
        self.solution = odeint(
            self.gradient,
            self.initial_conditions,
            self.t,
            args=(
                self.parameters,
                self.inputs,
                self.general_parameters,
            ),
        )

        return self.solution, self.t

    def gradient(self, y, t, parameters, inputs, general_parameters):
        """Defines the gradient for synTF model.
        
        Parameters
        ----------
        parameters
            List of floats defining the parameters
        
        inputs
            List of floats defining the inputs 
            
        general_parameters
            List of floats defining the general parameters that may be used
                k_txn, k_trans, kdeg_rna, kdeg_protein, kdeg_reporter, k_deg_ligand
        
            
        Returns
        -------
        dydt
            An list of floats corresponding to the gradient of each model state at time t
        
        """
        y_1, y_2, y_3, y_4 = y
        [g, a] = parameters
        [dose_a] = inputs

        k_txn, k_trans, kdeg_rna, kdeg_protein, kdeg_reporter, k_deg_ligand = general_parameters
        
        u_1 = k_txn * dose_a - kdeg_rna * y_1  # y1 synTF mRNA
        u_2 = k_trans * y_1 - kdeg_protein * y_2  # y2 synTF protein
        u_3 = k_txn * g * y_2 - kdeg_rna * y_3  # y3 Reporter mRNA
        u_4 = k_trans * y_3 - kdeg_reporter * y_4  # y4 Reporter protein
        dydt = np.array([u_1, u_2, u_3, u_4])

        return dydt
    
    def solve_synTF_sweep(self, x):
        """Solve synTF model for a list of synTF values.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        synTF_amounts
            A list of integers containing the synTF amounts to sweep over
        
        solutions
            A list of floats containing the value of the reporter protein 
            at the final timepoint for each synTF amount
        
        """
        solutions = []
        for synTF_amount in x:
            self.inputs = [synTF_amount]
            sol, t = model.solve_single()
            solutions.append(sol[-1, -1])
            
        return solutions
   
def Set_Model():
    """
    Defines the model class to use depending on the modelID defined in Settings
    
    Parameters
    ----------
    None
 
    Returns
    -------
    model
        object defining the model
    """
    if Settings.modelID == 'synTF_chem':
        model = synTF_chem(parameters = Settings.parameters)
       
    elif Settings.modelID == 'synTF':
        model = synTF(parameters = Settings.parameters)
        
    return model

model = Set_Model()


