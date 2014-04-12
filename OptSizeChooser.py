'''
Created on 18.11.2013

@author: Aaron Klein, Simon Bartels
'''
import numpy as np
import cPickle
import os.path as osp
import os
import scipy.optimize as spo
import spearmint.util as util
from .gp_model import GPModel, getNumberOfParameters, fetchKernel
from .acquisition_functions.expected_improvement import ExpectedImprovement
from .acquisition_functions.entropy_search import EntropySearch, compute_pmin_bins, fetch_acquisition_function

from multiprocessing import Pool
from support.hyper_parameter_sampling import sample_hyperparameters, sample_from_proposal_measure
import traceback
from spearmint.helpers import log

def init(expt_dir, arg_string):
    args = util.unpack_args(arg_string)
    return OptSizeChooser(expt_dir, **args)

def _apply_functions_asynchronously(funcs, cand, pool_size):
    '''
    Applies the given functions over all candidates in parallel.
    Calls _iterate_over_candidates for that purpose.
    Returns the average.
    Args:
        funcs: a list of function objects that have a #compute(x, gradient=True) method
        cand: a list of candidates to iterate over
        pool_size: the number of worker threads to use
    Returns:
        a numpy vector of values, one entry for each entry in cand
    '''

    #Prepare multiprocessing
    log("Employing " + str(pool_size)  + " threads to compute acquisition function value for "
        + str(cand.shape[0]) + " candidates.")
    pool = Pool(pool_size)
    results = []
    #Create a GP for each hyper-parameter sample
    for f in funcs:
        #Iterate over all candidates for each GP in parallel
        #apparently the last , needs to be there
        results.append(pool.apply_async(_iterate_over_candidates, (f, cand,)))
    pool.close()
    pool.join()

    number_of_functions = len(funcs)
    overall_value = np.zeros(len(cand))
    #get results
    for res in results:
        res.wait()
        try:
            ac_vals = res.get()
            overall_value += ac_vals/number_of_functions
        except Exception, ex:
            log("Worker Thread reported exception:" + str(ex) + "\n Action: ignoring result")
    return overall_value

def _iterate_over_candidates(func, cand):
    '''
    Is called by #_apply_functions_asynchronously. Multiprocessing can only call functions that
    are pickable. Thus these method must be at the top level of a class. This is the workaround.
    Args:
        func: the function object
        cand: the set of candidate points
    Returns:
        an array of function values (for each candidate one value)
    '''
    try:
        #This is what we want to distribute over different processes
        #We'll iterate over all candidates for each GP in a different thread
        values = np.zeros(len(cand))
        #Iterate over all candidates
        for i in xrange(0, len(cand)):
            values[i] = func.compute(cand[i])
        return values
    #This is to make the multi-process debugging easier
    except Exception, e: 
        print traceback.format_exc()
        raise e
    
def _call_minimizer(cand, func, arguments, opt_bounds):
    '''
    This function is also desgined to be called in parallel. It calls a minmizer with the given argument.
    Args:
        cand: the starting point for the minimizer
        func: the function to be minimized
        arguments: for func
        opt_bounds: the optimization bounds
    Returns:
        a triple consisting of the point, the value and the gradients
        
    '''
    try:
        return spo.fmin_l_bfgs_b(func, cand.flatten(), args=arguments, bounds=opt_bounds, disp=0)
    except Exception, e: 
        print traceback.format_exc()
        raise e

def _compute_average_gradient(x, functions):
    '''
    Computes value and gradient of all functions in the list for one candidate.
    The purpose of this function is to be called with a minimizer.
    Args:
        x: the candidate
        functions: a list of function objects that have a #compute(x, gradient=True) method
    Returns:
        a tuple (sum f(x)/len(functions), -sum grad f(x) /len(functions))
    '''
    number_of_functions = len(functions)
    grad_sum = np.zeros(x.shape).flatten()
    val_sum = 0
    for f in functions:
        (val, grad) = f.compute(x, True)
        val_sum += val/number_of_functions
        grad_sum += grad.flatten()/number_of_functions
    return (val_sum, grad_sum)

class OptSizeChooser(object):
    def __init__(self, expt_dir, covar='Matern52', cost_covar='Polynomial3', mcmc_iters=10,
                 pending_samples=100, noiseless=False, burnin=100,
                 grid_subset=20, acquisition_function_index=0,
                 model_costs=True, project_candidates=True, seed=None,
                 pool_size=10, do_visualization=False, incumbent_inter_sample_distance=20,
                 incumbent_number_of_minima=10, number_of_pmin_samples=1000):
        #TODO: use arguments!
        '''
        Constructor
        '''
        self.pending_samples = pending_samples
        self.grid_subset = grid_subset
        self.expt_dir = expt_dir
        self._hyper_samples = []
        self._cost_function_hyper_parameter_samples = []
        self._is_initialized = False
        #path where to store the state of the chooser
        self._state_path = os.path.join(self.expt_dir, 'chooser_state.pkl')

        self._pool_size = int(pool_size)
        self._mcmc_iters = int(mcmc_iters)
        self._noiseless = bool(noiseless)
        self._burnin = int(burnin)
        self._covar = covar
        self._cov_func, self._covar_derivative = fetchKernel(covar)
        self._cost_covar = cost_covar
        self._cost_cov_func, _ = fetchKernel(cost_covar)
        self._incumbent_inter_sample_distance = int(incumbent_inter_sample_distance)
        self._incumbent_number_of_minima = int(incumbent_number_of_minima)
        self._number_of_pmin_samples = number_of_pmin_samples
        #TODO: if false check that acquisition function can handle that
        self._model_costs = bool(model_costs)
        self._ac_func = fetch_acquisition_function(int(acquisition_function_index))


        self._do_visualization = do_visualization
        #FIXME: not working yet
        self._project_candidates = False #bool(project_candidates)

        if seed is not None:
            np.random.seed(int(seed))

    def _recover(self, dims, comp, values, durations):
        '''
        Performs some more initialization with the first call of next().
        Checks if the chooser crashed and restores the last parameters in that case.
        Calls #_burn_in if necessary.
        Args:
            dims: the dimension of the objective function
            comp: the points that have been evaluated so far
            values: the values that have been observed so far
            durations: the time it took to compute the points
        Returns:
            the seed that is used
        '''
        if self._is_initialized:
            seed = np.random.randint(65000)
        else:
            self._is_initialized = True
            self._general_initialization(dims, comp, values, durations)

            do_recover = osp.isfile(self._state_path)
            #if the file exists we have to make a recovery!
            if do_recover:
                seed = self._read_parameters_from_file()
            else:
                seed = np.random.randint(65000)
                self._burn_in(dims, comp, values, durations)
        self._write_parameters_to_file(seed)
        log("using seed: " + str(seed))
        np.random.seed(seed)
        return seed

    def _general_initialization(self, dims, comp, values, durations):
        '''
        Performs some general initialization that is necessary with the first call of #next.
        However, sets only those variables that are deterministic.
        Args:
            dims: the dimension of the objective function
            comp: the points that have been evaluated so far
            values: the values that have been observed so far
            durations: the time it took to compute the points
        '''
        #set visualizer
        if self._do_visualization:
            from ..entropy_search_chooser.Visualizer import Visualizer
            vis = Visualizer(comp.shape[0] - 2, self.expt_dir)
            if dims == 2:
               self._visualize = vis.plot3D
            else:
                #do nothing
                self._visualize = lambda *args: args
        #set optimization bounds for optimizer
        self._opt_bounds = []
        if self._project_candidates:
            #we need one bound less if we project the candidates
            dims = dims - 1
        for i in xrange(0, dims):
            self._opt_bounds.append((0, 1))
        self._init_trace_file(dims)

    def _write_parameters_to_file(self, seed):
        '''
        Writes the chooser's state in to a pickle file.
        Args:
            seed: the seed
        '''
        fh = open(self._state_path, 'w')
        cPickle.dump({'seed': seed,
                      'hyper_samples': self._hyper_samples,
                      'cost_hyper_samples': self._cost_function_hyper_parameter_samples},
                     fh)
        fh.close()

    def _read_parameters_from_file(self):
        '''
        Reads the chooser's state from a pickle file.
        Returns:
            the seed
        '''
        fh = open(self._state_path, 'r')
        state = cPickle.load(fh)
        fh.close()
        #this is not very robust but should be sufficient for our purposes
        os.remove(self._state_path)
        self._hyper_samples = state['hyper_samples']
        self._cost_function_hyper_parameter_samples = state['cost_hyper_samples']
        return state['seed']

    def _burn_in(self, dims, comp, values, durations):
        '''
        Performs a hyper parameter burn in for the Gaussian processes. Called by #_recover if necessary.
        Args:
            dims: the dimension of the objective function
            comp: the points that have been evaluated so far
            values: the values that have been observed so far
            durations: the time it took to compute the points
        '''
        # Initial length scales.
        ls = np.ones(getNumberOfParameters(self._covar, dims))

        # the number of candidates that are preselected before applying the actual acquisition function
        self._number_of_preselected_candidates = 10 * dims * self._mcmc_iters

        # Initial amplitude.
        amp2 = np.std(values)+1e-4

        # Initial observation noise.
        noise = 1e-3

        #burn in
        self._hyper_samples = sample_hyperparameters(self._burnin, self._noiseless, comp, values, self._cov_func,
                                                     noise, amp2, ls)

        if self._model_costs:
            amp2 = np.std(durations)+1e-4
            ls = np.ones(getNumberOfParameters(self._covar, dims))
            #burn in for the cost models
            self._cost_function_hyper_parameter_samples = sample_hyperparameters(self._burnin, self._noiseless,
                                                                                 comp, durations,
                                                                                 self._cost_cov_func, noise, amp2,
                                                                                 ls)

    def next(self, grid, values, durations, candidates, pending, complete):
        comp = grid[complete,:]
        if comp.shape[0] < 2:
            c = grid[candidates[0]]
            log("Evaluating: " + str(c))
            return candidates[0]
            #return (len(candidates)+1, c)
                
        vals = values[complete]
        dimension = comp.shape[1]
        durs = durations[complete]

        #check if the chooser crashed in between
        seed = self._recover(dimension, comp, vals, durs)

        #initialize Gaussian processes
        (models, cost_models) = self._initialize_models(comp, vals, durs)

        cand = grid[candidates,:]

        local_minima = self._find_average_local_minima(self._create_function_objects(models), cand)
        if local_minima.shape[0] > 1:
            pmin = self._compute_pmin_probabilities(models, local_minima)
            incumbent = local_minima[np.argmax(pmin)]
        else:
            pmin = np.ones(1)
            incumbent = local_minima[0]
        log("Current suspected location of the minimum: " + str(incumbent))

        cand = np.vstack((cand, local_minima))

        ac_funcs = self._initialize_acquisition_functions(self._ac_func, comp, vals, incumbent, models, cost_models)
        #overall results of the acquisition functions for each candidate over all models
        overall_ac_value = _apply_functions_asynchronously(ac_funcs, cand, self._pool_size)
            
        best_cand = np.argmax(overall_ac_value)

        #do visualization
        if self._do_visualization:
            log('Visualizing...')
            #should the Visualizer have any troubles we don't want to be affected
            try:
                self._visualize(comp, vals, models[0],
                               cost_models[0],
                               cand[best_cand],
                               cand, 100, overall_ac_value, incumbent)
            except Exception, e:
                log('Visualizer crashed: ' + traceback.format_exc())


        log("Evaluating: " + str(cand[best_cand]))
        self._write_trace(seed, incumbent, best_cand, cand[best_cand])
        if(best_cand >= len(candidates)):
            # the chosen candidate is not among the grid candidates
            return (len(candidates) + 1, cand[best_cand])
        return int(candidates[best_cand])

    def _write_trace(self, seed, incumbent, candidate_index, candidate):
        '''
        Writes
        '''
        try:
            #write incumbent to file
            incumbent_fh = open(os.path.join(self.expt_dir, 'incumbent.csv'), 'a')
            incumbent_as_string = ''
            for d in range(0, incumbent.shape[0]):
                incumbent_as_string+=str(incumbent[d]) + ','
            incumbent_as_string+=str(candidate_index) + ","
            for d in range(0, candidate.shape[0]):
                incumbent_as_string+=str(candidate[d]) + ','
            incumbent_as_string+=str(seed) + ","
            incumbent_as_string+=str(self._hyper_samples[len(self._hyper_samples)-1]) + ","
            incumbent_as_string+=str(self._cost_function_hyper_parameter_samples[len(self._hyper_samples)-1])
            incumbent_fh.write("%s\n"
                           % (incumbent_as_string))
            incumbent_fh.close()
        except Exception, e:
            log("WARNING: Could not write incumbent.csv!" + traceback.format_exc())


    def _init_trace_file(self, dims):
        try:
            incumbent_fh = open(os.path.join(self.expt_dir, 'incumbent.csv'), 'a')
            output_string = "incumbent,"
            for d in range(1, dims):
                output_string+=','
            output_string+="job index,"
            output_string+="chosen candidate,"
            for d in range(1, dims):
                output_string+=','
            output_string+="seed,"
            output_string+="mean,"
            output_string+="noise,"
            output_string+="amplitude,"
            output_string+="ls,"
            for d in range(1, dims):
                output_string+=','
            output_string+="mean,"
            output_string+="noise,"
            output_string+="amplitude,"
            output_string+="ls,"
            for d in range(1, dims):
                output_string+=','
            incumbent_fh.write("%s\n"
                           % (output_string))
            incumbent_fh.close()
        except Exception, e:
            log("WARNING: Could not initialize incumbent.csv!")
        
    def _initialize_models(self, comp, vals, durs):
        '''
        Initializes the models of the objective function and if required the models for the cost functions.
        Args:
            comp: the points where the objective function has been evaluated so far
            vals: the corresponding observed values
            durs: the time it took to compute the values
        Returns:
            a tuple of two lists. The first list is a list of Gaussian process models for the objective function.
            The second list is empty if self._model_costs is false. Otherwise it is a list of Gaussian processes
            that model the costs for evaluating the objective function. In this case the lists are of equal length.
        '''
        #Slice sampling of hyper parameters
        #Get last sampled hyper-parameters
        (_, noise, amp2, ls) = self._hyper_samples[len(self._hyper_samples)-1]
        log("last hyper parameters: " + str(self._hyper_samples[len(self._hyper_samples)-1]))
        self._hyper_samples = sample_hyperparameters(self._mcmc_iters, self._noiseless, comp, vals, 
                                                     self._cov_func, noise, amp2, ls)
        
        if self._model_costs:
                (_, noise, amp2, ls) = self._cost_function_hyper_parameter_samples[len(self._cost_function_hyper_parameter_samples)-1]
                self._cost_function_hyper_parameter_samples = sample_hyperparameters(self._mcmc_iters, 
                                                                                     self._noiseless, comp, durs, 
                                                                                     self._cost_cov_func, noise, 
                                                                                     amp2, ls)

        models = []
        cost_models = []
        for h in range(0, len(self._hyper_samples)):
            hyper = self._hyper_samples[h]
            gp = GPModel(comp, vals, hyper[0], hyper[1], hyper[2], hyper[3], self._covar)
            models.append(gp)
            if self._model_costs:
                cost_hyper = self._cost_function_hyper_parameter_samples[h]
                cost_gp = GPModel(comp, durs, cost_hyper[0], cost_hyper[1], cost_hyper[2], cost_hyper[3], self._cost_covar)
                cost_models.append(cost_gp)
        return (models, cost_models)
    
    def _initialize_acquisition_functions(self, ac_func, comp, vals, incumbent, models, cost_models):
        '''
        Initializes an acquisition function for each model.
        Args:
            ac_func: an (UNINITIALIZED) acquisition function
            comp: the points where the objective function has been evaluated so far
            vals: the corresponding observed values
            incumbent: current best
            models: a list of models
            cost_models: OPTIONAL. will only be used if the length of the list is equal to the length of the list of models
         Returns:
             a list of initialized acquisition functions
        '''
        cost_model = None
        ac_funcs = []
        for i in range(0, len(models)):
            if self._model_costs:
                cost_model = cost_models[i]
            ac_funcs.append(ac_func(comp, vals, incumbent, models[i], cost_model))
        return ac_funcs

    def _compute_pmin_probabilities(self, model_list, candidates):
        '''
        Computes the probability for each candidate to be the minimum. Ideally candidates was computed with
        #_find_local_minima.
        Args:
            model_list: a list of Gaussian process models
            candidates: a numpy matrix of candidates
        Returns:
            a numpy vector containing for each candidate the probability to be the minimum
        '''
        pmin = np.zeros(candidates.shape[0])
        number_of_models = len(model_list)
        for model in model_list:
            m, L = model.getCholeskyForJointSample(candidates)
            #use different Omega for different GPs
            Omega = np.random.normal(0, 1, (self._number_of_pmin_samples, candidates.shape[0]))
            pmin += compute_pmin_bins(m, L, Omega) / number_of_models
        return pmin

    def _find_average_local_minima(self, function_list, candidates):
        '''
        Tries to find as many local minima as possible of the given list of functions.
        Args:
            function_list: a list of function objects with a method #compute(x, bool)
                where x is a numpy vector. Must return a value if bool is false and otherwise
                a tupel with function value and gradient.
            candidates: a numpy matrix of candidates
        Returns:
            a numpy matrix of local minima
        '''
        number_of_functions = len(function_list)
        if self._project_candidates:
            #ignore first coordinate
            candidates = np.copy(candidates[:, 1:])
        values = _apply_functions_asynchronously(function_list, candidates, self._pool_size)
        starting_point = candidates[np.argmin(values)]
        # sample points from our proposal measure as starting points for the minimizer
        def objective_function(x):
            val = 0
            if np.any(x < 0) or np.any(x > 1):
                return -np.infty
            for i in range(0, number_of_functions):
                val += function_list[i].compute(x)/number_of_functions
            return val
        sampled_points = sample_from_proposal_measure(starting_point, objective_function,
                                     self._incumbent_number_of_minima, self._incumbent_inter_sample_distance)
        sampled_points= np.vstack((sampled_points, starting_point))
        minima = []
        log("Employing " + str(self._pool_size)  + " threads for local optimization of candidates.")
        pool = Pool(self._pool_size)
        #call minimizer in parallel
        results = [pool.apply_async(_call_minimizer, (sampled_points[i], _compute_average_gradient,
                                                      (function_list, ), self._opt_bounds))
                   for i in range(0, sampled_points.shape[0])]
        pool.close()
        pool.join()

        #fetch results
        for i in range(0, sampled_points.shape[0]):
            res = results[i]
            res.wait()
            try:
                r = res.get()
                point = r[0]
                value = r[1]
                #consistency check
                if value > objective_function(sampled_points[i]):
                    #happens sometimes, something with the Hessian not being positive definite
                    log('WARNING: Result of optimizer worse than initial guess! Ignoring result.')
                    point = sampled_points[i]

                #remove duplicates
                append = True
                for j in range(0, len(minima)):
                    if np.allclose(minima[j], point):
                        append = False
                        break
                if append:
                    if self._project_candidates:
                        #set first coordinate
                        point = np.insert(point, 0, 1)
                    minima.append(point)
            except Exception, ex:
                log("Worker Thread reported exception:" + str(ex) + "\n Action: ignoring result")
        return np.array(minima)

    def _create_function_objects(self, model_list):
        '''
        Args:
            model_list: a list of models
        '''
        function_objects = []
        if self._project_candidates:
            for m in model_list:
                function_objects.append(ProjectedObjectiveFunctionObject(m))
        else:
            for m in model_list:
                function_objects.append(ObjectiveFunctionObject(m))
        return function_objects

class ObjectiveFunctionObject(object):
    def __init__(this, model):
        this._model = model

    def compute(this, x, gradients=False):
        '''
        Computes the sum of mean and standard deviation of a Gaussian process in x.
        Args:
            x: a numpy vector (not a matrix)
            gradients: whether to compute the gradients
        Returns:
            the value or if gradients is True additionally a numpy vector containing the gradients
        '''
        x = np.array([x])
        return _objective_function(this._model, x, gradients)


class ProjectedObjectiveFunctionObject(object):
    def __init__(this, model):
        this._model = model

    def compute(this, x, gradients=False):
        '''
        Computes the sum of mean and standard deviation of a Gaussian process in x.
        Args:
            x: a numpy vector (not a matrix)
            gradients: whether to compute the gradients
        Returns:
            the value or if gradients is True additionally a numpy vector containing the gradients
        '''
        #add first coordinate and set it to 1
        x = np.vstack((np.ones([1, 1]), x[:, np.newaxis]))
        if gradients:
            v, g = _objective_function(this._model, x, gradients)
            #gradient is of the form [[x0,x1,...]]
            #we don't need the first
            return (v, np.array([g[0][1:]]))
        #else
        return _objective_function(this._model, x, False)

def _objective_function(model, x, gradients=False):
    mean, std = model.predict(x, True)
    #mean and std don't have the right form
    mean = mean[0]
    std = std[0]

    #take square root to get standard deviation
    std = np.sqrt(std)
    if not gradients:
        return mean + std
    mg, vg = model.getGradients(x[0])
    #getGradient returns the gradient of the variance - to get the gradients of the standard deviation
    # we need to apply chain rule (s(x)=sqrt[v(x)] => s'(x) = 1/2 * v'(x) / sqrt[v(x)]
    stdg = 0.5 * vg / std
    return (mean + std, mg + stdg)