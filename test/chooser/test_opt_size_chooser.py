'''
Created on 19.12.2013

@author: bartelss
'''
from ..abstract_test import AbstractTest, scale, d
import unittest
import scipy.optimize as spo
import numpy as np
import numpy.random as npr
from ...acquisition_functions.expected_improvement import ExpectedImprovement
from ...OptSizeChooser import OptSizeChooser
from ...support.hyper_parameter_sampling import sample_hyperparameters, sample_from_proposal_measure
from ...gp_model import GPModel
import scipy.linalg as spla
from time import time


class Test(AbstractTest):

    def setUp(self):
        super(Test, self).setUp()
        _hyper_samples = sample_hyperparameters(20, True, self.X, self.y, 
                                                     self.cov_func, self.noise, self.amp2, self.ls)#[100-1]]
        self._models = []
        for h in range(0, len(_hyper_samples)):
            hyper = _hyper_samples[h]
            gp = GPModel(self.X, self.y, hyper[0], hyper[1], hyper[2], hyper[3])
            self._models.append(gp)

    def xtest_minimizer2(self):
        '''
        Asserts that this function produces indeed something better than the starting point.
        '''
        opt_bounds = []# optimization bounds
        for i in xrange(0, d):
            opt_bounds.append((0, scale))

        chooser = OptSizeChooser("")
        chooser._mcmc_iters = len(self._models)
        minima = chooser._find_local_minima(self.X, self.y, self._models, None)

        #first dimension must be number of minima, second dimension the dimension of the points
        assert(minima.shape[1] == d)

        disturbed_minima = minima + npr.randn(minima.shape[0], d) / 1000
        disturbed_minima = np.clip(disturbed_minima, 0, 1)
        values = np.zeros(minima.shape[0])
        disturbed_values = np.zeros(minima.shape[0])
        gradients = np.zeros(minima.shape)
        for i in range(0, minima.shape[0]):
            values[i], gradients[i] = chooser._objective_function(minima[i], self._models, True)
            disturbed_values[i] = chooser._objective_function(disturbed_minima[i], self._models)

        #small perturbations of the minima should have higher values
        assert(np.all(values - disturbed_values < 1e-15))

        # gradients should be almost 0 except at the borders
        assert(np.all(np.abs(gradients) < 1e-5))


    def test_minimizer3(self):
        # TODO: for which reason ever the minimizer returns 1 as a minimum!
        # problematic seed: 53622
        n = 3
        X = np.array([np.linspace(1./(n+1), 1-1./(n+1), n)]).T
        y = np.ones(n)
        chooser = OptSizeChooser("")
        chooser._mcmc_iters = 1
        gp = GPModel(X, y, 2, 1e-15, 1, np.ones(1)/8)
        minima = chooser._find_local_minima(X, y, [gp], None)
        print minima
        for m in minima:
            assert(np.any(np.abs(m - X) < 1e-4))
        # print chooser._objective_function(np.array([0]), [gp], True)
        # print chooser._objective_function(np.array([1]), [gp], True)
        # print chooser._objective_function(np.array([0.5]), [gp], True)
        # starting_point =  np.array([1.])
        # opt_bounds = []# optimization bounds
        # for i in xrange(0, starting_point.shape[0]):
        #     opt_bounds.append((0, 1))
        # print spo.fmin_l_bfgs_b(chooser._objective_function, starting_point, args=([gp], True),
        #                                   bounds=opt_bounds, disp=0)

        # we have 3 minima and each should have almost same probability of being it
        assert(np.all(chooser._compute_pmin_probabilities([gp], minima) > 0.3))


    def xtest_gradient_computation(self):
        '''
        Asserts that the gradients are computed correctly.
        '''
        #problematic seed: 12233
        epsilon = 1e-6
        xstar = scale * npr.random(d)
        chooser = OptSizeChooser("")
        chooser._mcmc_iters = len(self._models)
        #get gradient in the right shape (for the test)
        gradient = np.array([np.array([chooser._objective_function(xstar, self._models, True)[1]])])
        def f(x):
            return np.array([chooser._objective_function(x[0], self._models, False)])
        self.assert_first_order_gradient_approximation(f, np.array([xstar]), gradient, epsilon)


    def xtest_candidate_preselection(self):
        print self.X.shape
        number_of_points_to_return = 10
        numer_of_acquisition_functions = 10
        chain_length = 20
        ac_funcs = []
        for i in range(0, numer_of_acquisition_functions):
            ac_funcs.append(ExpectedImprovement(self.X, self.y, self._models[i]))

        chooser = OptSizeChooser("")
        chooser._func_evals = 0
        starting_point = np.ones(d)/2

        t = time()
        chooser._sample_candidates1(ac_funcs, number_of_points_to_return, starting_point, chain_length)
        print "sum: " + str(time() - t)
        # chooser.func_evals = 0
        # t = time()
        # print chooser._sample_candidates2(ac_funcs, number_of_points_to_return, starting_point, chain_length)
        # print time() - t
        self.func_evals = 0
        t = time()
        self._sample_candidates3(ac_funcs, number_of_points_to_return, starting_point, chain_length)
        print "sequence: " + str(time() - t)
        print self.func_evals
        self.func_evals = 0
        # t = time()
        # self._sample_candidates3a(ac_funcs, number_of_points_to_return, starting_point, chain_length)
        # print time() - t
        # print self.func_evals
        # t = time()
        # self._sample_candidates4(ac_funcs, number_of_points_to_return, starting_point, chain_length)
        # print time() - t
        # t = time()
        # self._sample_candidates5(ac_funcs, number_of_points_to_return, starting_point, chain_length)
        # print time() - t
        # t = time()
        # self._sample_candidates6(ac_funcs, number_of_points_to_return, starting_point, chain_length)
        # print time() - t
        self.func_evals = 0
        t = time()
        sample_from_proposal_measure(starting_point, self._log_proposal_measure_with_cost,
                                     number_of_points_to_return, chain_length)
        print "aaron: " + str(time() - t)
        print self.func_evals

        self.func_evals = 0
        self._ei_ls = ac_funcs
        t = time()
        sample_from_proposal_measure(starting_point, self._log_proposal_measure_with_cost2,
                                     number_of_points_to_return, chain_length)
        print "aaron2: " + str(time() - t)
        print self.func_evals


    def _sample_candidates3(self, ac_funcs, number_of_points_to_return, starting_point, chain_length):
        '''
        Samples candidates from the given acquisition functions. Candidates are sampled from each function
        in turns.
        Args:
            ac_funcs: INITIALIZED acquisition functions
            number_of_points_to_return: the number of candidates to be returned
            starting_point: where to start the sampling
            chaing_length: how many samples to discard in between
        Returns:
            number_of_points_to_return candidates in a numpy matrix
        '''
        number_of_acquisition_functions = len(ac_funcs)
        sampled_candidates = np.zeros([number_of_points_to_return, starting_point.shape[0]])
        points_per_acquisition_function = number_of_points_to_return/number_of_acquisition_functions
        for i in range(0, number_of_acquisition_functions):
            self._ei = ac_funcs[i]
            sampled_candidates[i*points_per_acquisition_function:(i+1)*points_per_acquisition_function] = \
                sample_from_proposal_measure(starting_point,
                                                          self._log_proposal_measure,
                                                          points_per_acquisition_function,
                                                          chain_length)
        return sampled_candidates


    def _log_proposal_measure(self, x):
        self.func_evals+=1
        if np.any(x<0) or np.any(x>1):
            return -np.inf
        v = self._ei.compute(x)
        return np.log(v+1e-10)

    def _sample_candidates3a(self, ac_funcs, number_of_points_to_return, starting_point, chain_length):
        '''
        Samples candidates from the given acquisition functions. Candidates are sampled from each function
        in turns.
        Args:
            ac_funcs: INITIALIZED acquisition functions
            number_of_points_to_return: the number of candidates to be returned
            starting_point: where to start the sampling
            chaing_length: how many samples to discard in between
        Returns:
            number_of_points_to_return candidates in a numpy matrix
        '''
        number_of_acquisition_functions = len(ac_funcs)
        sampled_candidates = np.zeros([number_of_points_to_return, starting_point.shape[0]])
        points_per_acquisition_function = number_of_points_to_return/number_of_acquisition_functions
        for i in range(0, number_of_acquisition_functions):
            def __log_proposal_measure(x):
                self.func_evals+=1
                if np.any(x < 0) or np.any(x > 1):
                    return -np.inf
                v = ac_funcs[i].compute(x)
                return np.log(v+1e-15)
            sampled_candidates[i*points_per_acquisition_function:(i+1)*points_per_acquisition_function] = \
                sample_from_proposal_measure(starting_point,
                                                          __log_proposal_measure,
                                                          points_per_acquisition_function,
                                                          chain_length)
        return sampled_candidates

    def _sample_candidates4(self, ac_funcs, number_of_points_to_return, starting_point, chain_length):
        '''
        Samples candidates from the given acquisition functions. Candidates are sampled from each function
        in turns.
        Args:
            ac_funcs: INITIALIZED acquisition functions
            number_of_points_to_return: the number of candidates to be returned
            starting_point: where to start the sampling
            chaing_length: how many samples to discard in between
        Returns:
            number_of_points_to_return candidates in a numpy matrix
        '''
        number_of_acquisition_functions = len(ac_funcs)
        sampled_candidates = np.zeros([number_of_points_to_return, starting_point.shape[0]])
        points_per_acquisition_function = number_of_points_to_return/number_of_acquisition_functions
        for i in range(0, number_of_acquisition_functions):
            #f = lambda x: self._log_proposal_measure2(x, ac_funcs[i])
            def f(x): return self._log_proposal_measure2(x, ac_funcs[i])
            sampled_candidates[i*points_per_acquisition_function:(i+1)*points_per_acquisition_function] = \
                sample_from_proposal_measure(starting_point,
                                                          f,
                                                          points_per_acquisition_function,
                                                          chain_length)
        return sampled_candidates


    def _log_proposal_measure2(self, x, ac):
        if np.any(x<0) or np.any(x>1):
            return -np.inf
        v = ac.compute(x)
        return np.log(v+1e-10)



    def _sample_candidates5(self, ac_funcs, number_of_points_to_return, starting_point, chain_length):
        '''
        Samples candidates from the given acquisition functions. Candidates are sampled from each function
        in turns.
        Args:
            ac_funcs: INITIALIZED acquisition functions
            number_of_points_to_return: the number of candidates to be returned
            starting_point: where to start the sampling
            chaing_length: how many samples to discard in between
        Returns:
            number_of_points_to_return candidates in a numpy matrix
        '''
        number_of_acquisition_functions = len(ac_funcs)
        sampled_candidates = np.zeros([number_of_points_to_return, starting_point.shape[0]])

        for i in range(0, number_of_points_to_return):
            def f(x): return self._log_proposal_measure2(x, ac_funcs[i%number_of_acquisition_functions])
            sampled_candidates[i:i+1] = sample_from_proposal_measure(starting_point,
                                                          f,
                                                          1, chain_length)
            starting_point = sampled_candidates[i]
        return sampled_candidates

    def _sample_candidates6(self, ac_funcs, number_of_points_to_return, starting_point, chain_length):
        '''
        Samples candidates from the given acquisition functions. Candidates are sampled from each function
        in turns.
        Args:
            ac_funcs: INITIALIZED acquisition functions
            number_of_points_to_return: the number of candidates to be returned
            starting_point: where to start the sampling
            chaing_length: how many samples to discard in between
        Returns:
            number_of_points_to_return candidates in a numpy matrix
        '''
        number_of_acquisition_functions = len(ac_funcs)
        sampled_candidates = np.zeros([number_of_points_to_return, starting_point.shape[0]])
        ac_func_to_use = ac_funcs[0]
        def _log_proposal_measure3(x):
            if np.any(x<0) or np.any(x>1):
                return -np.inf
            v = ac_func_to_use.compute(x)
            return np.log(v+1e-10)

        for i in range(0, number_of_points_to_return):
            sampled_candidates[i:i+1] = sample_from_proposal_measure(starting_point,
                                                          _log_proposal_measure3,
                                                          1, chain_length)
            starting_point = sampled_candidates[i]
            ac_func_to_use = ac_funcs[i % number_of_acquisition_functions]
        return sampled_candidates

    def _log_proposal_measure_with_cost(self, x):
        self.func_evals+=1
        if np.any(x < 0) or np.any(x > 1):
            return -np.inf

        ei_value = 0
        for m in self._models:
            ei = ExpectedImprovement(self.X, self.y, m)
            ei_value += ei.compute(x) / len(self._models)

        return np.log(ei_value + 1e-50)

    def _log_proposal_measure_with_cost2(self, x):
        self.func_evals+=1
        if np.any(x < 0) or np.any(x > 1):
            return -np.inf

        ei_value = 0
        for ei in self._ei_ls:
            ei_value += ei.compute(x) / len(self._models)

        return np.log(ei_value + 1e-50)


        
    #def test_local_optimization(self):
        #TODO: implement
    #    raise NotImplementedError("to be implemented")


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()