'''
Created on 02.12.2013

@author: Aaron Klein, Simon Bartels

This class extends EntropySearch to be used for big data sets. I.e. it incorporates the cost models
and it ASSUMES that the first value of a point is the data set size.
'''
import numpy as np
from .entropy_search import EntropySearch, \
    NUMBER_OF_REPRESENTER_POINTS, compute_pmin_bins
from ..support.hyper_parameter_sampling import sample_from_proposal_measure

class EntropySearchBigData(EntropySearch):

    def __init__(self):
        raise NotImplementedError("EntropySearchBigData is an abstract class!")

    def _initialize(self, comp, vals, incumbent, gp, cost_gp=None):
        '''
        Default constructor.
        '''
        super(EntropySearchBigData, self)._general_initialization(comp, vals, gp, cost_gp)
        starting_point = incumbent[1:] #we don't want to slice sample over the first value
        representers = sample_from_proposal_measure(starting_point, self._sample_measure,
                                                                           NUMBER_OF_REPRESENTER_POINTS)
        self._func_sample_locations = np.empty([NUMBER_OF_REPRESENTER_POINTS, comp.shape[1]])
        #the representers miss the first coordinate: we need to add it here.
        for i in range(0, NUMBER_OF_REPRESENTER_POINTS):
            self._func_sample_locations[i] = np.insert(representers[i], 0, 1) #set first value to one
        super(EntropySearchBigData, self)._initialization_after_sampling_representers()

        #TODO: in future this is probbaly already available in the super class
        mean, cholesky = self._gp.getCholeskyForJointSample(self._func_sample_locations)
        current_pmin = compute_pmin_bins(mean, cholesky, self._Omega)
        #inverse Kullback-Leibler divergence
        self._current_kl = np.dot(current_pmin, np.log(current_pmin+1e-50)) \
                           + np.dot(current_pmin, self._log_proposal_vals)
    
    def _sample_measure(self, x):
        if np.any(x<0) or np.any(x>1):
            return -np.inf
        #set first value to one     
        x = np.insert(x, 0, 1)
        return self._log_proposal_measure(x)

    def compute(self, candidate, compute_gradient = False):
        kl = super(EntropySearchBigData, self).compute(candidate, compute_gradient)
        #take maximum of costs and e to avoid numerical problems if the child class divides by costs or log costs
        costs = np.max([1e1, self._cost_gp.predict(np.array([candidate]))])
        #abstract method defined in child classes (see bigdata package)
        return self._compute(kl, costs, candidate)