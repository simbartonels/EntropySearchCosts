'''
Created on 02.12.2013

@author: Aaron Klein, Simon Bartels

This class implements an acquisition function similar to what is proposed in 
"Entropy Search for Information-Efficient Global Optimization" by Hennig and Schuler in 2013.
Instead of using Expectation Propagation we do Monte Carlo integration.
'''
import numpy as np
from ..acquisition_functions.expected_improvement import ExpectedImprovement
from ..support.hyper_parameter_sampling import sample_from_proposal_measure
from scipy.stats import norm
from spearmint.sobol_lib import i4_sobol_generate
import scipy.linalg as spla

'''
The number of points used to represent/discretize Pmin (without the candidate).
'''
NUMBER_OF_REPRESENTER_POINTS = 20

'''
The number of independent joint samples drawn for the representer points.
'''
NUMBER_OF_PMIN_SAMPLES = 600

'''
The number of samples per candidate.
'''
NUMBER_OF_CANDIDATE_SAMPLES = 21

def compute_pmin_bins(mean, L, Omega):
    #TODO: deprecated! Update with implementation in compute_fast
    '''
    Computes a discrete probability measure where the minimum is in a given Gaussian process at certain locations.
    The Gaussian process is given in terms of mean and Cholesky decomposition of the covariance matrix at these
    locations.
    Args:
        mean: the mean prediction for the considered locations (numpy vector)
        L: the Cholesky decomposition of the covariance matrix (numpy matrix)
        Omega: a numpy matrix of samples with shape [#samples, #locations]
    Returns:
        a numpy array with a probability for each considered location
    '''
    number_of_samples = Omega.shape[0]
    Y = mean[:, np.newaxis] + np.dot(L, Omega.T)
    min_idx = np.argmin(Y, axis = 0)
    mins = np.zeros([mean.shape[0], number_of_samples])
    mins[min_idx, np.arange(0, number_of_samples)] = 1
    pmin = np.sum(mins, axis = 1)
    pmin = 1./ number_of_samples * pmin
    return pmin

def fetch_acquisition_function(index):
    from .big_data.entropy_search_big_data1 import EntropySearchBigData1
    from .big_data.entropy_search_big_data2 import EntropySearchBigData2
    from .big_data.entropy_search_big_data3 import EntropySearchBigData3
    from .big_data.entropy_search_big_data4 import EntropySearchBigData4
    from .big_data.entropy_search_big_data5 import EntropySearchBigData5
    from .big_data.entropy_search_big_data6 import EntropySearchBigData6
    from .big_data.entropy_search_big_data7 import EntropySearchBigData7
    from .big_data.entropy_search_big_data8 import EntropySearchBigData8
    if index == 0:
        return EntropySearch
    #TODO: there must be a more automatic way to this
    #however, the relative imports make it very difficult
    elif index == 1:
        return EntropySearchBigData1
    elif index == 2:
        return EntropySearchBigData2
    elif index == 3:
        return EntropySearchBigData3
    elif index == 4:
        return EntropySearchBigData4
    elif index == 5:
        return EntropySearchBigData5
    elif index == 6:
        return EntropySearchBigData6
    elif index == 7:
        return EntropySearchBigData7
    elif index == 8:
        return EntropySearchBigData8
    elif index == 9:
        from .big_data.entropy_search_big_data9 import EntropySearchBigData9
        return EntropySearchBigData9
    elif index == 10:
        from .big_data.entropy_search_big_data10 import EntropySearchBigData10
        return EntropySearchBigData10
    else:
        raise NotImplementedError("There exists no acquisition function associated with that number.")

class EntropySearch(object):
    def __init__(self, comp, vals, incumbent, gp, cost_gp=None):
        '''
        Default constructor.
        '''
        self._general_initialization(comp, vals, gp)

        starting_point = incumbent
        self._func_sample_locations = sample_from_proposal_measure(starting_point, self._log_proposal_measure,
                                        NUMBER_OF_REPRESENTER_POINTS)
        self._initialization_after_sampling_representers()

    def _general_initialization(self, comp, vals, gp, cost_gp=None):
        '''
        Is called in __init__ and performs initialization that should also apply to children of this class.
        '''
        self._X = comp
        self._gp = gp
        self._cost_gp = cost_gp
        self._ei = ExpectedImprovement(comp, vals, None, gp, cost_gp)

        #samples for the reprensenter points to compute Pmin
        self._Omega = np.random.normal(0, 1, (NUMBER_OF_PMIN_SAMPLES,
                                              NUMBER_OF_REPRESENTER_POINTS))
        #this speeds up things by a factor of 2!
        self._Omega = np.asfortranarray(self._Omega)

        #we skip the first entry since it will yield 0 and ppf(0)=-infty
        #self._Omega = norm.ppf(i4_sobol_generate(NUMBER_OF_REPRESENTER_POINTS, NUMBER_OF_PMIN_SAMPLES+1, 1)[:,1:]).T

        #samples for the candidates
        #self._omega_cands = np.random.normal(0, 1, NUMBER_OF_CANDIDATE_SAMPLES)
        #we use stratified sampling for the candidates
        self._omega_cands = norm.ppf(np.linspace(1./(NUMBER_OF_CANDIDATE_SAMPLES+1),
                                           1-1./(NUMBER_OF_CANDIDATE_SAMPLES+1),
                                           NUMBER_OF_CANDIDATE_SAMPLES))

    def _initialization_after_sampling_representers(self):
        '''
        Sets the values of the proposal distribution of each representer and computes the Cholesky of the
        covariance matrix.
        '''
        self._log_proposal_vals = np.zeros(NUMBER_OF_REPRESENTER_POINTS)
        #u(x) is fixed - therefore we can compute it here
        for i in range(0, NUMBER_OF_REPRESENTER_POINTS):
            self._log_proposal_vals[i] = self._log_proposal_measure(self._func_sample_locations[i])

        mean, L = self._gp.getCholeskyForJointSample(self._func_sample_locations)
        self._mean = mean[:, np.newaxis]
        self._L = L

    def compute(self, candidate, compute_gradient = False):
        return self.compute_fast(candidate, compute_gradient)

    def compute_fast(self, candidate, compute_gradient = False):
        if compute_gradient:
            raise NotImplementedError("computing gradients not supported by this acquisition function")
        number_of_samples = self._Omega.shape[0]
        idx = np.arange(0, number_of_samples)
        kl_divergence = 0
        # it would be faster to do a rank 1 update for each candidate
        mean, L = self._gp.getCholeskyForJointSample(
            np.append(np.array([candidate]), self._func_sample_locations, axis=0))

        # we don't need the first column of the Cholesky for the computation of Pmin
        # the first column is only multiplied with omega_cand and not Omega
        # therefore we can just add it to the mean
        #i.e. let L = (l0, 0 \\ l, L') and w = (w0, w') then
        # (Lw)_i = (l*w0 + L'w')_(i-1) for i>=2
        l = np.copy(L[1:,0]) #appearantly it's faster to copy
        mean = np.copy(mean[1:])
        L = np.copy(L[1:,1:])

        dotLOmegaT = np.dot(L, self._Omega.T)
        for i in range(0, NUMBER_OF_CANDIDATE_SAMPLES):
            #TODO: refactor! (use function to compute PMin - this is ugly)
            m = mean + l * self._omega_cands[i]
            Y = m[:, np.newaxis] + dotLOmegaT
            min_idx = np.argmin(Y, axis = 0)
            mins = np.zeros([mean.shape[0], number_of_samples])
            mins[min_idx, idx] = 1
            pmin = np.sum(mins, axis = 1)
            pmin = 1./ number_of_samples * pmin
            entropy_pmin = -np.dot(pmin, np.log(pmin+1e-50))
            log_proposal = np.dot(self._log_proposal_vals, pmin)
            #division by NUMBER_OF_CANDIDATE_SAMPLES to keep things numerically stable
            kl_divergence += (entropy_pmin - log_proposal)/NUMBER_OF_CANDIDATE_SAMPLES
        #since originally KL is minimized and we maximize our acquisition functions we have to turn the sign
        return -kl_divergence

    def _log_proposal_measure(self, x):
        if np.any(x<0) or np.any(x>1):
            return -np.inf
        v = self._ei.compute(x)
        return np.log(v+1e-10)

    def _log_proposal_measure_uniform(self, x):
        if np.any(x<0) or np.any(x>1):
            return -np.inf
        #our interval is [0, 1], i.e. p(x)=1/|I|=1 and log 1 = 0
        return 0

    def compute_naive(self, candidate, compute_gradient = False):
        if compute_gradient:
            raise NotImplementedError("computing gradients not supported by this acquisition function")
        kl_divergence = 0
        for i in range(0, NUMBER_OF_CANDIDATE_SAMPLES):
            #it does NOT produce better results if we don't reuse samples here
            # the function just looks more wiggly but has the same problem that it drifts if there are not enough
            # representer points
            y = self._gp.sample(candidate, self._omega_cands[i])
            gp2 = self._gp.copy()
            gp2.update(candidate, y)
            pmin = self._compute_pmin_bins(gp2)
            entropy_pmin = -np.dot(pmin, np.log(pmin+1e-50))
            log_proposal = np.dot(self._log_proposal_vals, pmin)
            #division by NUMBER_OF_CANDIDATE_SAMPLES to keep things numerically stable
            kl_divergence += (entropy_pmin - log_proposal)/NUMBER_OF_CANDIDATE_SAMPLES
        #since originally KL is minimized and we maximize our acquisition functions we have to turn the sign
        return -kl_divergence

    def _compute_pmin_bins(self, gp):
        '''
        Computes a discrete belief over Pmin given a Gaussian process using bin method. Leaves
        the Gaussian process unchanged.
        Returns:
            a numpy array with a probability for each representer point
        '''
        pmin = np.zeros(NUMBER_OF_REPRESENTER_POINTS)
        mean, L = gp.getCholeskyForJointSample(self._func_sample_locations)
        for omega in self._Omega:
            vals = gp.drawJointSample(mean, L, omega)
            mins = np.where(vals == vals.min())[0] #the return value is a tuple
            number_of_mins = len(mins)
            for m in mins:
                pmin[m] += 1./(number_of_mins)
        pmin = pmin / NUMBER_OF_PMIN_SAMPLES
        return pmin


    def _compute_pmin_bins_faster(self, mean, L):
        #legacy function to keep the tests running
        return compute_pmin_bins(mean, L, self._Omega)

    def _compute_pmin_kde(self, gp):
        '''
        THIS FUNCTION DOES NOT WORK!

        Computes a discrete belief over Pmin given a Gaussian process using kernel density estimator.
        Args:
            gp: the Gaussian process
        Returns:
            a numpy array with a probability for each representer point
        '''
        #Should anyone consider using this method: move the function below somewhere else and...
        def SE(xx1, xx2):
            r2 = np.maximum(-(np.dot(xx1, 2*xx2.T)
                           - np.sum(xx1*xx1, axis=1)[:,np.newaxis]
                           - np.sum(xx2*xx2, axis=1)[:,np.newaxis].T), 0.0)
            cov = np.exp(-0.5 * r2)
            return cov
        #... this part into the constructor.
        self._normalization_constant = np.zeros(NUMBER_OF_REPRESENTER_POINTS)
        for i in range(0, NUMBER_OF_REPRESENTER_POINTS):
            for j in range(0, NUMBER_OF_REPRESENTER_POINTS):
                self._normalization_constant[i] += SE(np.array([self._func_sample_locations[i]]),
                                                      np.array([self._func_sample_locations[j]]))

        pmin = np.zeros(NUMBER_OF_REPRESENTER_POINTS)
        for omega in self._Omega:
            vals = gp.drawJointSample(self._func_sample_locations, omega)
            mins = np.where(vals == vals.min())
            number_of_mins = len(mins)
            for m in mins:
                m = m[0]
                for i in range(0, NUMBER_OF_REPRESENTER_POINTS):
                    pmin[i]+=SE(np.array([self._func_sample_locations[i]]), np.array([self._func_sample_locations[m]]))\
                             /self._normalization_constant[m]
                    pmin[i]/=number_of_mins
        pmin = pmin / NUMBER_OF_PMIN_SAMPLES
        return pmin
