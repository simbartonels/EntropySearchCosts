'''
Created on 28.03.2014

@author: Aaron Klein, Simon Bartels

This class extends EntropySearchBigData (with all its assumptions) for a concrete transformation with costs.
'''
import numpy as np
from ..entropy_search_big_data import EntropySearchBigData

class EntropySearchBigData9(EntropySearchBigData):
    def __init__(self, comp, vals, incumbent, gp, cost_gp=None):
        super(EntropySearchBigData9, self)._initialize(comp, vals, incumbent, gp, cost_gp)

    def _compute(self, kl, costs, candidate):
        '''
        Returns exp[ kl - current_kl ] / log( costs ).
        Args:
            kl: the output of #super.compute(candidate)
            costs: the costs predicted for the candidate
            candidate: the candidate
        Returns:
            Some value that is a transformation of kl and costs
        '''
        return np.exp(kl - self._current_kl)/ np.log(costs)