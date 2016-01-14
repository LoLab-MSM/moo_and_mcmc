# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:27:34 2016

@author: Erin
"""

import numpy as np
from scipy.stats import norm
from scipy.stats import uniform


class SampledParam():
    def __init__(self, name, value):
        self.name = name
        self.dsize = np.array(value).size
    
class NormalParam(SampledParam):
    def __init__(self, name, value, mu, sd):
        self.name = name
        self.value = value
        self.dsize = np.array(value).size
        self.mu = mu
        self.sd = sd
    
    def random(self):
        return np.random.normal(self.mu, self.sd)
    
    def prior(self, q0):
        return np.sum(norm.logpdf(q0, self.mu, self.sd))

class UniformParam(SampledParam):
    def __init__(self, name, value, lower, upper):
        self.name = name
        self.value = value
        self.dsize = np.array(value).size
        self.lower = lower
        self.upper = upper
        self.range = self.upper - self.lower
        
    def random(self):
        return np.random.uniform(self.lower, self.upper)
    
    def prior(self, q0):
        return np.sum(uniform.logpdf(q0, self.lower, self.range))