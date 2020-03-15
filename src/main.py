#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 19:14:24 2020

@author: fhabibie
"""

import numpy as np
import random

class Chromosome:
    def __init__(self, gene):
        self.gene = gene
        self.fitness = self.fitness_function(gene)

    def mutation(self):
        return Chromosome(self.gene)
    
    def crossover(self):
        return Chromosome(self.gene)
    
    def is_feasible(self):
        return 'boolean value'
    
    def fitness_function(self):
        fitness = random()
        return fitness

class Population:
    def __init__(self, size=10, mutation_rate = 0.05, crossover_rate = 0.8):
        self.size = size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
    
    def generate_population(self):
        return 'list of chromosome'
    
    def roullete_wheel(self):
        return 'tuple of chromosome'
    
    def parent_selection(self):
        return 'population'

if __name__ == "__main__":
    generation_size = 100
    