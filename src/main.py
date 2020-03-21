import pandas as pd
import numpy as np
import random
import os
import datetime

# Helper function
def parse_date(date):
    return datetime.datetime.strptime(date, '%I:%M:%S %p')

def time_index(time):
    init_time = parse_date('12:00:00 AM')
    deltatime = time - init_time
    return int(deltatime.total_seconds()/3600*2)

# CONSTANT
GEN_SIZE = 200
POP_SIZE = 100
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.80


# time price periode: 
# start time index, end time index, prices (in dollar)
OFF_PEAK_1 = [time_index(parse_date('12:00:00 AM')), time_index(parse_date('05:30:00 AM')), 0.4]
PEAK_1     = [time_index(parse_date('06:00:00 AM')), time_index(parse_date('09:30:00 AM')), 1.0]
STANDARD   = [time_index(parse_date('10:00:00 AM')), time_index(parse_date('02:30:00 PM')), 0.6]
PEAK_2     = [time_index(parse_date('03:00:00 PM')), time_index(parse_date('08:30:00 PM')), 1.0]
OFF_PEAK_2 = [time_index(parse_date('09:00:00 PM')), time_index(parse_date('11:30:00 PM')), 0.4]
TARIFF     = [OFF_PEAK_1, PEAK_1, STANDARD, PEAK_2, OFF_PEAK_2]

class Chromosome:
    def __init__(self, gene, costs, time_rules):
        self._gene = gene
        self._costs = costs
        self.time_rules = time_rules
        self._fitness = self.fitness_function()
        
    @property
    def gene(self):
        return self._gene
    @property
    def costs(self):
        return self._costs
    @property
    def fitness(self):
        return self._fitness

    # Bitwise mutation
    def mutation(self):
        for i in range(len(self._gene)):
            start = self.time_rules[i][0]
            end = self.time_rules[i][1]
            for j in range(start, end+1):
                rand_number = random.random()
                if (rand_number <= MUTATION_RATE):
                    self._gene[i][j] = 1 if self._gene[i][j] == 0 else 0
        return Chromosome(self._gene, self.costs, self.time_rules)

    def crossover(self):
        return Chromosome(self._gene)

    # Return False if the total time usage of appliances is larger
    # than time usage in problems rules
    def is_feasible(self):
        feasible = True
        index_data = []
        for i in range(0, len(self._gene)):
            hour_size = np.sum(self._gene[i][self.time_rules[i][0]:self.time_rules[i][1]])
            if (hour_size > self.time_rules[i][2]):
                feasible = False
                index_data.append(i)
        return feasible, index_data
    
    # Reconstruct infeasible solution / chromosome
    def reconstruct_gene(self):
        feasible, index = self.is_feasible()
        if (not feasible):
            print('reconstruct chromosome until have feasible solutions')
            for i in index:
                on_index = [index for index in range(0, 48) if self.gene[i][index] == 1]
                set_off_index_len = len(on_index) - self.time_rules[i][2]
                print('index allelle one', on_index, 'delete:', set_off_index_len)
                off_index = random.sample(on_index, set_off_index_len)
                for j in off_index:
                    self._gene[i][j] = 0
        return 'done'
        

    def fitness_function(self):
        print(self.time_rules)
        tmp=0.0
        print(len(self._gene))
        for i in range(len(self._gene)):
            for j in range(len(TARIFF)):
                start = TARIFF[j][0]
                end = TARIFF[j][1]
                price = TARIFF[j][2]
                tmp = tmp + np.sum(self._gene[i][start:end+1]) / 2 * price * self._costs[i]
        return tmp


class Population():
    def __init__(self, appliances_df, size=POP_SIZE, mutation_rate=MUTATION_RATE, crossover_rate=CROSSOVER_RATE):
        self.size = size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.appliances_df = appliances_df
        
    def generate_random_individual(self):              
#       Create zeros element array with size = number of appliances * 48 (time sch)
        app_sch = np.zeros((len(self.appliances_df), 48), int)
        time_rules = []
        for i in range(0, len(app_sch)):
            max_duration = int(self.appliances_df.iloc[i]['duration']*2)
            start_element = time_index(self.appliances_df.iloc[i]['start'])
            end_element = time_index(self.appliances_df.iloc[i]['end'])
            time_rules.append([start_element, end_element, max_duration])
            for j in range(0, max_duration):
                rand_index = random.randrange(start_element, end_element+1)
                app_sch[i, rand_index] = 1
        return Chromosome(app_sch, self.appliances_df['costs'].to_numpy(), time_rules)
        
    def generate_population(self):
        pop_lists = []
        for i in range(0, self.size):
            pop_lists.append(self.generate_random_individual())
  
        return pop_lists

    def roullete_wheel(self):
        return 'tuple of chromosome'

    def parent_selection(self):
        return 'population'

#%%
if __name__ == "__main__":
    generation_size = 100
        
    appliances_df = pd.read_csv('sample-data.csv', parse_dates=['start', 'end'], date_parser=parse_date)
    
    pop = Population(appliances_df, 1)
    first_pop = pop.generate_population()
    print('Number of population', len(first_pop))
    print('Fitness', first_pop[0].fitness)
    print('Feasible?:', first_pop[0].is_feasible())
    
    print('\nDo mutation')
    xx = first_pop[0].mutation()
    print(xx.is_feasible())
    xx.reconstruct_gene()
    print(xx.is_feasible())