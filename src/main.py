import pandas as pd
import numpy as np
import random
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
MUTATION_RATE = 0.01
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
    @fitness.setter
    def fitness(self, fitness):
        self._fitness = fitness

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

    def crossover(self, mate):
        length = len(self._gene[0])
        pivot = random.randint(0, length)
        gene_1 = np.concatenate((self._gene[:,0:pivot], mate.gene[:,pivot:length]), axis=1)
        gene_2 = np.concatenate((mate.gene[:,0:pivot], self._gene[:,pivot:length]), axis=1)
        return Chromosome(gene_1, self.costs, self.time_rules), Chromosome(gene_2, self.costs, self.time_rules)

    # Return False if the total time usage of appliances is larger
    # than time usage in problems rules
    def is_feasible(self):
        feasible = True
        index_data = []
        for i in range(0, len(self._gene)):
            hour_size = np.sum(self._gene[i][self.time_rules[i][0]:self.time_rules[i][1]])
            if (hour_size != self.time_rules[i][2]):
                feasible = False
                index_data.append((i,hour_size))
        return feasible, index_data
    
    # Reconstruct infeasible solution / chromosome
    def reconstruct_gene(self):
        feasible, index = self.is_feasible()
        if (not feasible):
            # value is a tuple, consist of index and device usage
            for value in index:
                app_index = value[0]
                time_usage = self.time_rules[app_index][2]
                current_time_usage = value[1]
                if (current_time_usage > time_usage):
                    # turn off the device is random time
                    on_index = [index for index in range(0, 48) if self._gene[app_index][index] == 1] #list index of on devices
                    set_off_index_len = len(on_index) - time_usage
                    off_index = random.sample(on_index, set_off_index_len)
                    for j in off_index:
                        self._gene[app_index][j] = 0
                elif (current_time_usage < time_usage):
                    # turn on the device is random time
                    start = self.time_rules[app_index][0]
                    end = self.time_rules[app_index][1]
                    off_index = [index for index in range(start, end) if self._gene[app_index][index] == 0]
                    set_on_index_len = time_usage - current_time_usage
                    on_index = random.sample(off_index, set_on_index_len)
                    for j in on_index:
                        self._gene[app_index][j] = 1
        return 'done'
        

    def fitness_function(self):
        tmp=0.0
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
        self.population = self.generate_population()
        
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
    
    def roullette_wheel_normalized(self):
        # based on paper (Eunji Lee and Hyokyung Bahn)
        fitness_list = [chromosome.fitness for chromosome in self.population]
        best = min(fitness_list)
        worst = max(fitness_list)
        normalized_list =[(worst-current + (worst-best)/3) for current in fitness_list ]
        
        portion_wheel = [normalized_list[0]]
        for i in range(1, len(normalized_list)):
            portion_wheel.append(portion_wheel[i-1]+normalized_list[i])
        choice = random.uniform(0, sum(normalized_list))
        selected_index = 0
        for i in range(1, len(portion_wheel)):
            if (choice > portion_wheel[i-1] and choice < portion_wheel[i]):
                selected_index = i
                break
        return selected_index

    def roullette_wheel(self):
        # set the portion of wheel, fitness-nth/sum(fitness)
        fitness_list = [chromosome.fitness for chromosome in self.population]
        total_fitness = sum(fitness_list)
        prob_fitness = [fitness/total_fitness for fitness in fitness_list]
       
        portion_wheel = [prob_fitness[0]]
        for i in range(1,len(fitness_list)):
            portion_wheel.append(portion_wheel[i-1]+prob_fitness[i])
        
        choice = random.random()
        selected_index = 0
        for i in range(1, len(portion_wheel)):
            if (choice > portion_wheel[i-1] and choice < portion_wheel[i]):
                selected_index = i
                break
        return selected_index

    def parent_selection(self):
#        Select 2 parent randomly
        parent_1 = self.population[random.randrange(0,len(self.population))]
        parent_2 = self.population[random.randrange(0,len(self.population))]
#        parent_1 = self.population[self.roullete_wheel()]
#        parent_2 = self.population[self.roullete_wheel()]
        return parent_1, parent_2
    
    def evolve(self):
        new_pop = []
        for i in range(0, self.size):
            parent_1, parent_2 = self.parent_selection()
            rand_numb = random.random()
            if (rand_numb <= self.crossover_rate):
                offspring_1, offspring_2 = parent_1.crossover(parent_2)
#                offspring_1.reconstruct_gene()
#                offspring_2.reconstruct_gene()
                offspring_1.mutation()
                offspring_2.mutation()
                offspring_1.reconstruct_gene()
                offspring_2.reconstruct_gene()
                new_pop.append(offspring_1)
                new_pop.append(offspring_2)
            else:
                parent_1.mutation()
                parent_2.mutation()
                parent_1.reconstruct_gene()
                parent_2.reconstruct_gene()
                new_pop.append(parent_1)
                new_pop.append(parent_2)
#        append new pop with current population
        self.population = self.population + new_pop
        self.population.sort(key=lambda x: x.fitness)
        self.population = self.population[:self.size]
        
        best_fitness = self.population[0].fitness
        worst_fitness = self.population[self.size-1].fitness
        return self.population, best_fitness, worst_fitness
            
        

#%%
if __name__ == "__main__":
    generation_size = 200
        
    appliances_df = pd.read_csv('sample-data.csv', parse_dates=['start', 'end'], date_parser=parse_date)
    
    pop = Population(appliances_df, 100)
    for i in range(generation_size):
        population, best_fitness, worst_fitness = pop.evolve()
        print('GENERATION : ', i+1)
        print('best fitness:', best_fitness)
        print('worst fitness:', worst_fitness)
        print('')
        
        