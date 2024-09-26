"""

"""
# Résolution du problème du sac à dos ou KP (Knapsack Problem) à  l'aide d'algorithme génétique
# import des librairies

import numpy as np #utilisation des calculs matriciels
#import pandas as pd #générer et lire les fichiers csv
import random as rd #génération de nombres aléatoires
from random import randint # génération des nombres aléatoires  
import matplotlib.pyplot as plt
import instances

# Données du problème générées aléatoirement
nombre_objets = 10   #Le nombre d'objets
capacite_max = 2000    #La capacité du portefeuille

#paramètres de l'algorithme génétique
nbr_generations = 1000 # nombre de générations


# instances.create_csv_files(nombre_objets, 1, 15, 50, 350)
ID_objets, poids, valeur = np.array([]), np.array([]), np.array([]) 
with open('instances/instance3.csv', mode='r', newline ='') as file:
    for i, line in enumerate(file):
        if i == 0:
            continue
        fields = line.strip().split(',')
        ID_objets = np.append(ID_objets, int(fields[0]))
        poids = np.append(poids, int(fields[1]))
        valeur = np.append(valeur, int(fields[2]))

ID_objets = ID_objets.astype(int)
poids = poids.astype(int)
valeur = valeur.astype(int)

#affichage des objets: Une instance aléatoire du problème Knapsack
print('La liste des objet est la suivante :')
print('ID_objet   Poids   Valeur')
for i in range(ID_objets.shape[0]):
    print(f'{ID_objets[i]} \t {poids[i]} \t {valeur[i]}')
print()


# Créer la population initiale
solutions_par_pop = 8 #la taille de la population 
pop_size = (solutions_par_pop, ID_objets.shape[0])

max_budget_per_object = capacite_max*0.2        #calcul budget max pour chaque objet
max_nbr_per_object = np.array([])
population_initiale = np.full(pop_size, -1)
max_nbr_per_object = [max_budget_per_object // poids[i] \
                        for i in range(0, pop_size[1])]   #calcul nb max pour chaque objet
for i in range(0, pop_size[0]):
    individual = np.array([np.random.randint(max_nbr_per_object[j]) \
                        for j in range(0, pop_size[1])]) #generation de la population init
    
    population_initiale[i] = individual 

print(max_nbr_per_object)

print(f'Taille de la population: {pop_size}')
print(f'Population Initiale: \n{population_initiale}')

def inverseIndividu(individual, S1, S2, valeur, poids):
    while(S2 > capacite_max):
        index = np.random.randint(0, individual.shape[0]-1)
        while(individual[index]!=1):
            index = np.random.randint(0, individual.shape[0])
        individual[index] = 0
        S1 = np.sum(individual*valeur)
        S2 = np.sum(individual*poids)
    return S1, S2

def cal_fitness(poids, valeur, population, capacite):
    fitness = np.empty(population.shape[0])
    poidsf = np.empty(population.shape[0])
    for i in range(population.shape[0]):
        S1 = np.sum(population[i] * valeur)
        S2 = np.sum(population[i] * poids)

        if S2 <= capacite:
            fitness[i] = S1
        else:
            fitness[i] = capacite-S2
        poidsf[i] = S2
    return fitness.astype(int)  

def selection(fitness, nbr_parents, population):
    fitness = list(fitness)
    parents = np.empty((nbr_parents, population.shape[1]))

    for i in range(nbr_parents):
        indice_max_fitness = np.where(fitness == np.max(fitness))
        parents[i,:] = population[indice_max_fitness[0][0], :]
        fitness[indice_max_fitness[0][0]] = -999999

    return parents

def croisement(parents, nbr_enfants):
    enfants = np.empty((nbr_enfants, parents.shape[1]))
    point_de_croisement = int(parents.shape[1]/2) #croisement au milieu
    taux_de_croisement = 0.8
    i = 0

    while (i < nbr_enfants): #parents.shape[0]
        indice_parent1 = i%parents.shape[0]
        indice_parent2 = (i+1)%parents.shape[0]
        x = rd.random()
        if x > taux_de_croisement: # probabilité de parents stériles
            continue
        indice_parent1 = i%parents.shape[0]
        indice_parent2 = (i+1)%parents.shape[0]
        enfants[i,0:point_de_croisement] = parents[indice_parent1,0:point_de_croisement]
        enfants[i,point_de_croisement:] = parents[indice_parent2,point_de_croisement:]
        i+=1

    return enfants

# La mutation consiste à changer le bit
def mutation(enfants):
    mutants = np.empty((enfants.shape))
    taux_mutation = 0.5
    for i in range(mutants.shape[0]):
        random_valeur = rd.random()
        mutants[i,:] = enfants[i,:]
        if random_valeur > taux_mutation:
            continue
        #choisir aléatoirement le bit à changer
        int_random_valeur = randint(0,enfants.shape[1]-1)    
        value = np.random.randint(max_nbr_per_object[int_random_valeur])
        mutants[i,int_random_valeur] = value
    return mutants  

def optimize(poids, valeur, population, pop_size, nbr_generations, capacite):
    sol_opt, historique_fitness = [], []
    nbr_parents = pop_size[0]//2
    nbr_enfants = pop_size[0] - nbr_parents 
    for _ in range(nbr_generations):
        fitness = cal_fitness(poids, valeur, population, capacite)
        historique_fitness.append(fitness)
        parents = selection(fitness, nbr_parents, population)
        enfants = croisement(parents, nbr_enfants)
        mutants = mutation(enfants)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutants

    print(f'Voici la dernière génération de la population: \n{population}\n') 
    fitness_derniere_generation = cal_fitness(poids, valeur, population, capacite)      
    print(f'Fitness de la dernière génération: \n{fitness_derniere_generation}\n')
    max_fitness = np.where(fitness_derniere_generation == np.max(fitness_derniere_generation))
    sol_opt.append(population[max_fitness[0][0],:])

    return sol_opt, historique_fitness


#lancement de l'algorithme génétique
sol_opt, historique_fitness = optimize(poids, valeur, population_initiale, pop_size, nbr_generations, capacite_max)


#affichage du résultat
print('La solution optimale est:')
print('objets n°', [i for i, j in enumerate(sol_opt[0]) if j!=0])
print('onts pris', [sol_opt[0][i] for i in range (0, len(sol_opt[0])) if sol_opt[0][i]!=0], 'fois')

print(f"Avec un gain de {np.amax(historique_fitness)} € et un valeur d'achat de {np.sum(sol_opt * poids)} €")


historique_fitness_moyenne = [np.mean(fitness) for fitness in historique_fitness]
historique_fitness_max = [np.max(fitness) for fitness in historique_fitness]
plt.plot(list(range(nbr_generations)), historique_fitness_moyenne, label='Valeurs moyennes')
plt.plot(list(range(nbr_generations)), historique_fitness_max, label='Valeur maximale')
plt.legend()
plt.title('Evolution de la Fitness à travers les générations en Euros')
plt.xlabel('Générations')
plt.ylabel('Fitness')
plt.show()