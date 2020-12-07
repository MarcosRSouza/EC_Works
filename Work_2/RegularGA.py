# Author: MarcosRSouza
# The following code is under a MIT license
# So feel free to modify/improve it in your own applications :)

import math
import random
import numpy as np
import matplotlib.pyplot as plt
#random.seed(10)

class Subject:  #abstração para criação do individuo
    def __init__(self, chromosome, fitness, rouletteParticipation):   #todo individuo tem genes e um fitness
        self.chromosome = chromosome
        self.fitness = fitness  #fitness = grau de adaptabilidade (quao mais apto ele é)
        self.rouletteParticipation = rouletteParticipation #indica a faixa que o individuo ocupa na roleta

def calcFitness(chromosome): #Calcular fitness de verdade !!!!
    XandYlist = decodXY(chromosome)
    x = XandYlist[0]
    y = XandYlist[1]
    fitness = 0.5 - (math.pow(math.sin(math.sqrt(x*x+y*y)), 2) - 0.5)/math.pow((1 + 0.001*(x*x+y*y)), 2)
    return fitness

def decodXY(chromosome):
    xInt = 0
    yInt = 0
    for i in range(0,int(len(chromosome)/2)): #corta a metade que corresponde a x
        if chromosome[i] == 1:
            xInt += math.pow(2, i)
    j = 0 # iterador diferente para não confundir o indice de i na conta de y
    for i in range(int(len(chromosome)/2),int(len(chromosome))): #corta a metade que corresponde a y
        if chromosome[i] == 1:
            yInt += math.pow(2, j)
        j+=1
    xFloat = (xInt/(math.pow(2,len(chromosome)/2) - 1) * 200) - 100 #formula para conversão de x para real
    yFloat = (yInt/(math.pow(2,len(chromosome)/2) - 1) * 200) - 100 #formula para conversão de x para real
    return [xFloat,yFloat]

def generateRouletteParticipation(population):
    sum = 0
    for i in range(100):
        sum += population[i].fitness
        population[i].rouletteParticipation = sum
    return population

def generatePop(popSize):
    population = []
    for p in range(popSize): #vai até o tamanho da população escolhida
        cromo = []
        for i in range (56): #constroi o gene de cada individuo
            cromo.append(random.choice([0, 1]))
        fit = calcFitness(cromo) # calcula o fitness do individuo
        randomSub = Subject(cromo, fit, 0) #cria um individuo aleatorio com cromoss. e fitness proprios
        population.append(randomSub)    #adiciona individuo a população
    return population #ao final teremos uma população com nºpopSize individuos aleatorios

def generateChildren(father1chromo, father2chromo):
    child1Cromo = father1chromo.copy()
    child2Cromo = father2chromo.copy()
    cuttingPoint = random.randrange(len(father1chromo) - 1)
    for k in range(cuttingPoint, len(father1chromo)):
        child1Cromo[k] = father2chromo[k]
    for j in range(cuttingPoint, len(father1chromo)):
        child2Cromo[j] = father1chromo[j]

    #test mutation for child 1
    if random.random()  <= mutationRate:
        randomIndex = random.randrange(len(child1Cromo) - 1) #escolhe uma posição aleatória do cromossomo
        child1Cromo[randomIndex] = not(child1Cromo[randomIndex])
    #test mutation for child 2
    if random.random()  <= mutationRate:
        randomIndex = random.randrange(len(child1Cromo) - 1) #escolhe uma posição aleatória do cromossomo
        child2Cromo[randomIndex] = not(child2Cromo[randomIndex])
    child1Fitness = calcFitness(child1Cromo)
    child2Fitness = calcFitness(child2Cromo)
    child1 = Subject(child1Cromo, child1Fitness, 0)
    child2 = Subject(child2Cromo, child2Fitness, 0)
    return [child1, child2]
    
# print("Qual o tamanho da população? ")
# populationSizeEntered = input("R: ")
# print("Quantas gerações? ")
# numberOfGenerations = input("R: ")
# print("Qual a taxa de cruzamento? ")
# reproductionRate = input("R: ")
# print("Qual a taxa de mutação? ")
# mutationRate = input("R: ")
populationSizeEntered = 100
numberOfGenerations = 500
reproductionRate = 0.75
mutationRate = 0.1

#marca 1 execução
executionNumberN = 0 #marca o numero de experimentos

worstIndividualsInNexecutions = []
bestIndividualsInNexecutions = []
populationAverageInNexecutions = []
initialPopulationXPositionAverage = []
initialPopulationYPositionAverage = []
middlePopulationXPositionAverage = []
middlePopulationYPositionAverage = []
finalPopulationXPositionAverage = []
finalPopulationYPositionAverage = []

for  i in range(0, numberOfGenerations):
    worstIndividualsInNexecutions.append(0)
bestIndividualsInNexecutions = worstIndividualsInNexecutions.copy()
populationAverageInNexecutions = worstIndividualsInNexecutions.copy()
initialPopulationXPositionAverage = worstIndividualsInNexecutions.copy()
initialPopulationYPositionAverage = worstIndividualsInNexecutions.copy()
middlePopulationXPositionAverage = worstIndividualsInNexecutions.copy()
middlePopulationYPositionAverage = worstIndividualsInNexecutions.copy()
finalPopulationXPositionAverage = worstIndividualsInNexecutions.copy()
finalPopulationYPositionAverage = worstIndividualsInNexecutions.copy() #apenas cria uma lista com 100 nºs 0

while executionNumberN < 10:

    generationN = 0
    initialPopulation = generatePop(int(populationSizeEntered)) #cria população aleatoria
    currentPopulation = initialPopulation
    childrenList = [] #stores the new population to substitute old population
    worstIndividualsFitness = []
    bestIndividualsFitness = []
    populationAverageFitness = []

    xData1 = []
    yData1 = []
    xData2 = []
    yData2 = []
    xData3 = []
    yData3 = []

    while generationN < numberOfGenerations:

        currentPopulation = sorted(currentPopulation, key=lambda subject: subject.fitness) #ordenar população por fitness
        currentPopulation = generateRouletteParticipation(currentPopulation)

        #lembre-se que a lista esta ordenada, logo é possivel fazer coisas como:
        worstIndividualsFitness.append(currentPopulation[0].fitness) #primeiro é o pior individuo
        bestIndividualsFitness.append(currentPopulation[99].fitness) #ultimo é o melhor individuo
        populationAverageFitness.append(currentPopulation[99].rouletteParticipation/100) #o ultimo tem a soma dos fitness da populaçao toda
        
        #etapa de cruzamento
        if random.random() <= reproductionRate: #teste para ver se ha cruzamento
            father1RouletteNumber = random.random()*currentPopulation[99].rouletteParticipation
            father2RouletteNumber = random.random()*currentPopulation[99].rouletteParticipation
            father1 = []
            father2 = []
            for i in range(100):
                if father1RouletteNumber < currentPopulation[i].rouletteParticipation:
                    father1 = currentPopulation[i]
                if father2RouletteNumber < currentPopulation[i].rouletteParticipation:
                    father2 = currentPopulation[i]

            Children1and2List = generateChildren(father1.chromosome, father2.chromosome)
            childrenList.append(Children1and2List[0])
            childrenList.append(Children1and2List[1])

        if len(childrenList) ==  100:
            currentPopulation = childrenList.copy()
            childrenList.clear()

        if generationN ==  0:
            for subject in currentPopulation:
                XandY = decodXY(subject.chromosome)
                xData1.append(XandY[0])
                yData1.append(XandY[1])
        elif generationN ==  numberOfGenerations/2:
            for subject in currentPopulation:
                XandY = decodXY(subject.chromosome)
                xData2.append(XandY[0])
                yData2.append(XandY[1])
        elif generationN == numberOfGenerations - 1:
            for subject in currentPopulation:
                XandY = decodXY(subject.chromosome)
                xData3.append(XandY[0])
                yData3.append(XandY[1])
        generationN+=1

    for j in range(0, numberOfGenerations):
        worstIndividualsInNexecutions[j] += worstIndividualsFitness[j]
        bestIndividualsInNexecutions[j] += bestIndividualsFitness[j]
        populationAverageInNexecutions[j] += populationAverageFitness[j]
    for k in range(populationSizeEntered):
        initialPopulationXPositionAverage[k] += xData1[k]
        initialPopulationYPositionAverage[k] += yData1[k]
        middlePopulationXPositionAverage[k] += xData2[k]
        middlePopulationYPositionAverage[k] += yData2[k]
        finalPopulationXPositionAverage[k] += xData3[k]
        finalPopulationYPositionAverage[k] += yData3[k]
    executionNumberN += 1
#termina 10 execuções

for i in range(numberOfGenerations):
    worstIndividualsInNexecutions[i] /= 10
    bestIndividualsInNexecutions[i] /= 10
    populationAverageInNexecutions[i] /= 10
    initialPopulationXPositionAverage[i] /= 10
    initialPopulationYPositionAverage[i] /= 10
    middlePopulationXPositionAverage[i] /= 10
    middlePopulationYPositionAverage[i] /= 10
    finalPopulationXPositionAverage[i] /= 10
    finalPopulationYPositionAverage[i] /= 10

#plotando fitness do pior individuo e media da populacao por geração
generationNumber = range(numberOfGenerations)
#bestIndividualsFitnessForScatter = []
#populationAverageFitnessForScatter = []
#generationNumberForScatter = []
#for i in range(0,numberOfGenerations, 2):
#    bestIndividualsFitnessForScatter.append(bestIndividualsFitness[i])
#    populationAverageFitnessForScatter.append(populationAverageFitness[i])
#    generationNumberForScatter.append(i)
# #plt.plot(generationNumber, worstIndividualsFitness, label = "worst ind. fitness")
fig = plt.figure(figsize=(6,5))

#plt.plot(generationNumber, bestIndividualsFitness, label = "best ind. fitness")
#plt.plot(generationNumber, populationAverageFitness, label = "pop avg. fitness")

#bif = plt.scatter(generationNumberForScatter, bestIndividualsFitnessForScatter, marker='x', c='blue')
#paf = plt.scatter(generationNumberForScatter, populationAverageFitnessForScatter, marker='x',c='red')
bif = plt.scatter(generationNumber, bestIndividualsInNexecutions, marker='x', c='blue')
paf = plt.scatter(generationNumber, populationAverageInNexecutions, marker='x',c='red')

plt.xlabel('Generation number n') #naming the x axis
plt.ylabel('Fitness') #naming the  y axis
plt.title('Best Individuals and Population Average Fitness') #giving a title to the graph
#plt.legend()
plt.legend((bif, paf),
           ('Best Ind. Fitness', 'Pop Avg. Fitness'),
           scatterpoints=1,
           loc='lower right',
           fontsize=8)
plt.show() #finally showing the plot
#plotando fitness da população inicial

left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height]) 

start, stop, n_values = -100, 100, 2000

x_vals = np.linspace(start, stop, n_values)
y_vals = np.linspace(start, stop, n_values)
X, Y = np.meshgrid(x_vals, y_vals)

Z = 0.5 - (np.sin(np.sqrt(X**2 + Y**2)) ** 2 - 0.5)/((1 + 0.001*(X**2 + Y**2)) ** 2)

cp = plt.contourf(X, Y, Z)
plt.colorbar(cp)
gen0 = plt.scatter(initialPopulationXPositionAverage, initialPopulationYPositionAverage, c='red')
genHalf = plt.scatter(middlePopulationXPositionAverage, middlePopulationYPositionAverage, c='purple')
genLast = plt.scatter(finalPopulationXPositionAverage, finalPopulationYPositionAverage, c='black')
plt.legend((gen0, genHalf, genLast),
           ('First Generation', 'Middle Generation', 'Last Generation'),
           scatterpoints=1,
           loc='lower right',
           fontsize=8)

ax.set_title('Contour Plot')
ax.set_xlabel('X value')
ax.set_ylabel('Y value')
plt.show()