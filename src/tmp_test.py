#!/usr/bin/env python
from src.neti_neti_trainer import NetiNetiTrainer
from src.neti_neti import NetiNeti
import subprocess
import shlex
import math
import sys
import time

num_cycles = 3

if len(sys.argv) > 1:
    try:
        num_cycles =  int(sys.argv[1]) + 1
    except ValueError:
        pass

def variance(population):
    n = 0
    mean = 0.0
    s = 0.0
    for x in population:
        n = n + 1
        delta = x - mean
        mean = mean + (delta / n)
        s = s + delta * (x - mean)

    return s / (n-1)

# calculate the standard deviation of a population
# accepts: an array, the population
# returns: the standard deviation
def standard_deviation(population):
    if len(population) == 1:
        return 0
    return math.sqrt(variance(population))


population = []

time_start = time.clock()
classifier = NetiNetiTrainer()
time_training = time.clock()
print ("Training time: %s" % (time_training - time_start))
nn = NetiNeti(classifier)
for i in range(1, num_cycles):
    print ("going through the cycle %s" % i)
    time_start = time.clock()
    result = nn.find_names(open("data/test.txt").read())
    print ("Name finding time: %s" % (time.clock() - time_start))

    test_result_file = open("data/test_result_after_refactoring.txt", 'w')

    for i in result[1]:
        test_result_file.write(i + "\n")

    test_result_file.close()

    args = shlex.split('diff -d  data/test_result_before_refactoring.txt data/test_result_after_refactoring.txt')
    result = subprocess.Popen(args, stdout = subprocess.PIPE, stderr= subprocess.PIPE).communicate()
    result = result[0].split("\n")
    ins = []
    outs = []
    for i in result:
        if len(i) > 0:
            if i[0] == '>':
                outs.append(i.strip())
            if i[0] == '<':
                ins.append(i.strip())
        differences = ins + outs
    population.append(len(differences))

dev = standard_deviation(population)

sum = 0
for i in population:
    sum = sum + i
set = ','.join([str(p) for p in population])
print ("Mean: %s, St. dev: %s, set: %s" % (float(sum)/float(len(population)), dev, set))