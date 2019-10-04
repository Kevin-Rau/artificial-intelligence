#HW 5 A not so sad attempt by
#Kevin Rau collaborated with Jesus Ortiz Tovar and Ryan O'Connell
#CSCI 3202 Fall 2016


import sys
import math
import random
import time

from random import random
from random import randint
from sys import argv


'''
This is soley used for calulating things that we can use in the print statement 
Ignore this now, but keeping cause its nice to know how to do this

def grabGraph(argv):
	
	with open(filename) as file:
		arrayd = [[char for char in line.split()] for line in file]

	population_size  = len(array2d) * len(array2d[0])
	district_size = population_size/len(array2d)

	return population_size, district_size
'''
#The below code I can thank Ryan for beacause I for the life of me could not figure out how to make these
#districs 

def dist_positions(district_lists, dist_locs):
	if fitnessFunction(district_lists) == 0:
		global global_locs
		global_locs = dist_locs[:]
		global found
		found = 1

def generateSolution(data):
	district_lists = [[] for i in range(len(data))]
	dist_locs = [[] for i in range(len(data))]
	if len(district_lists) == 8:
		for i in xrange(len(data)):
			for j in xrange(len(data)):
				k = int(random() * 2)
				if k == 0 and len(district_lists[i]) != 8:
					district_lists[i].append(data[i][j])
					dist_locs[i].append((i,j))
				else:
					if i != 7:
						district_lists[i + 1].append(data[i][j])
						dist_locs[i + 1].append((i,j))
					else:
						if len(district_lists[7]) != 8:
							district_lists[7].append(data[i][j])
							dist_locs[7].append((i,j))
						elif len(district_lists[6]) != 8:
							district_lists[6].append(data[i][j])
							dist_locs[6].append((i,j))
						elif len(district_lists[5]) != 8:
							district_lists[5].append(data[i][j])
							dist_locs[5].append((i,j))
						elif len(district_lists[4]) != 8:
							district_lists[4].append(data[i][j])
							dist_locs[4].append((i,j))
						elif len(district_lists[3]) != 8:
							district_lists[3].append(data[i][j])
							dist_locs[3].append((i,j))
						elif len(district_lists[2]) != 8:
							district_lists[2].append(data[i][j])
							dist_locs[2].append((i,j))
						elif len(district_lists[1]) != 8:
							district_lists[1].append(data[i][j])
							dist_locs[1].append((i,j))
						elif len(district_lists[0]) != 8:
							district_lists[0].append(data[i][j])
							dist_locs[0].append((i,j))
	else:
		for i in xrange(len(data)):
			for j in xrange(len(data)):
				k = int(random() * 2)
				if k == 0 and len(district_lists[i]) != 10:
					district_lists[i].append(data[i][j])
					dist_locs[i].append((i,j))
				else:
					if i != 9:
						district_lists[i + 1].append(data[i][j])
						dist_locs[i + 1].append((i,j))
					else:
						if len(district_lists[9]) != 10:
							district_lists[9].append(data[i][j])
							dist_locs[9].append((i,j))
						elif len(district_lists[8]) != 10:
							district_lists[8].append(data[i][j])
							dist_locs[8].append((i,j))
						elif len(district_lists[7]) != 10:
							district_lists[7].append(data[i][j])
							dist_locs[7].append((i,j))
						elif len(district_lists[6]) != 10:
							district_lists[6].append(data[i][j])
							dist_locs[6].append((i,j))
						elif len(district_lists[5]) != 10:
							district_lists[5].append(data[i][j])
							dist_locs[5].append((i,j))
						elif len(district_lists[4]) != 10:
							district_lists[4].append(data[i][j])
							dist_locs[4].append((i,j))
						elif len(district_lists[3]) != 10:
							district_lists[3].append(data[i][j])
							dist_locs[3].append((i,j))
						elif len(district_lists[2]) != 10:
							district_lists[2].append(data[i][j])
							dist_locs[2].append((i,j))
						elif len(district_lists[1]) != 10:
							district_lists[1].append(data[i][j])
							dist_locs[1].append((i,j))
						elif len(district_lists[0]) != 10:
							district_lists[0].append(data[i][j])
							dist_locs[0].append((i,j))

	dist_positions(district_lists, dist_locs)
	return district_lists

#code and methods inspired from http://katrinaeg.com/simulated-annealing.html

def simulatedAnnealing(solution):
	old_cost = fitnessFunction(solution)
	T = 1.0
	T_min = 0.00001
	alpha = 0.9

	while T > T_min:
		i = 1
		while i <= 100:
			new_solution = generateSolution(solution)
			new_cost = fitnessFunction(new_solution)
			ap = acceptanceProbablity(new_cost, old_cost, T)
			if ap > random():
				solution = new_solution
				old_cost = new_cost
			i += 1
			update_global()
		T = T*alpha
	return solution, old_cost


def acceptanceProbablity(new, old, t):
	ap = math.exp((old - new) / t)
	return ap


def fitnessFunction(districts):
	#do some math here to shoot back to simulatedAnnealing
	d_majority = 0
	r_majority = 0
	equal = 0

	for i in xrange(len(districts)):
		d_count = 0
		r_count = 0
		for j in xrange(len(districts)):
			if districts[i][j] == 'D':
				d_count = d_count + 1
			else:
				r_count = r_count + 1
			if d_count < r_count:
				r_majority = r_majority + 1
			elif d_count > r_count:
				d_majority = d_majority + 1
			else:
				equal = equal + 1
		return abs(d_majority - r_majority)


def update_global():
	global search_state_iterations
	search_state_iterations = search_state_iterations + 1

def districtMajority(districts_random):
	district_div_D = 0
	district_div_R = 0

	for i in xrange(len(districts_random)):
		D_tots = 0
		R_tots = 0
		for j in xrange(len(districts_random)):
			if districts_random[i][j] == 'D':
				D_tots = D_tots + 1
			else:
				R_tots = R_tots + 1
		if D_tots > R_tots:
			district_div_D = district_div_D + 1
		elif D_tots < R_tots:
			district_div_R = district_div_R + 1
	return district_div_D, district_div_R

def partyDivision():
    R = 0
    D = 0
    total = 0
    percent_R = 0
    percent_D = 0
    data = {}
    district_const = []
    for lines in content:
    	data = lines.strip('[').strip('\n').strip('\r').strip(']').split(' ')
    	district_const.append(data)
    	R = R + lines.count("R")
    	D = D + lines.count("D")
    total = D + R
    percent_R = float(R)/float(total)
    percent_D = float(D)/float(total)
    const_value = percent_D - percent_R
    return (percent_R, percent_D, district_const)

if __name__ == '__main__':
	file_data = open(sys.argv[1])
	content = file_data.readlines()
	search_state_iterations = 0
	global_locs = []
	(rabbit, dragon, data) = partyDivision()
	(found_dist, fitness_tot) = simulatedAnnealing(generateSolution(data))



	script, filename = argv
	with open(filename) as file:
		array2d = [[char for char in line.split()] for line in file]

	population_size  = len(array2d) * len(array2d[0])
	district_length = population_size/len(array2d)
	


	print("Party division in population:")
	print("*************************************")
	print("R:"), rabbit
	print("D:"), dragon
	print("*************************************")
	print("")

	(major_d, major_r) = districtMajority(found_dist)

	print("Number of districts with a majority for each party:")
	print("*************************************")
	print("R:"), major_d
	print("D:"), major_r
	print("*************************************")
	print("")
	print('Locations Assigned to Each District')

	print("District 1: "), global_locs[0]
	print("District 2: "), global_locs[1]
	print("District 3: "), global_locs[2]
	print("District 4: "), global_locs[3]
	print("District 5: "), global_locs[4]
	print("District 6: "), global_locs[5]
	print("District 7: "), global_locs[6]


	if district_length == 8:

		print("District 8: "), global_locs[7]

	if district_length == 10:

		print("District 8: "), global_locs[7]
		print("District 9: "), global_locs[8]
		print("District 10: "), global_locs[9]


	print('*****************************'+ '\n')


	print('*****************************'+ '\n')
	print('Algorithm Applied:' + ' ' + 'SA')
	print('*****************************'+ '\n')
	print('*****************************'+ '\n')
	print("Number of search states explored:"), search_state_iterations
	print("*************************************")
	print("")