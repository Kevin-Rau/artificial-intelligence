#Kevin Rau
#101289616
#Assignment 8 AI 3202
#collaborated with Jesus, Ryan, and Chase
#Various sources were used for some code given through 
#http://verbs.colorado.edu/~mahu0110/teaching/ling5832/5832-hw3.html as well

from __future__ import division




#Use the same processs that was given in the write-up to process the file
def processFile(filename):
	lines = []
	lines.append("SSSS")

	with open(filename, "r") as sentences:

			for line in sentences:
				if(line=="\n"):
					lines.append("EEEE")
					lines.append("SSSS")
				else:
					lines.append(line.rstrip())

	lines.append("EEEE")

	data = lines[:len(lines) - 2]
	return data

#create a class to accummulate tags and keep track of thier types
class Tags():
	tag_type = ""
	count = 0


def transitionProbability(lines):
	tags = {}
	tag_types = []

	#identify the tags in the file and create the tags array
	for line in lines:
		if "\t" in line:
			index = line.index("\t") + 1
			tag = line[index:]
		else:
			tag = line

		#How many tags have been accounted for
		if tag not in tags:
			new_tag = Tags()
			new_tag.tag_type = tag
			new_tag.count = 1
			tags[tag] = new_tag
		else:
			tags[tag].count += 1

		#add new tag type
		if tag not in tag_types:
			tag_types.append(tag)



	for tag in tags:
		tag_initial = tags[tag]
		for tag_type in tag_types:
			setattr(tag_initial, tag_type, 0)

	# count of (TAGi TAGj)
	for x in range (1, len(lines)):
		start = lines[x - 1]
		next = lines[x]
		if "\t" in start:
			index = start.index("\t") + 1
			initial_tag = start[index:]
		else:
			initial_tag = start

		if "\t" in next:
			index = next.index("\t") + 1
			next_tag = next[index:]
		else:
			next_tag = next

		current = tags[initial_tag]
		setattr(current, next_tag, getattr(current, next_tag) + 1)

	return tags, tag_types
	#returns a dictonary of tags and the list of tag names associated with the objects
def emissionProbablity(lines, tags):
	observations = {}
	words = []
	states = []


	for line in lines:
		if "\t" in line:
			index = line.index("\t") + 1
			observ = line[:index - 1]
			state = line[index:]
			if observ not in words:
				words.append(observ)

		else:
			observ = line
			state = line
			if observ not in words:
				words.append(observ)


		if state not in observations:
			observations[state] = {}

		if state not in states:
			states.append(state)

	#calculate the count (word and TAG)
	for line in lines:
		if "\t" in line:
			index = line.index("\t") + 1
			observ = line[:index -1]
			state = line[index:]

		else:
			observ = line
			state = line 

		initial_state = observations[state]
		if observ not in initial_state:
			initial_state[observ] = 1
		else:
			initial_state[observ] += 1
			#same as above, return dictionary of dictionay of tags and list of words associated with
	return observations, words

def viterbi(obs, states, start_p, trans_p, emit_p):
	# For each state, establish the initial probability of occupying a state
    #V is a list of dictionaries. Each dictionary is a time
    #has a dictionary of states
	V = [{}]

	#emitp is the pr(evidence|state)
    #obs is evidence at each time
    #obs[0] = normal
	for st in states:
		try:
			emission_prob = emit_p[st][obs[0]]
		except KeyError:
			emission_prob = 0.0
		V[0][st] = {"prob": start_p[st] * emission_prob, "prev": None}
	    # Run Viterbi when t > 0
	
	for t in range(1, len(obs)):
		V.append({})
		for st in states:
		    #V[t-1][prev_st]["prob"] is the probability of being in state prev_st t-1
            #calculated previous time through loop
            #trans_p[prev_st][st] is transition probabilities 

            # for each state, determine which state is most likely to be transitioned to from the evidence available
			max_tr_prob = max(V[t-1][prev_st]["prob"]*trans_p[prev_st][st] for prev_st in states)
			for prev_st in states:
			# if the evidence for the candidate state is greatest, this is the new most likely state
				if V[t-1][prev_st]["prob"] * trans_p[prev_st][st] == max_tr_prob:
					try:
						emission_prob = emit_p[st][obs[t]]
					except KeyError:
						emission_prob = 0.0
					#emit_p[st][obs[t]] is emission probability of seeing observation in this state
                    #obs[t] is observation at time t 
					max_prob = max_tr_prob * emission_prob
                    #Store V for time t in state st
					V[t][st] = {"prob": max_prob, "prev": prev_st}
					break

	opt = []
    # The highest probability
	max_prob = max(value["prob"] for value in V[-1].values())
	previous = None

	# Get most probable state and its backtrack
	for st, data in V[-1].items():
		if data["prob"] == max_prob:
			opt.append(st)
			previous = st
			break

	# Follow the backtrack till the first observation
	for t in range(len(V) - 2, -1, -1):
		opt.insert(0, V[t + 1][previous]["prev"])
		previous = V[t + 1][previous]["prev"]

	opt = opt[1:len(opt) - 1]

	return 'The steps of states are: \n' + "-->\t" + ' '.join(opt)

def dptable(V):
	yield " ".join(("%12d" % i) for i in range(len(V)))
	for state in V[0]:
		yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)

'''
if __name__ == "__main__":
    states = ('Healthy', 'Fever')
    observations = ('normal', 'cold', 'dizzy')
    start_probability = {'Healthy': 0.6, 'Fever': 0.4}
    transition_probability = {
        'Healthy' : {'Healthy': 0.7, 'Fever': 0.3},
        'Fever' : {'Healthy': 0.4, 'Fever': 0.6}
    }

    emission_probability = {
        'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
        'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6}
    }
    viterbi(observations, states, start_probability, transition_probability, emission_probability)
'''

if __name__ == "__main__":

	#Input a sentence
	input_sentence  = raw_input("Type a sentence: ")
	print "...working"
	print "\n"
	lines = processFile("penntree.tag")
	tags, tag_types = transitionProbability(lines)
	observ, words = emissionProbablity(lines, tag_types)
	states = []
	for tag in tag_types:
		states.append(tag)
	observations = []
	observations.append("SSSS")
	word = input_sentence.split()
	for x in range(0, len(word) -1):
		observations.append(word[x])
	observations.append(word[-1][:len(word[-1])-1])
	observations.append(word[-1][-1])
	observations.append("EEEE")

	probability_check = {}
	for state in states:
		if state != "SSSS":
			probability_check[state] = 0.0
		else:
			probability_check[state] = 1.0

	transition_probability = {}
	for state in states:
		current_state = tags[state]
		transition_probability[current_state.tag_type] = {}
		for tag in tag_types:
			get_count = getattr(current_state, tag)
			total = current_state.count
			transition_probability[current_state.tag_type][tag] = get_count/total

	for obv in observ:
		state_total = tags[obv].count
		for word in observ[obv]:
			observ[obv][word] /= state_total
	emission_probability = observ


	print viterbi(observations, states, probability_check, transition_probability, emission_probability)