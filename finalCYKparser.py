import collections
from posixpath import split
import numpy as np
from numpy import dtype
import trees
import pickle
import sys
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
with open('sample.vars','rb') as f:
    # These sample variables are the result of treating the first 3 sentences as our entire training data
    sample_vars = pickle.load(f)
    #inputs and outputs to first three sentences. Use to check

def read_trees(fname):
    """Read in all trees from a given file
    input: filename
    outputs: rule lookup dictionary, count of each rule seen"""
  
    rules_lookup = {}
    rule_counts = collections.Counter()
    for tree in open(fname):
        sample_tree = trees.Tree.from_str(tree)
        # print(sample_tree.pretty_print())
        for node in sample_tree.bottomup():
            if len(node.children)>0:
                children = node.children
                new_LHS = node.label
                new_RHS_list = []
                if len(children)>1:
                    for child in children:
                        new_RHS_list.append(child.label)
                    new_RHS = tuple(new_RHS_list,)
                else:
                    new_RHS = children[0].label
                rule_counts[new_LHS,new_RHS] +=1
               
                if rules_lookup.get(new_RHS) != None:
                    old_LHS = rules_lookup.get(new_RHS)
                    if(new_LHS not in old_LHS):
                        new_LHS = old_LHS, new_LHS
                
                rules_lookup[new_RHS] = new_LHS

    #normaling formatting so every rule is a set
    for RHS in rules_lookup:
        # print(type(rules_lookup[RHS]))
        if isinstance(rules_lookup[RHS],str):
            # print(rules_lookup[RHS])
            rules_lookup[RHS] = set((rules_lookup[RHS],))
        else:
            rules_lookup[RHS] = set(rules_lookup[RHS])
        
    return rules_lookup, rule_counts

fname = "train.trees.pre.unk"
rules_lookup, rule_counts = read_trees(fname)

# Answers to 2.1:
# print("Number of unique rules: ",len(rule_counts))
# top_5_rule_counts = sorted(rule_counts.items(), key=lambda pair: pair[1], reverse=True)
# print("Top 5 most frequent rules in training: ",top_5_rule_counts[:5])

def get_probs(rules_lookup, rule_counts):
    """Given the rules_lookup dictionary and the total count of rules
    return a dictionary with keys as rules and values as probabilities"""
    grammar = {}
    for rule in rule_counts.keys():
        numerator = rule_counts[rule]
        LHS = rule[0]
        RHS = rule[1]
        denom = 0 #total instances of this LHS going to any RHS
        for potential_matching_LHS_rule in rule_counts.keys():
            if(potential_matching_LHS_rule[0]==LHS):
                # print(rule,": ",rule_counts[rule])
                denom += rule_counts[potential_matching_LHS_rule]
        grammar[rule] = np.log2(numerator/(denom))
    return grammar

grammar = get_probs(rules_lookup, rule_counts)

# Answers to 2.2:
# all_NNP_rules = {}
# for rule in grammar:
#     if rule[0]=="NNP":
#         all_NNP_rules[rule] = grammar[rule]  
# top_NNP_rules = sorted(all_NNP_rules.items(),key=lambda x:x[1],reverse=True)
# for rule in top_NNP_rules[:5]:
#     print(list(rule[0])[0],"->",list(rule[0])[1],"#",rule[1])

def CKY(sent, grammar):
    """Given a space separated sentence and your grammar,
    run CKY to fill the chart with the highest probability partial parses.
    Return the filled in chart from CKY"""
    sent_split = sent.split(' ')
    chart = np.empty_like((len(sent),len(sent)))
    chart_size = len(sent_split)
    chart = [[{} for x in range(chart_size)] for y in range(chart_size)]
    indexOfUnk = []
    for i in range(len(sent_split)):
        for rule in grammar:
            if rule[1] == sent_split[i]: # matching RHS of rule
                chart[i][i][rule[0]]  = grammar[rule],rule[1],i,None,None # set cell to LHS
        if chart[i][i] == {}:
            for rule in grammar:
                #new loop that checks for all possibe rules = LHS, <unk> = RHS
                if rule[1] == "<unk>": 
                    chart[i][i][rule[0]]  = grammar[rule],"<unk>",rule[1],i,None,None
                    indexOfUnk.append(i)
    for diagonals in range(1,len(sent_split)):
        for i in range(len(sent_split) - diagonals):
            for x in range(diagonals): # how many comparisons
                target = []
                target1 = list(chart[i][i+x].keys())
                target2 = list(chart[i+1+x][i+diagonals].keys())
                locationToFill = [i,i+diagonals]
                checkingLocations = [(i,(i+x)),(((i+1+x),(i+diagonals)))]
                for rule in grammar:
                    for RHS1 in target1:
                        for RHS2 in target2:
                            RHS = []
                            RHS.append(RHS1)
                            RHS.append(RHS2)
                            RHS = tuple(RHS)
                            if rule[1] == RHS:
                                previous_score_left = chart[i][i+x][RHS1][0]
                                previous_score_down = chart[i+1+x][i+diagonals][RHS2][0]
                                new_score = previous_score_left + previous_score_down + grammar[rule]
                                current_tags = chart[i][i+diagonals]
                                new_tag = rule[0]
                                if rule[0] in chart[i][i+diagonals]: #"same tag from different path"
                                    try:
                                        previous_score = chart[i][i+diagonals][rule[0]][0]
                                    except:
                                         previous_score = 0
                                    if new_score > previous_score:#"only keep if this path has higher score"
                                        chart[i][i+diagonals][rule[0]] = new_score,rule[1],i,x,diagonals
                                else: 
                                    chart[i][i+diagonals][rule[0]] = new_score,rule[1],i,x,diagonals
    try:
        top_score = chart[0][-1]["TOP"][0]
    except:
        top_score = 0
    return chart, top_score
    
def parse(chart, s='TOP', row=0, col=-1):
    """Given the following:
    chart - filled in chart from CKY
    s - used for recursion, starts with 'TOP'
    row - the index of the row that i appears in
    col - the index of the column that s appears in
    This will be recursive, with s row and col changing"""
    #case 1:
    if s not in chart[row][col].keys(): #no top
        # print("none found")
        return ""
    children = (chart[row][col][s][1])
    #case 2:
    if type(children)==str: #only 1 child (1 RHS element)
        return s,children
    #case 3: multiple children (RHS elements)
    #first tag coordinates in chart
    child_1_row = row
    child_1_col = col-(chart[row][col][s][4]-chart[row][col][s][3])
    #second tag coordinates in chart
    child_2_row = row+1+chart[row][col][s][3]
    child_2_col = col

    return s,parse(chart,children[0],child_1_row,child_1_col), parse(chart,children[1],child_2_row,child_2_col)

#my function takes in either dev.strings or test.strings,
#  and a boolean representing whether you want to print the scores of the first 5 parses
def test(testingFile,printFirstFiveScores):
    sentence_count = 0
    if printFirstFiveScores:
        devFirstFive = open("devFirst5Scores.txt","a") #stores scores for 3.1
        devFirstFive.truncate(0)
    for sent in open(testingFile):
        sent = sent.strip()
        if printFirstFiveScores:
            if sentence_count < 5:
                chart, top_score = CKY(sent, grammar)
                devFirstFive.write(str(top_score))
                devFirstFive.write("\n")
        try:
            chart, top_score = CKY(sent, grammar)
            parse_string = parse(chart)
            parse_string = str(parse_string)
            parse_string = parse_string.replace("'","")
            parse_string = parse_string.replace(",","")
            print(parse_string) 
        except:
            pass
        sentence_count +=1

# test("dev.strings",True)       # <--uncomment to run on dev.
test("test.strings",False)


