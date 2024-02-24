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
    # Replace this with your code, for now I put a placeholder in
    #rules_lookup = sample_vars[0] #when you do this with "<unk>", there will be 15-20 options after training
    #rules_lookup can also take in two terminal 
    #rule_counts = sample_vars[1]
    rules_lookup = {}
    rule_counts = collections.Counter()
    count = 0
    for tree in open("train.trees.pre.unk"):
        # if(count>2):
        #     # print(rules_lookup)
        #     # print(rule_counts)
        #     break
        sample_tree = trees.Tree.from_str(tree)
        # print(sample_tree.pretty_print())
        for node in sample_tree.bottomup():
            # print(node)
            # print()
            if len(node.children)>0:
                # print(node,"'s children are:")
                
                children = node.children
                new_LHS = node.label
                new_RHS_list = []
                if len(children)>1:
                    for child in children:
                        new_RHS_list.append(child.label)
                    new_RHS = tuple(new_RHS_list)
                    
                else:
                    new_RHS = children[0].label
                rule_counts[new_LHS,new_RHS] +=1
                # rules_lookup[new_RHS] = list(rules_lookup[new_RHS]),new_LHS
                # try:
                #     old_LHS = (rules_lookup.get(new_RHS))
                #     print(old_LHS)
                #     new_LHS = old_LHS +","+ new_LHS
                #     # new_LHS = tuple(old_LHS_list)
                #     new_LHS = tuple(new_LHS)
                #     rules_lookup[new_RHS] = new_LHS
                # except:
                #     new_LHS = tuple(new_LHS)
                #     rules_lookup[new_RHS] = new_LHS
                #could have used a dictionary for the values of the rules_lookup 
                #but i didnt have time?
                if rules_lookup.get(new_RHS) != None:
                    old_LHS = rules_lookup.get(new_RHS)
                    if(new_LHS not in old_LHS):
                        new_LHS = old_LHS, new_LHS
                #     # new_LHS = tuple(old_LHS_list)
                #     new_LHS = tuple(new_LHS)
                #     rules_lookup[new_RHS] = new_LHS
                # except:
                #     new_LHS = tuple(new_LHS)
                #     rules_lookup[new_RHS] = new_LHS
                rules_lookup[new_RHS] = new_LHS
                # print(type(rules_lookup.get(new_RHS)))
                # tuple(list(rules_lookup[new_RHS]) + new_LHS)
                
                # return rules_lookup, rule_counts

                #each node is labeled with...
        count+=1
    return rules_lookup, rule_counts

fname = 'train.trees.pre.unk'
rules_lookup, rule_counts = read_trees(fname)
# print(rule_counts['NP',('DT','NN')])


''' Format of rules_lookup provided is:
    key = RHS of rule
    value = set of valid LHS of rules
    For example:
    rules_lookup['the'] = {'DT'} corresponds to rule DT -> the
    rules_lookup['stop'] = {'VBP','NN'} corresponds to rules stop -> VBP and stop -> NN'''
    

''' Format of rule_counts provided is:
    key = rule as tuple of (LHS, RHS)
    value = how many times that rule was seen in the data
    For example:
    rule_counts[('PP', ('IN', 'NP_NNP'))] = 4
    rule_counts[('DT', 'this')] = 1'''

def get_probs(rules_lookup, rule_counts):
    """Given the rules_lookup dictionary and the total count of rules
    return a dictionary with keys as rules and values as probabilities"""
    grammar = {}
    

    for rule in rule_counts.keys():
        numerator = rule_counts[rule]
        # print(rule,": ", numerator)
        # print(rule[1])
        LHS = rule[0]
        RHS = rule[1]
        # print(rhs,rules_lookup[rhs],len(tuple(rules_lookup[rhs])))
        denom = 0 #total instances of this LHS going to any RHS
        for potential_matching_LHS_rule in rule_counts.keys():
            if(potential_matching_LHS_rule[0]==LHS):
                # print(rule,": ",rule_counts[rule])
                denom += rule_counts[potential_matching_LHS_rule]
        # print(numerator)
        # print(denom)
        # print(rule)
        grammar[rule] = numerator/denom

        # print(grammar[rule])
        # print(grammar)

            
        
    return grammar

grammar = get_probs(rules_lookup, rule_counts)
# for rule in grammar:
#     # print(rule)
#     if rule[0] == "TOP":
#         print(rule)

# print(rules_lookup["TOP"])
# all_NNP_rules = {}
# for rule in grammar:
#     if rule[0]=="NNP":
#         all_NNP_rules[rule] = grammar[rule]  
# top_NNP_rules = sorted(all_NNP_rules.items(),key=lambda x:x[1],reverse=True)
# print(top_NNP_rules[:5])


# for item in grammar:
#     print(item,": ",grammar[item])
# print(grammar['NN','stop'])
# print(grammar['VBP','stop'])
# print(grammar[('NP',('DT', 'NNS'))])
''' Format of grammar provided is:
    key = rule as tuple of (LHS, RHS)
    value = conditional probability of RHS given LHS
    For example:
    grammar[('PP', ('IN', 'NP_NNP'))] = 0.66666666
    grammar[('DT', 'this')] = 0.25'''



def CKY(sent, grammar):
    """Given a space separated sentence and your grammar,
    run CKY to fill the chart with the highest probability partial parses.
    Return the filled in chart from CKY"""
    sent_split = sent.split(' ')
    # print(sent_split)
    # print(len(sent_split))
    chart = np.empty_like((len(sent),len(sent)))
    chart_size = len(sent_split)
    chart = [[{} for x in range(chart_size)] for y in range(chart_size)]
    # print(chart)
    for i in range(len(sent_split)):
    #chart[i][i] = ????
        # chart[i][i] = {}
        for rule in grammar:
            if rule[1] == sent_split[i]: # matching RHS of rule
                chart[i][i][rule[0]]  = grammar[rule],rule[1],i,None,None # set cell to LHS
                # grammar[rule],rule[1],i,None,None # set cell to LHS
                # print(rule[0])
        # print("for word",sent_split[i],"assigned",chart[i][i])
        if chart[i][i] == {}:
            # print("no rule for",sent_split[i])

            for rule in grammar:
                if rule[1] == "<unk>": 
                    chart[i][i][rule[0]]  = grammar[rule],"<unk>",rule[1],i,None,None
            # print("for word",sent_split[i],"assigned",chart[i][i])
            #new loop that checks for all possibe rules = LHS, <unk> = RHS
    # for line in chart:
        # print(line)
    for diagonals in range(1,len(sent_split)):
        for i in range(len(sent_split) - diagonals):
            #print('cell to fill:', i, i+diagonals)
            for x in range(diagonals): # how many comparisons
                #print('\t','merge from: ',(i,i+x),(i+1+x,i+diagonals))
                target = []
                target1 = list(chart[i][i+x].keys())
                target2 = list(chart[i+1+x][i+diagonals].keys())
                # target.append(chart[i+1+x][i+diagonals].keys())
                # target = tuple(target)
                # RHS = target
                # chart[i][i+diagonals] = {}
                # print("TARGET1:",target1)
                # print("TARGET2:",target2)

                for rule in grammar:
                    #print("rule RHS and our RHS",rule[1], RHS)
                    for RHS1 in target1:
                        for RHS2 in target2:
                            RHS = []
                            RHS.append(RHS1)
                            RHS.append(RHS2)
                            RHS = tuple(RHS)
                            # print("looking for rule w/ RHS = ",RHS)
                            if rule[1] == RHS:
                        #print("We matched: ", rule[0])
                                previous_score_left = chart[i][i+x][RHS1][0]
                                previous_score_down = chart[i+1+x][i+diagonals][RHS2][0]
                                new_score = previous_score_left * previous_score_down * grammar[rule]
                                current_tags = chart[i][i+diagonals]
                                new_tag = rule[0]
                                if rule[0] in chart[i][i+diagonals]:
                                    # print("same tag from different path")
                                    if new_score > chart[i][i+diagonals].get(rule[0][0],0):
                                        chart[i][i+diagonals][rule[0]] = new_score,rule[1],i,x,diagonals
                                        # print(rule[0])
                                else: 
                                    chart[i][i+diagonals][rule[0]] = new_score,rule[1],i,x,diagonals
                                    # print(rule[0])
                        # grammar[rule],rule[1],i,None,None
                #print("our RHS", RHS)
    # weight, rule, i, x, diagonals = chart[0][-1]['TOP']
    # print('Weight of this parse ',weight)
    # print('Rule applied to get here ', rule)
    # print('Left child entry ', chart[i][i+x])
    # print('Right child entry ', chart[i+x+1][i+diagonals])
    # for line in chart:
    #     print(line)
    # print()
    # print()

    # print()

    return chart

    #chart[0][-1] is top right
    #dictionary stored with 
    #weight, rule, i, x, diagnoals = chart[0][-1]["TOP"]
    # left child chart[]

    
''' Format of chart provided is:
    chart[row][column] = dictionary with:
        key = parse for that span (LHS of rule applied)
        value = [weight, RHS of rule applied, index of first word  of span (i), split index (x), diagonal # (diagonals)]
    For example:
    chart[0][0]['VBZ'] = [1.0, 'Does', 0, None, None]
    weight is 1.0, RHS of rule is Does, comes from word 0, there is no split, there is no diagonal #
    
    chart[0][-1]['TOP'] = [0.000992063492063492, ('SQ', 'PUNC'), 0, 4, 5]
    weight is 0.00099, RHS of rule is ('SQ', 'PUNC'), starts at word 0, splits after word index 4, 5th diagonal processed
    '''

def parse(chart, s='TOP', row=0, col=-1):
    """Given the following:
    chart - filled in chart from CKY
    s - used for recursion, starts with 'TOP'
    row - the index of the row that i appears in
    col - the index of the column that s appears in
    This will be recursive, with s row and col changing"""
    #case 1:

    # if s not in chart[row][col].keys():
    #     return "---"
    # print(len(chart[0]))
    # print("new ARI")
    # print(s)
    children = (chart[row][col][s][1])
    
    # print(children)

    # children = list(children)
    # print(children)
    # print(len(children)) 


    
    if type(children)==str:
        # print("none left at:",s)
        return s,children
    if len(children)==1:
        child_1_row = row
        child_1_col = chart[row][col][s][3]
        return parse(chart,children[0],child_1_row,child_1_col)
    # print("len>1 for",s)
    child_1_row = row
    child_1_col = col-(chart[row][col][s][4]-chart[row][col][s][3])
    child_2_row = row+1+chart[row][col][s][3]
    child_2_col = col
    child1 = (chart[child_1_row][child_1_col])
    child2 = (chart[child_2_row][child_2_col])

    # chart[row][col][s][3]

    

    # print(s)
    # print(children)
    if len(children)==2:
        
        return s,parse(chart,children[0],child_1_row,child_1_col), parse(chart,children[1],child_2_row,child_2_col)
    if len(children)==1:
        return s,parse(children)
    #handle 3 scenarios, you make two children, 1 or none
    # can print parse_string with pretty.print



###############################################################################

# print("test")
# with open("dev.strings") as file:
#     # sent = sample_vars[4]
#     sent = file.readline()
#     # sent = file.readline()
#     # sent = file.readline()
#     # sent = file.readline()
#     # sent = file.readline()
#     # sent = file.readline()
#     # sent = file.readline()
#     # sent = file.readline()
#     # sent = file.readline()
#     # sent = file.readline()
#     # sent = file.readline()
#     # sent = file.readline()
#     #problem! not handling unknowns 









#     # print(sent)
#     sent = sent.strip()
#     print(sent)
#     chart = CKY(sent, grammar)
#     # print()
#     # print()

#     # for line in chart:
#     #     print(line)
#     # print(chart)

#     parse_string = parse(chart)
#     ''' Format of parse_string matches train.trees'''

#     parse_string = str(parse_string)
#     parse_string = parse_string.replace("'","")
#     parse_string = parse_string.replace(",","")
#     print(parse_string)
#     # print(parse_string)
#     # (TOP (SQ (VBZ Does) (SQ* (NP (DT this) (NN flight)) (VP (VB serve) (NP_NN dinner)))) (PUNC ?))
#     sample_tree = trees.Tree.from_str(parse_string)
#     print(sample_tree.pretty_print())

# for sent in open("dev.strings"):
#     # print(sent)
#     # sent = sample_vars[4]
#     # sent = file.readline()
#     # sent = file.readline()
#     # sent = file.readline()
#     # sent = file.readline()
#     # sent = file.readline()
#     # sent = file.readline()
#     # sent = file.readline()
#     # sent = file.readline()
#     # sent = file.readline()
#     # sent = file.readline()
#     # sent = file.readline()
#     # sent = file.readline()
#     #problem! not handling unknowns 









#     # print(sent)
#     sent = sent.strip()
#     # print(sent)
#     # print()
#     # print()

#     # for line in chart:
#     #     print(line)
#     # print(chart)
#     try:
#         chart = CKY(sent, grammar)
#         parse_string = parse(chart)
#         parse_string = str(parse_string)
#         parse_string = parse_string.replace("'","")
#         parse_string = parse_string.replace(",","")
#         print(parse_string)
#     except:
#         pass
#     ''' Format of parse_string matches train.trees'''

#     # parse_string = str(parse_string)
#     # parse_string = parse_string.replace("'","")
#     # parse_string = parse_string.replace(",","")
#     # print(parse_string)
#     # print(parse_string)
#     # (TOP (SQ (VBZ Does) (SQ* (NP (DT this) (NN flight)) (VP (VB serve) (NP_NN dinner)))) (PUNC ?))
#     # sample_tree = trees.Tree.from_str(parse_string)
#     # print(sample_tree.pretty_print())

# # with open("train.trees.pre.unk") as file:
# #     line = file.readline()
# #     # print(line)
# #     sample_tree = trees.Tree.from_str(line)
# #     print(sample_tree.pretty_print())



# for sent in open("dev.strings"):
#     # print(sent)
#     # sent = sample_vars[4]
#     # sent = file.readline()
#     # sent = file.readline()
#     # sent = file.readline()
#     # sent = file.readline()
#     # sent = file.readline()
#     # sent = file.readline()
#     # sent = file.readline()
#     # sent = file.readline()
#     # sent = file.readline()
#     # sent = file.readline()
#     # sent = file.readline()
#     # sent = file.readline()
#     #problem! not handling unknowns 









#     # print(sent)
#     sent = sent.strip()
#     # print(sent)
#     # print()
#     # print()

#     # for line in chart:
#     #     print(line)
#     # print(chart)
#     try:
#         chart = CKY(sent, grammar)
#         parse_string = parse(chart)
#         parse_string = str(parse_string)
#         parse_string = parse_string.replace("'","")
#         parse_string = parse_string.replace(",","")
#         print(parse_string)
#     except:
#         print()
#         pass



#   #     max = 0
#     #     for top in chart[row][col]:
#     #         if chart[row][col][top][0] > max:
#     #             max = chart[row][col][top][0]
#     #             maxTags = chart[row][col][top]