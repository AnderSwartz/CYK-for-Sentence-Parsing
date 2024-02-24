import collections

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
        if(count>2):
            # print(rules_lookup)
            # print(rule_counts)
            break
        sample_tree = trees.Tree.from_str(tree)
        # print(sample_tree.pretty_print())
        for node in sample_tree.bottomup():
            # print(node)
            # print()
            if len(node.children)>0:
                # print(node,"'s children are:")
                
                children = node.children
                
                new_RHS_list = []
                if len(children)>1:
                    for child in children:
                        new_RHS_list.append(child.label)
                    new_RHS = tuple(new_RHS_list)
                else:
                    new_RHS = children[0].label
                new_LHS = node.label
                
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
                    # print(new_LHS)
                #     print(rules_lookup.get(new_RHS))
                    old_LHS = rules_lookup[new_RHS]#should be a tuple
                    print("More than one LHS produces this LHS")
                    print("old and new LHS:",old_LHS,new_LHS)
                    print("RHS:",new_RHS)
                    # print(old_LHS)
                    if(new_LHS not in old_LHS):
                        print(new_LHS)
                        print(old_LHS)
                        old_LHS = {old_LHS}
                        print(old_LHS)
                        old_LHS.add(new_LHS)
                        new_LHS = old_LHS
                        print(new_LHS)
                        print()
                #         new_LHS = tuple(new_LHS)
                        # print(new_LHS)
                #     # new_LHS = tuple(old_LHS_list)
                    new_LHS = tuple(new_LHS)
                    rules_lookup[new_RHS] = new_LHS
                # except:
                #     new_LHS = tuple(new_LHS)
                #     rules_lookup[new_RHS] = new_LHS
                # else:
                
                print(new_LHS)
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
    rule_probabilities = {}
    for rhs in rules_lookup.keys():
        print(rhs,rules_lookup[rhs],len(rules_lookup[rhs]))
        # for rule in rules_lookup[rhs]:
            # print(rule)
            
        
    return grammar

grammar = get_probs(rules_lookup, rule_counts)
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
    chart = sample_vars[3]
    return chart
    #chart[0][-1] is top right
    #dictionary stored with 
    #weight, rule, i, x, diagnoals = chart[0][-1]["TOP"]
    # left child chart[]

sent = sample_vars[4]
chart = CKY(sent, grammar)
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
    parse_string = sample_vars[5]
    return parse_string
    #handle 3 scenarios, you make two children, 1 or none
    # can print parse_string with pretty.print

parse_string = parse(chart)
''' Format of parse_string matches train.trees'''

# with open("train.trees.pre.unk") as file:
#     line = file.readline()
#     # print(line)
#     sample_tree = trees.Tree.from_str(line)
#     print(sample_tree.pretty_print())
