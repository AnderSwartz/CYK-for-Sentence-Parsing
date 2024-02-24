import trees
import sys
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
# with open("train.trees.pre.unk") as file:
#     line = file.readline()
#     # print(line)
#     sample_tree = trees.Tree.from_str(line)
#     print(sample_tree.pretty_print())

# s = "(TOP (S (NP (DT The) (NN flight)) (VP (MD should) (VP (VB be) (NP (NP (CD eleven) (RB a.m)) (NP (NN tomorrow)))))) (PUNC .))"
# s2 = "(TOP (SBARQ (WHNP (WHNP (WDT Which)) (PP (IN of) (NP (DT these)))) (SQ (VP (VBP serve) (NP (NN dinner))))) (PUNC ?))"
s3 = "(TOP (S (NP_PRP I) (VP (VBP need) (NP (NP (DT a) (NN flight)) (NP* (PP (TO to) (NP_NNP Seattle)) (NP* (VP (VBG leaving) (PP (IN from) (NP_NNP Baltimore))) (VP (VBG <unk>) (NP (NP (DT a) (NN stop)) (PP (IN in) (NP_NNP Minneapolis))))))))) (PUNC .))"
sample_tree = trees.Tree.from_str(s3)
print(sample_tree.pretty_print())

# for line in open("dev.trees"):
#     sample_tree = trees.Tree.from_str(line)
#     print(sample_tree.pretty_print())

