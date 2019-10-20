from numpy import prod
from nltk.corpus import wordnet as wn

def fuzzy_set_membership_measure(x,A,m):

    return 0 if A == [] else max(map(lambda a: m(x,a), A))


def score_it(A,T,m):

    if A == [] and T == []:
        return 1

    score_left = 0 if A == [] else prod(list(map(lambda a: m(a,T), A)))
    score_right = 0 if T == [] else prod(list(map(lambda t: m(t,A), T)))

    return min(score_left, score_right)


def compute_accuracy(A,T):
    l = len(A)
    return sum([A[i]==T[i] for i in range(l)]).item()/l


def wup_measure(a,b,similarity_threshold = 0.0):

    def get_semantic_field(a):
        weight = 1.0
        semantic_field = wn.synsets(a,pos=wn.NOUN)
        return (semantic_field, weight)

    def get_stem_word(a):
        weight = 1.0
        return (a,weight)

    global_weight = 1.0

    (a, global_weight_a) = get_stem_word(a)
    (b, global_weight_b) = get_stem_word(b)
    global_weight = min(global_weight_a,global_weight_b)

    if a == b:
        return 1.0*global_weight

    if a == [] or b == []:
        return 0

    interp_a,weight_a = get_semantic_field(a)
    interp_b,weight_b = get_semantic_field(b)

    if interp_a == [] or interp_b == []:
        return 0

    global_max = 0

    for x in interp_a:
        for y in interp_b:
            local_score = x.wup_similarity(y)
            if local_score > global_max:
                global_max = local_score

    if global_max < similarity_threshold:
        interp_weight = 0.1
    else:
        interp_weight = 1.0

    final_score = global_max * weight_a * weight_b * interp_weight

    return final_score


def compute_wups(A,T):

    our_element_membership = lambda x,y: wup_measure(x,y)
    our_set_membership = lambda x,A: fuzzy_set_membership_measure(x,A,our_element_membership)

    return score_it(A,T,our_set_membership)
