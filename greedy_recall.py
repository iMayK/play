import json
import math

def get_snm(segment, n, n_idx, cos, selected):
    if n in selected:
        return 0
    q_distri = cos[segment]
    q_distri = [(score+1)/2 for score in q_distri] 
    sum_q_distri = sum(q_distri)
    q_distri = [score/sum_q_distri for score in q_distri]
    entropy = -sum([p*math.log(p) for p in q_distri])  #
    #weigh = 1/entropy if entropy != 0 else 0
    weight = 1/(1+math.exp(-entropy))                  #

    return weight * cos[i][n_idx]

def get_eij(n, selected, reward_1=1, reqard_2=0.5):
    if n in selected:
        return 0
    adj_score = 0
    for item in selected:
        if item.split('.')[:2] == n.split('.')[:2]:
            adj_score += reward_1
        elif item.split('.')[:1] == n.split('.')[:1]:
            adj_score += reward_2 
    return adj_score

def get_complete_score(segment, n, n_idx, cos, selected):
    snm = get_snm(segment, n, n_idx, cos, selected)
    eij = get_eij(n, selected)
    return (snm + eij, n)

def get_data(docs, segments, TOPK):
    cos = defaultdict(list)
    schema_items = []
    for doc in docs[:TOPK]:
        schema_items.append(doc['name'])
        for segment in segments:
            cos[segment].append(doc[segment])
    return cos, schema_items

def greedy_select(segments, docs):
    '''
        segments: list of segments
        docs: sorted list of dicts, where each dict is of the form 
              {
                'name': ..., 
                'score': ...,           # final score
                'segment1': ...,        # score for segment1
                'segment2': ...,        # score for segment2
                ...
                }
    '''
    cos, schema_items = get_data(docs, TOPK)
    M = len(segments)
    covered = [0 for _ in range(M)]
    selected = set()
    while sum(covered) < M:
        lst = []
        for i, segment in enumerate(segments):
            if covered[i] == 1:
                continue

            lst_score = []              # list of (score, n) tuples
            for n_idx, n in enumerate(schema_items):
                lst_score.append(get_complete_score(segment, n, n_idx, cos, selected))

            s_dash, n_dash = max(lst_score, key=lambda x: x[0])
            best_n = (s_dash, n_dash)

            lst.append((i, best_n))
        i_dash, (s_dash, n_dash) = max(lst, key=lambda x: x[1][0])
        covered[i_dash] = 1
        selected.add(n_dash)
    return list(selected)

TOPK = 20

metadata = json.load(open('../../data/spider_dev_W_schema_elements_AND_decomposition.json'))

data = json.load(open()) # path to predictions file

lst_recall = []
for idx, item in enumerate(data):
    if '*' not in metadata[idx]['query']:
        gold_elements = item['gold_dbs']
        segments = item['segments']
        docs = item['docs']
        selected = greedy_select(segments, docs)

        recall = len(set(gold_elements).intersection(set(selected))) / len(gold_elements)
        lst_recall.append(recall)

print('greedy recall: ')
print(sum(lst_recall)/len(lst_recall))