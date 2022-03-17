

import _pickle

with open("testmin/first/out/stacks.pkl") as f:
    res = _pickle.load(f)

    ######################




d = {}
d['target1'] = pd.DataFrame(index=['1', '2', '3', '4', '5'], columns=['c1', 'c2', 'c3', 'c4', 'c5'])
d['target2'] = pd.DataFrame(index=['1', '2', '3', '4', '5'], columns=['c1', 'c2', 'c3', 'c4', 'c5'])

for item in d.keys():
    filename = item
    df = d[item]
    df.to_csv(filename, encoding='utf-8', index=False)

