"""Small checks for voting and experiment isolation before freezing."""
import json
import os
from protocol import *
assert os.environ.get('TMUX')
assert verdict([True,False,False]) is False
assert verdict([True,False,True]) is True
assert verdict([False]) is False
assert verdict([True]) is True
r=dict(question='Where?',gold_answer='Rome',stored_answer='Paris',f1=0.99,reference_correct=True)
p=prompt(r,'base_0')
assert '0.99' not in p and 'reference_correct' not in p and 'previous_judgments' not in p
prior=[dict(correct=False,reason='Wrong city.')]
m=prompt(r,'meta_single_0',prior)
assert json.loads(m.split('\n\n')[-1])['previous_judgments']==prior
assert 'layer_2' not in KINDS and len(KINDS)==7
assert set(SETTINGS)=={'L0_V1','L0_V3','L1_V1','L1_V3'}
print('Voting, hidden-label isolation, meta inputs and four-condition scope checks passed.')
