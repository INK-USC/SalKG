NUM_RELS = 35
NUM_CHOICES = {
    'csqa': 5,
    'obqa': 4,
    'codah': 4,
    'qasc': 8
}
CHOICE_KEYS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
ADJ_KEYS = ['adj_A', 'adj_B', 'adj_C', 'adj_D', 'adj_E', 'adj_F', 'adj_G', 'adj_H']
NUM_CSQA_INHOUSE = {
    'train': 8500,
    'valid': 1221,
    'test': 1241
}
NUM_CSQA_OFFICIAL = {
    'train': 9741,
    'valid': 1221,
    'test': 1140
}
NUM_OBQA_OFFICIAL = {
    'train': 4957,
    'valid': 500,
    'test': 500
}
NUM_CODAH_FOLD_0 = {
    'train': 1665,
    'valid': 556,
    'test': 555
}
NUM_QASC_INHOUSE = {
    'train': 7214,
    'valid': 926,
    'test': 920
}

TRAINING_TQDM_BAD_FORMAT = (
    '{l_bar}{bar}| '
    '{n_fmt}/{total_fmt} [{elapsed}<{remaining} {postfix}]'
)
