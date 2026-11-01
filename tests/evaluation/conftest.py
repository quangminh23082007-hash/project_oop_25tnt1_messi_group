from src.evaluation.protocol import Interaction


class FixedScoreModel:
    def __init__(self, scores=None, num_users=3, num_items=6):
        self.name = "FixedScoreModel"
        self.is_fitted = True
        self.num_users = num_users
        self.num_items = num_items
        self.scores = {} if scores is None else scores
        self.calls = []

    def score_items(self, user_id, item_ids):
        self.calls.append((user_id, tuple(item_ids)))
        return [self.scores.get((user_id, item_id), 0.0) for item_id in item_ids]


def make_interactions(rows):
    return tuple(Interaction(*row) for row in rows)


def make_evaluation_case():
    train = make_interactions(((0, 0, 1), (0, 1, 0), (1, 2, 1)))
    test = make_interactions(((0, 2, 1), (0, 4, 0), (1, 3, 1), (2, 5, 1)))
    scores = {(0, 2): 5.0, (1, 0): 5.0, (1, 3): 4.0}
    return FixedScoreModel(scores), train, test
