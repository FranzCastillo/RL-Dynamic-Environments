class Node:
    def __init__(self, n_actions):
        self.visit_count = 0
        self.total_value = 0.0
        self.action_stats = {
            action: {
                "visit_count": 0,
                "total_value": 0.0
            }
            for action in range(n_actions)
        }