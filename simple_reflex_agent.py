#1.2 Simple reflex agent implementation
class SimpleReflexAgent:
    def __init__(self):
        pass

    def sense_and_act(self, percept: dict) -> str:
        if percept.get('food_here'):
            return 'suck'
        elif percept.get('wall_ahead'):
            return 'turn_left'
        else:
            return 'move_forward'