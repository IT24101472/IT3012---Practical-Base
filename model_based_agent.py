class ModelBasedAgent:
    def __init__(self):
        # 1. INTERNAL STATE & MEMORY
        self.visited_cells = set()     # Stores visited relative coordinates (x, y)
        self.current_pos = (0, 0)      # Internal relative coordinate tracker
        self.facing_index = 0          # 0: North, 1: East, 2: South, 3: West
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.last_action = None

    def update_state(self, percept: dict):
        """
        TRANSITION & SENSOR MODEL:
        Updates internal memory state using the last action taken and current percepts.
        """
        # If last action was move_forward and we didn't hit a wall, update internal position
        if self.last_action == 'move_forward' and not percept.get('hit_wall', False):
            dx, dy = self.directions[self.facing_index]
            self.current_pos = (self.current_pos[0] + dx, self.current_pos[1] + dy)
        
        # Track turning actions
        elif self.last_action == 'turn_left':
            self.facing_index = (self.facing_index - 1) % 4
        elif self.last_action == 'turn_right':
            self.facing_index = (self.facing_index + 1) % 4

        # Record current relative cell into memory
        self.visited_cells.add(self.current_pos)

    def sense_and_act(self, percept: dict) -> str:
        # Step A: Update internal state first
        self.update_state(percept)

        # Calculate coordinates of the cell directly ahead
        dx, dy = self.directions[self.facing_index]
        front_pos = (self.current_pos[0] + dx, self.current_pos[1] + dy)

        # Step B: Memory-Aware Condition-Action Rules
        if percept.get('food_here'):
            action = 'suck'
        elif percept.get('wall_ahead'):
            action = 'turn_right'  # Turn right on hitting a wall
        elif front_pos in self.visited_cells:
            # MEMORY CHECK: If the cell ahead has already been visited, turn to break loop
            action = 'turn_left'
        else:
            action = 'move_forward'

        self.last_action = action
        return action