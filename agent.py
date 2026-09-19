import heapq
import random
from collections import deque

try:
    from simple_reflex_agent import SimpleReflexAgent
except ImportError:
    pass

try:
    from model_based_agent import ModelBasedAgent
except ImportError:
    pass


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept['agent_pos']
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)


class SearchAgent:
    """Problem-solving agent using classical graph search algorithms for offline path planning."""

    def __init__(self):
        self.plan = []
        self.active_algo = 'BFS'

    def bfs_search(self, start, goal, grid_size=None, walls=None):
        """
        Finds the shortest path from start to goal using Breadth-First Search (BFS).
        
        Args:
            start: tuple or list (x, y) starting coordinate.
            goal: tuple or list (x, y) target coordinate.
            grid_size: tuple (width, height) or walls list if passed positionally.
            walls: collection of (x, y) wall coordinates or grid_size if passed positionally.
            
        Returns:
            list of action strings (e.g. ['Up', 'Right', ...]) leading to the goal,
            or [] if no path is found or start == goal.
        """
        # Support both positional orders: (start, goal, grid_size, walls) and (start, goal, walls, grid_size)
        if isinstance(grid_size, (list, set)) and isinstance(walls, tuple) and len(walls) == 2:
            walls, grid_size = grid_size, walls
        elif isinstance(walls, (tuple, list)) and len(walls) == 2 and isinstance(grid_size, (list, set)):
            walls, grid_size = grid_size, walls

        start = tuple(start)
        goal = tuple(goal)
        walls_set = {tuple(w) for w in (walls or [])}
        width, height = grid_size

        if start == goal:
            return []

        # FIFO frontier storing tuples of (current_position, action_sequence_so_far)
        frontier = deque([(start, [])])
        # Set of reached positions to prevent re-exploration and cycles
        reached = {start}

        # Coordinate transitions for grid movement matching visual_grid_game.py:
        # 'Up': y + 1, 'Down': y - 1, 'Left': x - 1, 'Right': x + 1
        moves = [
            ('Up', 0, 1),
            ('Down', 0, -1),
            ('Left', -1, 0),
            ('Right', 1, 0)
        ]

        while frontier:
            current, path = frontier.popleft()

            if current == goal:
                return path

            for action, dx, dy in moves:
                next_pos = (current[0] + dx, current[1] + dy)

                # Boundary checking: ensure within valid grid dimensions
                if 0 <= next_pos[0] < width and 0 <= next_pos[1] < height:
                    # Ignore walls and already reached states
                    if next_pos not in walls_set and next_pos not in reached:
                        reached.add(next_pos)
                        frontier.append((next_pos, path + [action]))

        return []

    def dfs_search(self, start, goal, grid_size=None, walls=None):
        """
        Finds a path from start to goal using Depth-First Search (DFS).
        
        Args:
            start: tuple or list (x, y) starting coordinate.
            goal: tuple or list (x, y) target coordinate.
            grid_size: tuple (width, height) or walls list if passed positionally.
            walls: collection of (x, y) wall coordinates or grid_size if passed positionally.
            
        Returns:
            list of action strings (e.g. ['Up', 'Right', ...]) leading to the goal,
            or [] if no path is found or start == goal.
        """
        # Support both positional orders: (start, goal, grid_size, walls) and (start, goal, walls, grid_size)
        if isinstance(grid_size, (list, set)) and isinstance(walls, tuple) and len(walls) == 2:
            walls, grid_size = grid_size, walls
        elif isinstance(walls, (tuple, list)) and len(walls) == 2 and isinstance(grid_size, (list, set)):
            walls, grid_size = grid_size, walls

        start = tuple(start)
        goal = tuple(goal)
        walls_set = {tuple(w) for w in (walls or [])}
        width, height = grid_size

        if start == goal:
            return []

        # LIFO stack storing tuples of (current_position, action_sequence_so_far)
        frontier = [(start, [])]
        # Set of reached positions to prevent cycles and re-exploration
        reached = {start}

        # Coordinate transitions for grid movement matching visual_grid_game.py:
        # 'Up': y + 1, 'Down': y - 1, 'Left': x - 1, 'Right': x + 1
        moves = [
            ('Up', 0, 1),
            ('Down', 0, -1),
            ('Left', -1, 0),
            ('Right', 1, 0)
        ]

        while frontier:
            current, path = frontier.pop()

            if current == goal:
                return path

            for action, dx, dy in moves:
                next_pos = (current[0] + dx, current[1] + dy)

                # Boundary checking: ensure within valid grid dimensions
                if 0 <= next_pos[0] < width and 0 <= next_pos[1] < height:
                    # Ignore walls and already reached states
                    if next_pos not in walls_set and next_pos not in reached:
                        reached.add(next_pos)
                        frontier.append((next_pos, path + [action]))

        return []

    def ucs_search(self, start, goal, grid_size=None, walls=None):
        """
        Finds the least-cost path from start to goal using Uniform-Cost Search (UCS).
        
        Args:
            start: tuple or list (x, y) starting coordinate.
            goal: tuple or list (x, y) target coordinate.
            grid_size: tuple (width, height) or walls list if passed positionally.
            walls: collection of (x, y) wall coordinates or grid_size if passed positionally.
            
        Returns:
            list of action strings (e.g. ['Up', 'Right', ...]) leading to the goal,
            or [] if no path is found or start == goal.
        """
        # Support both positional orders: (start, goal, grid_size, walls) and (start, goal, walls, grid_size)
        if isinstance(grid_size, (list, set)) and isinstance(walls, tuple) and len(walls) == 2:
            walls, grid_size = grid_size, walls
        elif isinstance(walls, (tuple, list)) and len(walls) == 2 and isinstance(grid_size, (list, set)):
            walls, grid_size = grid_size, walls

        start = tuple(start)
        goal = tuple(goal)
        walls_set = {tuple(w) for w in (walls or [])}
        width, height = grid_size

        if start == goal:
            return []

        # Priority queue storing tuples: (cost, entry_count, current_position, path)
        # Entry count serves as an incremental tie-breaker for equal-cost nodes
        entry_count = 0
        frontier = [(0, entry_count, start, [])]
        # Maps reached positions to their lowest known path cost g(n)
        cost_so_far = {start: 0}

        # Coordinate transitions for grid movement matching visual_grid_game.py:
        # 'Up': y + 1, 'Down': y - 1, 'Left': x - 1, 'Right': x + 1
        moves = [
            ('Up', 0, 1),
            ('Down', 0, -1),
            ('Left', -1, 0),
            ('Right', 1, 0)
        ]

        while frontier:
            cost, _, current, path = heapq.heappop(frontier)

            # Goal test upon expansion guarantees the optimal (least-cost) path
            if current == goal:
                return path

            # Skip if a lower cost path to current has already been explored
            if cost > cost_so_far.get(current, float('inf')):
                continue

            for action, dx, dy in moves:
                next_pos = (current[0] + dx, current[1] + dy)

                # Boundary checking: ensure within valid grid dimensions
                if 0 <= next_pos[0] < width and 0 <= next_pos[1] < height:
                    # Ignore walls
                    if next_pos not in walls_set:
                        new_cost = cost + 1  # Uniform step cost = 1
                        if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                            cost_so_far[next_pos] = new_cost
                            entry_count += 1
                            heapq.heappush(frontier, (new_cost, entry_count, next_pos, path + [action]))

        return []

    def sense_and_act(self, percept: dict) -> str:
        """Minimal execution method to follow an offline plan."""
        if self.plan:
            return self.plan.pop(0)
        return 'Stay'