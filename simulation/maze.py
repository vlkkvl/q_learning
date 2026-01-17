from simulation import items
from simulation import util_math
from simulation.signs import SignType

class Maze:
    def __init__(self, active_level):
        self.active_level = active_level
        self.moves = 0
        self.tile_size = 24
        self.goal_reached = False
        self.wall = items.Wall()
        self.goal = items.Goal()
        self.score = items.Score(self)
        self.cor_start = None
        self.cor_goal = None
        self.signs = {}  # (x, y) -> one-hot rule
        self.sign_turtles = {
            "Q": items.Sign(SignType.DEAD_END_LEFT, color="yellow", heading=180),
            "W": items.Sign(SignType.DEAD_END_RIGHT, color="yellow", heading=0),
            "S": items.Sign(SignType.DEAD_END_UP, color="yellow", heading=90),
            "A": items.Sign(SignType.DEAD_END_DOWN, color="yellow", heading=270),
            "L": items.Sign(SignType.PREFER_LEFT, color="orange", heading=180),
            "R": items.Sign(SignType.PREFER_RIGHT, color="orange", heading=0),
            "U": items.Sign(SignType.PREFER_UP, color="orange", heading=90),
            "D": items.Sign(SignType.PREFER_DOWN, color="orange", heading=270),
        }

    def setup_maze(self):
        self.reset_moves()
        self.reset_goal()
        self.signs.clear()

        for y in range(len(self.active_level)):
            for x in range(len(self.active_level[y])):
                character = self.active_level[y][x]
                screen_x, screen_y = util_math.list_to_screen(x, y)

                if character == "X":
                    self.wall.goto(screen_x, screen_y)
                    self.wall.stamp()
                if character == "G":
                    self.goal.goto(screen_x, screen_y)
                    self.cor_goal = (x, y)
                if character == "P":
                    self.cor_start = (screen_x, screen_y)
                if character in self.sign_turtles:
                    sign = self.sign_turtles[character]
                    sign.goto(screen_x, screen_y)
                    sign.stamp()
                    self.signs[(x, y)] = sign.rule_vector

    # one-hot representation of the rule, default is 'no rule'
    def get_rule_at(self, x, y):
        return self.signs.get((x, y), SignType.NO_RULE.as_list())

    def update_moves(self):
        self.moves = self.moves + 1
        self.score.update()

    def reset_moves(self):
        self.moves = 0
        self.score.update()

    def reset_goal(self):
        """Clear goal-reached flag for level restarts."""
        self.goal_reached = False

    def check_for_goal(self, x, y):
        """Return True if (x, y) is the goal tile."""
        if self.active_level[y][x] == "G":
            self.goal_reached = True
        else:
            self.goal_reached = False
        return self.goal_reached

    def check_for_border(self, x, y):
        """
        Returns True if player attempts to move:
        - outside the grid boundaries
        - onto a wall tile ('X')
        """
        if y < 0 or y >= len(self.active_level):
            return True
        if x < 0 or x >= len(self.active_level[y]):
            return True
        if self.active_level[y][x] == "X":
            return True
        return False