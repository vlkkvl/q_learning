import turtle
from communication.fulcrum import Fulcrum
from simulation import util_math
import numpy as np

# grid movement (maze coords)
DIRS = {
    "left": (-1, 0),
    "right": (1, 0),
    "up": (0, -1),
    "down": (0, 1),
}

ACTIONS = {
    "GO_LEFT": 0,
    "GO_RIGHT": 1,
    "GO_UP": 2,
    "GO_DOWN": 3,
}


class Player(turtle.Turtle, Fulcrum):
    def __init__(self, maze):
        super().__init__()
        self.shape("square")
        self.color("blue")
        self.penup()
        self.speed(0)
        self.maze = maze
        self.maze_width = len(self.maze.active_level[0])  # number of columns
        self.maze_height = len(self.maze.active_level)  # number of rows
        if maze is not None: self.move_to_start()

    def get_coordinates(self):
        """
        :returns current player **maze** coordinates
        """
        x_screen, y_screen = self.position()
        x, y = util_math.screen_to_list(x_screen, y_screen)
        return x, y

    def get_goal_coordinates(self):
        """
        :returns goal **maze** coordinates
        """
        return self.maze.cor_goal

    def move(self, direction):
        """
        Moves player to a specified position.
        :param direction: direction to go as string (left, right, up, down)
        :returns:
        - moved - was agent able to move (no wall)?
        - goal_reached: was the goal reached?
        """
        dx, dy = DIRS[direction]  # grid delta (±1, 0)
        tile = self.maze.tile_size

        sx = dx * tile  # screen movement x
        sy = -dy * tile  # screen movement y

        x, y = self.get_coordinates()

        new_x = x + dx
        new_y = y + dy

        goal_reached = self.maze.check_for_goal(new_x, new_y)
        able_to_move = not self.maze.check_for_border(new_x, new_y)  # no wall ahead
        if not able_to_move:
            moved = False
        else:
            self.goto(self.xcor() + sx, self.ycor() + sy)
            self.maze.update_moves()
            moved = True
        return moved, goal_reached


    def move_to_start(self):
        """
        Moves player to a start position.
        """
        if self.maze.cor_start is not None:
            x_start, y_start = self.maze.cor_start
            self.goto(x_start, y_start)

    def get_state(self):
        """
        Returns current observable state of the agent as np.ndarray.

        STATE REPRESENTATION (16 features):
        [0:4]  - available actions (left_free, right_free, up_free, down_free)
        [4:12]  - one-hot sign encoding
        [12]    - normalized x position (0 to 1)
        [13]   - normalized y position (0 to 1)
        [14]   - normalized dx to goal (signed, -1 to 1)
        [15]   - normalized dy to goal (signed, -1 to 1)
        """
        x, y = self.get_coordinates()
        goal_x, goal_y = self.maze.cor_goal

        # Wall detection
        up_free = 0 if self.maze.check_for_border(x, y - 1) else 1
        left_free = 0 if self.maze.check_for_border(x - 1, y) else 1
        right_free = 0 if self.maze.check_for_border(x + 1, y) else 1
        down_free = 0 if self.maze.check_for_border(x, y + 1) else 1

        # Sign at current position
        sign_vec = self.maze.get_rule_at(x, y)

        # Position features (normalized to [0, 1])
        norm_x = x / self.maze_width
        norm_y = y / self.maze_height

        # Goal-relative features (normalized to [-1, 1])
        dx_to_goal = (goal_x - x) / self.maze_width
        dy_to_goal = (goal_y - y) / self.maze_height

        features = (
                [left_free, right_free, up_free, down_free] +
                sign_vec +
                [norm_x, norm_y, dx_to_goal, dy_to_goal]
        )
        return np.array(features, dtype=float)

    def execute_action(self, action):
        """
        Executes one action in the environment.
        The actions are 0 - left, 1 - right, 2 - up, 3 - down

        :return: (next_state, done, info)
        - next_state: observable state (left_free, right_free, up_free, down_free, sign)
        - moved: was agent able to move (no wall)?
        - done: goal reached?
        """
        if action == ACTIONS["GO_LEFT"]:
            moved, goal_reached = self.move("left")
        elif action == ACTIONS["GO_RIGHT"]:
            moved, goal_reached = self.move("right")
        elif action == ACTIONS["GO_UP"]:
            moved, goal_reached = self.move("up")
        elif action == ACTIONS["GO_DOWN"]:
            moved, goal_reached = self.move("down")
        else:
            raise ValueError("invalid action")

        next_state = self.get_state()
        return next_state, moved, goal_reached

    def reset(self):
        """
        Resets environment:
         - moves player to start position
        - sets goal as not reached
        - resets moves to 0
        """
        self.move_to_start()
        self.maze.reset_moves()
        self.maze.reset_goal()
