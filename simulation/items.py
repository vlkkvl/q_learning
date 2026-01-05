import turtle
from simulation.signs import SignType

class Score(turtle.Turtle):
    def __init__(self, maze):
        super().__init__()
        self.maze = maze
        self.hideturtle()
        self.penup()
        self.color("white")
        self.speed(0)
        self.goto(-330, 310)  # obere linke Ecke
        self.write("Moves: 0", font=("Arial", 16, "normal"))

    def update(self):
        self.clear()
        self.write(f"Moves: {self.maze.moves}", font=("Arial", 16, "normal"))

class Wall(turtle.Turtle):
    def __init__(self):
        super().__init__()
        self.shape("square")
        self.color("white")
        self.penup()
        self.speed(0)

class Goal(turtle.Turtle):
    def __init__(self):
        super().__init__()
        self.shape("square")
        self.color("green")
        self.penup()
        self.speed(0)

# Add new signs here
class Sign(turtle.Turtle):
    def __init__(self, sign_type: SignType, color="yellow", shape="triangle", heading=0):
        super().__init__()
        self.penup()
        self.speed(0)
        self.color(color)
        self.shape(shape)
        self.setheading(heading)
        self.rule_vector = sign_type.as_list()