class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y;

    def __str__(self):
        return f"P({self.x}, {self.y})"