import pygame
import sys
import heapq
import random
from typing import List, Tuple

# Game settings
SCREEN_SIZE = 400
GRID_SIZE = 20
GRID_WIDTH = SCREEN_SIZE // GRID_SIZE
GRID_HEIGHT = SCREEN_SIZE // GRID_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 100, 0)

# Agent roles
PLAYER1_ROLE = "Food collector"
PLAYER2_ROLE = "Landmine defuser"


class Sensor:
    def __init__(self, game_world):
        self.game_world = game_world

    def get_visible_food_positions(self):
        return self.game_world.food_positions

    def get_visible_landmine_positions(self):
        return self.game_world.landmine_positions


class Actuator:
    def __init__(self, game_world):
        self.game_world = game_world

    def move(self, agent, new_position):
        old_x, old_y = agent.position
        new_x, new_y = new_position
        self.game_world.grid[old_y][old_x] = '.'
        self.game_world.grid[new_y][new_x] = agent.symbol
        agent.position = new_position

    def collect_food(self, agent):
        if agent.position in self.game_world.food_positions:
            self.game_world.food_positions.remove(agent.position)
            agent.health += 10
            agent.score += 1
            agent.collected_food += 1

    def defuse_landmine(self, agent):
        if agent.position in self.game_world.landmine_positions:
            self.game_world.landmine_positions.remove(agent.position)
            agent.defused_landmines += 1


class Agent:
    def __init__(self, symbol: str, role: str, game_world):
        self.symbol = symbol
        self.role = role
        self.position = game_world.get_position(symbol)
        self.sensor = Sensor(game_world)
        self.actuator = Actuator(game_world)
        self.health = 10
        self.score = 0
        self.collected_food = 0
        self.defused_landmines = 0

    def get_visible_food_positions(self):
        return self.sensor.get_visible_food_positions()

    def get_visible_landmine_positions(self):
        return self.sensor.get_visible_landmine_positions()

    def move(self, new_position):
        self.actuator.move(self, new_position)

    def collect_food(self):
        self.actuator.collect_food(self)

    def defuse_landmine(self):
        self.actuator.defuse_landmine(self)


class Communication:
    def __init__(self, agent1: Agent, agent2: Agent):
        self.agent1 = agent1
        self.agent2 = agent2

    def share_food_positions(self):
        food_positions = self.agent1.get_visible_food_positions() + self.agent2.get_visible_food_positions()
        return list(set(food_positions))

    def share_landmine_positions(self):
        landmine_positions = self.agent1.get_visible_landmine_positions() + self.agent2.get_visible_landmine_positions()
        return list(set(landmine_positions))


class GameWorld:
    def __init__(self, filename: str):
        self.grid = self.load_grid_from_file(filename)
        self.player1 = Agent('P', PLAYER1_ROLE, self)
        self.player2 = Agent('H', PLAYER2_ROLE, self)
        self.food_positions = self.get_positions('F')
        self.landmine_positions = self.get_positions('L')
        self.communication = Communication(self.player1, self.player2)


    def load_grid_from_file(self, filename: str) -> List[List[str]]:
        with open(filename, 'r') as f:
            lines = f.readlines()
        return [list(line.strip()) for line in lines]

    def get_position(self, item: str) -> Tuple[int, int]:
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == item:
                    return (x, y)

    def get_positions(self, item: str) -> List[Tuple[int, int]]:
        positions = []
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == item:
                    positions.append((x, y))
        return positions

    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        x1, y1 = pos1
        x2, y2 = pos2
        random_weight = random.uniform(0.9, 1.1)  # Add random weight between 0.9 and 1.1
        return int((abs(x1 - x2) + abs(y1 - y2)) * random_weight)

    def astar(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        pq = [(0, start, [])]
        visited = set()

        while pq:
            (f, (x, y), path) = heapq.heappop(pq)
            if (x, y) == end:
                return path + [(x, y)]

            if (x, y) not in visited:
                visited.add((x, y))
                neighbors = [(x + dx, y + dy) for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0))]
                random.shuffle(neighbors)  # Add randomness
                for next_x, next_y in neighbors:
                    if (0 <= next_x < GRID_WIDTH and 0 <= next_y < GRID_HEIGHT and
                            self.grid[next_y][next_x] != 'W' and
                            (next_x, next_y) not in visited):
                        g = len(path) + 1
                        h = self.heuristic((next_x, next_y), end)
                        heapq.heappush(pq, (g + h, (next_x, next_y), path + [(x, y)]))
        return []

    def update_player1(self):
        if not self.food_positions:
            return

        closest_food, min_dist = self.food_positions[0], float('inf')
        for food in self.food_positions:
            path = self.astar(self.player1.position, food)  # Use the astar function
            if len(path) < min_dist:
                min_dist = len(path)
                closest_food = food

        next_pos = self.astar(self.player1.position, closest_food)[1]  # Use the astar function
        if next_pos in self.landmine_positions:
            self.player1.health = 0
            print("Player 1 destroyed by a landmine! :(")
            return

        # Update the grid
        prev_x, prev_y = self.player1.position
        self.grid[prev_y][prev_x] = '.'
        next_x, next_y = next_pos
        self.grid[next_y][next_x] = 'P'

        self.player1.position = next_pos
        self.player1.health -= 1

        if self.player1.position in self.food_positions:
            self.food_positions.remove(self.player1.position)
            self.player1.health += 10
            self.player1.collected_food += 1  # Update the score when collecting food

            # Update the grid to remove food
            self.grid[next_y][next_x] = 'P'



    def update_player2(self):
        if not self.landmine_positions:
            return

        closest_landmine, min_dist = self.landmine_positions[0], float('inf')
        for landmine in self.landmine_positions:
            path = self.astar(self.player2.position, landmine)  # Use the astar function
            if len(path) < min_dist:
                min_dist = len(path)
                closest_landmine = landmine

        next_pos = self.astar(self.player2.position, closest_landmine)[1]  # Use the astar function
        if next_pos in self.food_positions:
            self.player2.health -= 1
            if self.player2.health <= 0:
                print("Player 2 destroyed! :(")
                return

        self.player2.move(next_pos)
        self.player2.defuse_landmine()

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))

        while True:
            screen.fill(WHITE)
            for y, row in enumerate(self.grid):
                for x, cell in enumerate(row):
                    if cell == 'W':
                        pygame.draw.rect(screen, BLACK, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))
            for (x, y) in self.food_positions:
                pygame.draw.rect(screen, GREEN, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))
            for (x, y) in self.landmine_positions:
                pygame.draw.rect(screen, RED, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

            px, py = self.player1.position
            pygame.draw.rect(screen, BLUE, (px * GRID_SIZE, py * GRID_SIZE, GRID_SIZE, GRID_SIZE))
            hx, hy = self.player2.position
            pygame.draw.rect(screen, ORANGE, (hx * GRID_SIZE, hy * GRID_SIZE, GRID_SIZE, GRID_SIZE))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

            self.update_player1()
            self.update_player2()
            print(f"Player 1 collected food: {self.player1.collected_food}")
            print(f"Player 2 defused landmines: {self.player2.defused_landmines}")

            pygame.time.delay(200)

if __name__ == "__main__":
    filename = "world.txt"
    game = GameWorld(filename)
    game.run()

