import numpy as np
import matplotlib.pyplot as plt
import time, math


class Pong(object):

    def __init__(self, gridsize=50, speed=1, paddle_len=10, paddle_width=3, restart=False):

        self.gridsize = gridsize
        self.h = gridsize + 1
        self.w = gridsize + 1
        self.v = speed
        self.paddle_len = paddle_len
        self.paddle_width = paddle_width
        self.restart_on_right_bounce = restart

        self.paddle_t = self.gridsize // 2 - self.paddle_len // 2
        self.paddle_b = self.gridsize // 2 + self.paddle_len // 2

        plt.ion()
        plt.show()

    def start_rollout(self):

        self.phi = math.radians(np.random.choice(np.arange(60) - 30))
        self._intercept = np.random.choice(self.gridsize)
        self.y = self._intercept
        self.x = self.gridsize

        self._xreverse = 1
        self._yreverse = 1

        self.screen = np.zeros((self.h, self.w), dtype=np.int32)

        print(f'phi:{self.phi} intercept: {self._intercept} paddle_t:{self.paddle_t} paddle_b: {self.paddle_b}')

    def step(self, action=0):
        self.y += self._yreverse * self.v * np.sin(self.phi)
        self.x -= self._xreverse * self.v * np.cos(self.phi)

        reward = 0
        end = False

        if (self.x <= self.paddle_width) and ((self.y <= self.paddle_b) and (self.y >= self.paddle_t)):
            self._intercept = self.y
            self.x = self.paddle_width
            self._xreverse *= -1
            reward = 1
        if self.x <= 0:
            self._intercept = self.y
            self.x = 0
            self._xreverse *= -1
            reward = -1

        # if the ball reaches the right wall:
        if self.x >= self.gridsize:
            if not self.restart_on_right_bounce:
                self.x = self.gridsize
                self._intercept = self.y
                self.x = self.gridsize
                self._xreverse *= -1
            else:
                end = True
        if self.y <= 0:
            self.y = 0
            self._intercept = self.y
            self._yreverse *= -1
        if self.y >= self.gridsize:
            self.y = self.gridsize
            self._intercept = self.y
            self._yreverse *= -1

        if action == -1:
            self.paddle_b -= 1
            self.paddle_t -= 1
        elif action == 1:
            self.paddle_b += 1
            self.paddle_t += 1
        else:
            pass

        # boundary condition for the paddle
        if self.paddle_b > self.gridsize:
            self.paddle_b = self.gridsize
            self.paddle_t = self.gridsize - self.paddle_len
        elif self.paddle_t < 0:
            self.paddle_t = 0
            self.paddle_b = self.paddle_len
        else:
            pass

        print(
            f'x: {self.x:.2f}, y: {self.y:.2f}, paddle_b: {self.paddle_b:.2f}, paddle_t: {self.paddle_t:.2f}, reward: {reward}'
        )

        return reward, end

    def render(self, reward=0, end=False):
        if end:
            return
        plt.clf()  # clear figure
        self.screen *= 0
        self.screen[int(np.round(self.y)), int(np.round(self.x))] = 1

        self.screen[self.paddle_t:self.paddle_b, 0:self.paddle_width] = 1
        if reward == 1:
            self.screen *= -1
        plt.imshow(self.screen)
        plt.draw()
        plt.pause(0.0001)


# usage

# env = Pong(
#     gridsize=20,
#     speed=1,
#     paddle_len=7,
#     paddle_width=1,
#     restart=True,
# )
# while True:
#     env.start_rollout()

#     for i in range(1000):

#         reward, end = env.step(action=np.random.choice([-1, 0, 1], p=[0.2, 0.1, 0.7]))
#         env.render(reward=reward, end=end)
#         if reward == -1:
#             print('MISS!!!!')
#             break
#         if reward == 1:
#             print('YES!!!!')
#         if end:
#             break
