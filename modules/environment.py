import numpy as np
import matplotlib.pyplot as plt
import time, math
from scipy.ndimage import gaussian_filter


class Pong(object):

    def __init__(
        self,
        gridsize=50,
        speed=1,
        paddle_len=10,
        paddle_width=3,
        restart=False,
        visible=True,
    ):
        self.visible = visible
        self.sigma = 1
        self.gridsize = gridsize
        self.max_x = gridsize - 1
        self.max_y = gridsize - 1
        self.v = speed
        self.paddle_len = paddle_len
        self.paddle_width = paddle_width
        self.restart_on_right_bounce = restart

        self.paddle_t = self.max_y // 2 - self.paddle_len // 2
        self.paddle_b = self.max_y // 2 + self.paddle_len // 2
        self.stepid = 0
        self.paddle_on = True

        if self.visible:
            plt.ion()
            plt.show()

    def _start_rollout(self):

        self.phi = math.radians(45)  #math.radians(np.random.choice(np.arange(60) - 30))
        self._intercept = 3  #np.random.choice(self.gridsize)
        self.y = self._intercept
        self.x = self.max_x

        self._xreverse = 1
        self._yreverse = 1

        self.screen = np.zeros((self.gridsize, self.gridsize), dtype=np.float32)
        self.screen[self.y, self.x] = 1

        # print(f'phi:{self.phi} intercept: {self._intercept} paddle_t:{self.paddle_t} paddle_b: {self.paddle_b}')

    def step(self, action=0):
        reward = 0
        end = False

        if self.stepid == 0:
            self._start_rollout()
        else:
            self.y += self._yreverse * self.v * np.sin(self.phi)
            self.x -= self._xreverse * self.v * np.cos(self.phi)

            if self.paddle_on:
                # deflect if the ball hits the paddle
                if (self.x <= self.paddle_width) and ((self.y <= self.paddle_b) and (self.y >= self.paddle_t)):
                    self._intercept = self.y
                    self.x = self.paddle_width
                    self._xreverse *= -1
                    reward = 1

            # if the ball goes beyond the left wall
            if np.round(self.x) < 0:
                self._intercept = self.y
                if self.paddle_on:
                    reward = -1
                self._xreverse *= -1

            if np.round(self.x) == 0:
                self._intercept = self.y
                self.x = 0
                if not self.paddle_on:
                    self._xreverse *= -1

            # end if the ball goes beyond the right wall:
            if self.x > self.max_x:
                if not self.restart_on_right_bounce:
                    self.x = np.round(self.x)
                    self._intercept = self.y
                    self._xreverse *= -1
                else:
                    end = True

            # bounce off the top wall
            if self.y < 0:
                self.y = 0
                self._intercept = self.y
                self._yreverse *= -1
            # bounce off the bottom wall
            if self.y >= self.max_y:
                self.y = self.max_y
                self._intercept = self.y
                self._yreverse *= -1

            if self.paddle_on:
                # move the paddle up
                if action == -1:
                    self.paddle_b -= 1
                    self.paddle_t -= 1
                # move the paddle down
                elif action == 1:
                    self.paddle_b += 1
                    self.paddle_t += 1
                else:
                    pass

            # boundary condition for the paddle (force it to stay within the screen)
            if self.paddle_b > self.max_y:
                self.paddle_b = self.gridsize
                self.paddle_t = self.gridsize - self.paddle_len
            elif self.paddle_t < 0:
                self.paddle_t = 0
                self.paddle_b = self.paddle_len
            else:
                pass

            # print(
            #     f'x: {self.x:.2f}, y: {self.y:.2f}',
            #     f'paddle_b: {self.paddle_b:.2f}, paddle_t: {self.paddle_t:.2f}',
            #     f'reward: {reward}',
            # )
        if end or reward == -1:
            self.stepid = 0
        else:
            self.stepid += 1

        return reward, end

    def render(self, gauss=False, reward=0):

        if self.visible:
            plt.clf()  # clear figure

        self.screen *= 0

        # if the ball is not outsize the grid, draw the ball
        if not ((self.x > self.max_x) or (reward == -1)):
            self.screen[int(np.round(self.y)), int(np.round(self.x))] = 3
        # draw the paddle
        if self.paddle_on:
            self.screen[self.paddle_t:self.paddle_b, 0:self.paddle_width] = 1

        # if reward == 1:
        #     self.screen *= -1

        im = gaussian_filter(self.screen, sigma=self.sigma) if gauss else self.screen
        if not im.max() == 0.0:
            im = im / im.max()
        if self.visible:
            plt.imshow(im)
            plt.title(f'{self.y:.2f}, {self.x:.2f}, {reward}')
            plt.draw()
            plt.pause(0.0001)
        return im


class InfinitePong(object):

    def __init__(self, visible=True):
        self.env = Pong(gridsize=20, speed=1, paddle_len=6, paddle_width=1, restart=True, visible=visible)
        self.end = False
        self.reward = 0
        self.visible = visible

    def step(self, action=0, gauss=True):
        # call this every 50 ms of model time
        self.reward, self.end = self.env.step(action=action)

        im = self.env.render(reward=self.reward, gauss=gauss)

        if self.visible:
            if self.end:
                print('end')
            elif self.reward == -1:
                print('miss')
            else:
                pass

        return im, self.env.x, self.env.y, self.reward


# env = InfinitePong(visible=True)