import matplotlib.pyplot as plt
from matplotlib import animation, rc
import numpy as np
from dataloader import DataLoader, Dataset
from IPython.display import HTML

class Animate:
    def __init__(self, x_lim, y_lim):
        """Instantiates an object to visualize the generated poses."""
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=x_lim, ylim=y_lim)
        self.line, = self.ax.plot([], [], lw=2, c='b')
        self.line2, = self.ax.plot([], [], lw=2, c='b')
        self.line3, = self.ax.plot([], [], lw=2, c='b')

    def initLines(self):
        """Creates line objects which are drawn later."""
        self.line.set_data([], [])
        self.line2.set_data([], [])
        self.line3.set_data([], [])
        return self.line,

    def animateframe(self, i):
        """Animation method, draws a pose per frame."""
        merged_x = list(i[0:10:2])
        merged_y = list(i[1:10:2])
        x = []
        y = []
        x.append(i[2])
        x.append(i[10])
        y.append(i[3])
        y.append(i[11])
        self.line3.set_data(x, y)
        self.line2.set_data(i[10::2], i[11::2])
        self.line.set_data(merged_x, merged_y)
        return (self.line, )

    def animate(self, frames_to_play, interval):
        """Returns a matplotlib animation object that can be saved as a video."""
        anim = animation.FuncAnimation(self.fig, self.animateframe, init_func=self.initLines, frames=frames_to_play, interval=interval, blit=True)
        return anim

def main():
    dl = DataLoader("./data/preprocessed_1295videos.pickle", "./data/glove.6B.300d.txt", 10000, 10)
    mean_pose = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    mean_pose = dl.pca.inverse_transform(mean_pose)
    animation_object = Animate((-1, 1), (-1, 1))
    anim = animation_object.animate(mean_pose, 500)
    anim.save('animation.gif', writer='imagemick', fps=60)

if __name__ == '__main__':
    main()
