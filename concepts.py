from collections import namedtuple
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
import operator

Event = namedtuple('Event', ['x', 'y', 'ts', 'polarity'])
Bound = namedtuple('Bound', ['lower', 'upper'])
Coordinate = namedtuple('Coordinate', ['x', 'y'])
Camera = namedtuple('Camera', ['width', 'height'])
Polarity = namedtuple('Polarity', ['name', 'encoding'])
TrainingSample = namedtuple('TrainingSample', ['features', 'label'])

class MotionFrame:

    def __init__(self, camera, events=None, **kwargs):
        """
            Creates a matrix representation of the passed events, very similar to a frame.

            :param       camera: a named tuple containing width and height of the frame
            :param       events: a list of events to be included in the frame
            :param     polarity: the polarity of the frame: 'both', 'positive' or 'negative'
            :param   frame_type: the polarity of the frame that determines the representation
            :param       kwargs: additional parameters like tau, decoder, etc
        """
        self.width = camera.width
        self.height = camera.height

        self.events = events

        self.image = np.zeros((self.height, self.width), dtype=np.int8)

        default_decoder = {'negative': {-1}, 'positive': {1}}
        self.decoder = kwargs.get('decoder', default_decoder)

        self.index = 0
        self.last_ts = 0

        step_by_step = kwargs.get('step_by_step', False)
        if events is not None:
            if not step_by_step:
                self.add_all()
            # else add events one by one with `step`
        # else add events manually with `add_event`


    def add_event(self, ev):
        """
        Adds the event to the internal representation considering the event polarity
        and the chosen representation
        :param ev: the event to be added
        """
        if ev.polarity in self.decoder['positive']:
            self.image[int(ev.y), int(ev.x)] = +1

        if ev.polarity in self.decoder['negative']:
            self.image[int(ev.y), int(ev.x)] = -1

    def add_all(self):
        for ev in self.events:
            self.add_event(ev)


    def step(self):
        if self.index < len(self.events):
            self.add_event(self.events[self.index])
            self.index += 1
        else:
            print("All events processed")

    def show(self):
        plt.imshow(self.image, cmap='gray', vmin=-1, vmax=1)

