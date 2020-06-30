import math
import numpy as np
from concepts import Bound, Coordinate, MotionFrame, Event, Polarity, Camera
from sklearn.metrics.pairwise import cosine_similarity
import random


class TimeSurface:

    def __init__(self, camera, events=None, polarities=None, **kwargs):
        """
            Creates a matrix representation of the passed events, very similar to a frame.

            :param       camera: a namedtuple containing width and height of the frame
            :param       events: a list of events to be included in the frame
            :param   polarities: a list or set of Polarity namedtuples
            :param       kwargs: additional parameters like tau, decoder, etc
        """
        self.width = camera.width
        self.height = camera.height

        self.events = events
        self.polarities = polarities

        # Each polarity has an encoding (eg. negative, -1)
        # We need to keep a separate part for each polarity and we need
        # to keep in mind, which part corresponds to which polarity
        layer_decoder = []
        polarity_decoder = []
        for i, polarity in enumerate(polarities):
            layer_decoder.append((polarity.name, i))
            polarity_decoder.append((polarity.encoding, polarity.name))
        self.polarity_decoder = dict(polarity_decoder)
        self.layer_decoder = dict(layer_decoder)

        # A time surface is composed of several parts, each for each polarity.
        self.data = np.zeros((self.height, self.width, len(self.polarities)), dtype=np.float32)

        self.tau = kwargs.get('tau', 0.1)

        # index of the next event that will be processed
        self.index = 0
        # last t_s processed, useful for the exponential decay
        self.last_ts = 0

        step_by_step = kwargs.get('step_by_step', False)
        if events is not None:
            if not step_by_step:
                self.add_all()
            # else add events with one by one with `step`
        # else add one by one with `add_event`


    def add_event(self, ev):
        """
        Adds the event to the internal representation considering the event polarity
        and the chosen representation
        :param ev: the event to be added
        """
        delta_t = ev.ts - self.last_ts
        self.data = np.multiply(self.data, math.exp(-delta_t / self.tau))
        layer_idx = self.layer_decoder[self.polarity_decoder[ev.polarity]]
        self.data[int(ev.y), int(ev.x), layer_idx] = 1
        self.last_ts = ev.ts


    def add_all(self):
        for ev in self.events:
            self.add_event(ev)


    def step(self):
        if self.index < len(self.events):
            self.add_event(self.events[self.index])
            self.index += 1
        else:
            print("All events processed")

class Neighborhood:

    def __init__(self, time_surface, x, y, R=20):
        """

        :param time_surface: a time surface object
        :param x: the x coordinate
        :param y: the y coordinate
        :param R: the dimension parameter of the neighborhood, 
                  size = (2R+1) x (2R+1)
        """

        self.surface_maxbounds = Coordinate(time_surface.data.shape[1], time_surface.data.shape[0])

        self.center = Coordinate(x,y)
        self.xbounds = Bound(max(0, x - R), min(self.surface_maxbounds.x -1, x + R))
        self.ybounds = Bound(max(0, y - R), min(self.surface_maxbounds.y -1, y + R))

        self.width = 2 * R + 1
        self.height = 2 * R + 1

        # Padding
        # If the neighborhood is not (2R+1)^2, the data must be padded in
        # order to make the triggering event the central event, (R+1, R+1)
        extra_left = max(0, -(x - R))
        extra_right = max(0, (x + R - (self.surface_maxbounds.x-1)))
        extra_top = max(0, -(y - R))
        extra_bottom = max(0, (y + R - (self.surface_maxbounds.y-1)))

        data = time_surface.data[self.ybounds.lower:self.ybounds.upper+1, self.xbounds.lower:self.xbounds.upper+1, :]
        self.data = np.pad(data, ((extra_top, extra_bottom), (extra_left, extra_right), (0,0)))

        self.index = 0
        self.last_ts = 0

        # useful to remap the events coordinates considering it has been sliced
        # new_coord_rep = old_coord_rep - offset
        # e.g. (23,5) with center (20,10), R= 5 becomes (8,0)
        self.offset = Coordinate(self.xbounds.lower, self.ybounds.lower)


    def print_bounds(self):
        print(f" x:{(self.xbounds.left, self.xbounds.right)}, y: {self.ybounds.left, self.ybounds.right} ")


class SurfaceClusterLayer:

    def __init__(self, camera, polarities, events=None, N=8, R=4, tau=0.5):
        """
            Assumes :param events: is a list of events sorted by timestamp in ascending order
            and all of the same polarity, no further checks will be performed. It also assumes
            there are more than :param N: events.

        """
        self.camera = camera
        self.events = events
        self.polarities = polarities
        self.N = N
        self.R = R
        self.tau = tau

        # Index for tracking the events processed
        self.index = 0

        # global canvas that tracks all events, with one part per each polarity
        self.time_surface = TimeSurface(camera, events=events, polarities=polarities)

        # counts how many time surfaces have been assigned to the
        # prototype C_k with k in [0,N-1]
        self.proto_activation_count = np.zeros(N)

        # Output for the layer (prototype activations)
        self.output = []

        self.initialized = False


    def reset(self):
        """
            Resets global time_surface tracking events, prototypes remain untouched
        """
        self.index = 0
        self.output = []
        self.time_surface = TimeSurface(self.camera, events=None, polarities=self.polarities)

    def set_events(self, events):
        self.events = events

    def process(self, events, mode):
        # start with a blank canvas for tracking events
        self.reset()
        # use as events the ones received as parameter
        self.events = events
        # initialize the prototype surfaces
        if self.initialized is False:
            self.initialize_centers()
            self.initialized = True
        # process all events one by one
        while self.index < len(self.events) - 1: self.step(mode)

    def initialize_centers(self):

        # container for the prototype time surfaces
        self.prototypes = []

        for i in range(self.N):
            ev = self.events[i]
            self.prototypes.append(Neighborhood(self.time_surface, ev.x, ev.y, R=self.R))
            self.time_surface.add_event(ev)
            # add one to the tracker of the global number of events processed
            self.index += 1

    def step(self, mode):
        """ :param mode: determines whether we are in train or test mode"""

        # get the next event to process
        ev = self.events[self.index]
        self.add_event(ev, mode)

    def add_event(self, ev, mode):

        # add it to the global event tracker canvas (time context)
        self.time_surface.add_event(ev)

        # get the neighborhood of the event
        neighborhood = Neighborhood(self.time_surface, int(ev.x), int(ev.y), R=self.R)
        s_i = neighborhood.data

        # get closest according to euclidean distance
        c_k, c_k_index = self.find_closest_time_surface_to(s_i)

        if mode == 'train':
            # update the prototype
            p_k = self.proto_activation_count[c_k_index]
            alpha = 0.01 / (1 + p_k / 20000)
            beta = cosine_similarity(c_k.reshape(1,-1), s_i.reshape(1,-1))
            self.prototypes[c_k_index].data = c_k + alpha * (s_i - beta * c_k)

            self.proto_activation_count[c_k_index] += 1

        self.index += 1

        # the prototype index becomes a polarity for next Layer
        prototype_activation = Event(ev.x, ev.y, ev.ts, c_k_index)
        self.output.append(prototype_activation)
        return prototype_activation

    def process_all(self, mode='train'):
        if self.initialized is False:
            self.initialize_centers()
            self.initialized = True
        while self.index < len(self.events) -1 : self.step(mode)


    def find_closest_time_surface_to(self, s_i):

        min_distance = 1e10
        index = None

        for i, c_i in enumerate([tsf.data for tsf in self.prototypes]):
            distance = np.linalg.norm(c_i - s_i)
            if distance < min_distance:
                min_distance = distance
                index = i
            # if distance == min_distance and random.random() < 2 / self.N:
            #     index = i
            # if distance < 5 :
            #     index = random.choice(range(self.N))
        return self.prototypes[index].data, index


class Hierarchy:

    def __init__(self, camera, N=4, R=4, tau=0.04, KN=2, KR=2, KT=5):
        """
            HOTS: Hierarchy of Time surfaces

        :param camera: camera Namedtuple representing the sensor on which the data
                            was obtained
        :param      N: number of features
        :param      R: size of the neighborhood radius
        :param    tau: decay parameter
        :param     KN: feature increase factor
        :param     KR: neighborhood radius increase factor
        :param     KT: decay parameter increase factor
        """
        self.camera = camera

        self.N = N
        self.R = R
        self.tau = tau
        self.KN = KN
        self.KR = KR
        self.KT = KT

        # First Layer
        l1_polarities = {Polarity(name='negative', encoding= -1), Polarity(name='positive', encoding=1) }
        self.layer1 = self.create_layer(layer_idx=1, layer_polarities=l1_polarities)
        # Second Layer
        self.layer2 = self.create_layer(layer_idx=2)
        # Third Layer
        self.layer3 = self.create_layer(layer_idx=3)


    def create_layer(self, layer_idx=None, events = None, layer_polarities = None):

        num_features = self.N*(self.KN**(layer_idx-1))
        neighborhood_size = self.R*(self.KR**(layer_idx-1))
        integration_scale = self.tau*(self.KT**(layer_idx-1))

        if layer_polarities is None:
            layer_polarities = []
            for i in range(num_features//self.KN):
                layer_polarities.append(Polarity(name=f'layer{layer_idx-1}_proto{i}', encoding=i))

        layer = SurfaceClusterLayer(self.camera, layer_polarities, events, num_features, neighborhood_size, integration_scale)
        return layer


    def process(self, events, mode='train'):
        """
        :param events: the AER data produced by the event-based camera as a list of
                            Namedtuples
        """
        self.layer1.process(events=events, mode=mode)
        self.layer2.process(events=self.layer1.output, mode=mode)
        self.layer3.process(events=self.layer2.output, mode=mode)

        return self.layer3.output


    def add_to_train(self, new_events):

        if self.events is None:
            self.events = new_events
        else :
            self.events.extend(new_events)

if __name__ == "__main__":
    camera = Camera(width=34, height=34)
    hots =  Hierarchy(camera)
    hots.create_layer(layer_idx=2, events=None, layer_polarities=None)