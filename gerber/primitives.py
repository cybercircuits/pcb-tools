#! /usr/bin/env python
# -*- coding: utf-8 -*-

# copyright 2014 Hamilton Kibbe <ham@hamiltonkib.be>

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from operator import sub


class Primitive(object):
    
    def __init__(self, level_polarity='dark'):
        self.level_polarity = level_polarity
        
    def bounding_box(self):
        """ Calculate bounding box

        will be helpful for sweep & prune during DRC clearance checks.

        Return ((min x, max x), (min y, max y))
        """
        pass


class Line(Primitive):
    """
    """
    def __init__(self, start, end, width, level_polarity='dark'):
        super(Line, self).__init__(level_polarity)
        self.start = start
        self.end = end
        self.width = width

    @property
    def angle(self):
        delta_x, delta_y = tuple(map(sub, self.end, self.start))
        angle = math.atan2(delta_y, delta_x)
        return angle

    @property
    def bounding_box(self):
        width_2 = self.width / 2.
        min_x = min(self.start[0], self.end[0]) - width_2
        max_x = max(self.start[0], self.end[0]) + width_2
        min_y = min(self.start[1], self.end[1]) - width_2
        max_y = max(self.start[1], self.end[1]) + width_2
        return ((min_x, max_x), (min_y, max_y))


class Arc(Primitive):
    """
    """
    def __init__(self, start, end, center, direction, width, level_polarity='dark'):
        super(Arc, self).__init__(level_polarity)
        self.start = start
        self.end = end
        self.center = center
        self.direction = direction
        self.width = width

    @property
    def radius(self):
        dy, dx = map(sub, self.start, self.center)
        return math.sqrt(dy**2 + dx**2)

    @property
    def start_angle(self):
        dy, dx = map(sub, self.start, self.center)
        return math.atan2(dx, dy)

    @property
    def end_angle(self):
        dy, dx = map(sub, self.end, self.center)
        return math.atan2(dx, dy)

    @property
    def sweep_angle(self):
        two_pi = 2 * math.pi
        theta0 = (self.start_angle + two_pi) % two_pi
        theta1 = (self.end_angle + two_pi) % two_pi
        if self.direction == 'counterclockwise':
            return abs(theta1 - theta0)
        else:
            theta0 += two_pi
            return abs(theta0 - theta1) % two_pi

    @property
    def bounding_box(self):
        two_pi = 2 * math.pi
        theta0 = (self.start_angle + two_pi) % two_pi
        theta1 = (self.end_angle + two_pi) % two_pi
        points = [self.start, self.end]
        #Shit's about to get ugly...
        if self.direction == 'counterclockwise':
            # Passes through 0 degrees
            if theta0 > theta1:
                points.append((self.center[0] + self.radius, self.center[1]))
            # Passes through 90 degrees
            if theta0 <= math.pi / 2. and (theta1 >= math.pi / 2. or theta1 < theta0):
                points.append((self.center[0], self.center[1] + self.radius))
            # Passes through 180 degrees
            if theta0 <= math.pi and (theta1 >= math.pi or theta1 < theta0):
                points.append((self.center[0] - self.radius, self.center[1]))
            # Passes through 270 degrees
            if theta0 <= math.pi * 1.5 and (theta1 >= math.pi * 1.5 or theta1 < theta0):
                points.append((self.center[0], self.center[1] - self.radius ))
        else:
             # Passes through 0 degrees
            if theta1 > theta0:
                points.append((self.center[0] + self.radius, self.center[1]))
            # Passes through 90 degrees
            if theta1 <= math.pi / 2. and (theta0 >= math.pi / 2. or theta0 < theta1):
                points.append((self.center[0], self.center[1] + self.radius))
            # Passes through 180 degrees
            if theta1 <= math.pi and (theta0 >= math.pi or theta0 < theta1):
                points.append((self.center[0] - self.radius, self.center[1]))
            # Passes through 270 degrees
            if theta1 <= math.pi * 1.5 and (theta0 >= math.pi * 1.5 or theta0 < theta1):
                points.append((self.center[0], self.center[1] - self.radius ))
        x, y = zip(*points)
        min_x = min(x)
        max_x = max(x)
        min_y = min(y)
        max_y = max(y)
        return ((min_x, max_x), (min_y, max_y))


class Circle(Primitive):
    """
    """
    def __init__(self, position, diameter, level_polarity='dark'):
        super(Circle, self).__init__(level_polarity)
        self.position = position
        self.diameter = diameter

    @property
    def radius(self):
        return self.diameter / 2.

    @property
    def bounding_box(self):
        min_x = self.position[0] - self.radius
        max_x = self.position[0] + self.radius
        min_y = self.position[1] - self.radius
        max_y = self.position[1] + self.radius
        return ((min_x, max_x), (min_y, max_y))

    @property
    def stroke_width(self):
        return self.diameter


class Rectangle(Primitive):
    """
    """
    def __init__(self, position, width, height, level_polarity='dark'):
        super(Rectangle, self).__init__(level_polarity)
        self.position = position
        self.width = width
        self.height = height

    @property
    def lower_left(self):
        return (self.position[0] - (self.width / 2.), 
                self.position[1] - (self.height / 2.))

    @property
    def upper_right(self):
        return (self.position[0] + (self.width / 2.), 
                self.position[1] + (self.height / 2.))

    @property
    def bounding_box(self):
        min_x = self.lower_left[0]
        max_x = self.upper_right[0]
        min_y = self.lower_left[1]
        max_y = self.upper_right[1]
        return ((min_x, max_x), (min_y, max_y))

    @property
    def stroke_width(self):
        return max((self.width, self.height))


class Obround(Primitive):
    """
    """
    def __init__(self, position, width, height, level_polarity='dark'):
        super(Obround, self).__init__(level_polarity)
        self.position = position
        self.width = width
        self.height = height

    @property
    def orientation(self):
        return 'vertical' if self.height > self.width else 'horizontal'

    @property
    def lower_left(self):
        return (self.position[0] - (self.width / 2.), 
                self.position[1] - (self.height / 2.))

    @property
    def upper_right(self):
        return (self.position[0] + (self.width / 2.), 
                self.position[1] + (self.height / 2.))

    @property
    def bounding_box(self):
        min_x = self.lower_left[0]
        max_x = self.upper_right[0]
        min_y = self.lower_left[1]
        max_y = self.upper_right[1]
        return ((min_x, max_x), (min_y, max_y))

    @property
    def subshapes(self):
        if self.orientation == 'vertical':
            circle1 = Circle((self.position[0], self.position[1] +
                              (self.height-self.width) / 2.), self.width)
            circle2 = Circle((self.position[0], self.position[1] -
                              (self.height-self.width) / 2.), self.width)
            rect = Rectangle(self.position, self.width,
                            (self.height - self.width))
        else:
            circle1 = Circle((self.position[0] - (self.height - self.width) / 2.,
                              self.position[1]), self.height)
            circle2 = Circle((self.position[0] - (self.height - self.width) / 2.,
                              self.position[1]), self.height)
            rect = Rectangle(self.position, (self.width - self.height),
                            self.height)
        return {'circle1': circle1, 'circle2': circle2, 'rectangle': rect}

class Polygon(Primitive):
    """
    """
    def __init__(self, position, sides, radius, level_polarity='dark'):
        super(Polygon, self).__init__(level_polarity)
        self.position = position
        self.sides = sides
        self.radius = radius

    @property
    def bounding_box(self):
        min_x = self.position[0] - self.radius
        max_x = self.position[0] + self.radius
        min_y = self.position[1] - self.radius
        max_y = self.position[1] + self.radius
        return ((min_x, max_x), (min_y, max_y))


class Region(Primitive):
    """
    """
    def __init__(self, points, level_polarity='dark'):
        super(Region, self).__init__(level_polarity)
        self.points = points

    @property
    def bounding_box(self):
        x_list, y_list = zip(*self.points)
        min_x = min(x_list)
        max_x = max(x_list)
        min_y = min(y_list)
        max_y = max(y_list)
        return ((min_x, max_x), (min_y, max_y))


class Drill(Primitive):
    """
    """
    def __init__(self, position, diameter):
        super(Drill, self).__init__('dark')
        self.position = position
        self.diameter = diameter

    @property
    def radius(self):
        return self.diameter / 2.

    @property
    def bounding_box(self):
        min_x = self.position[0] - self.radius
        max_x = self.position[0] + self.radius
        min_y = self.position[1] - self.radius
        max_y = self.position[1] + self.radius
        return ((min_x, max_x), (min_y, max_y))
