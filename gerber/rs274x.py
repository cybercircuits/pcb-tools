#! /usr/bin/env python
# -*- coding: utf-8 -*-

# copyright 2014 Hamilton Kibbe <ham@hamiltonkib.be>
# Modified from parser.py by Paulo Henrique Silva <ph.silva@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" This module provides an RS-274-X class and parser.
"""


import copy
import json
import re
from .gerber_statements import *
from .primitives import *
from .cam import CamFile, FileSettings

def read(filename, bbox=None):
    """ Read data from filename and return a GerberFile

    Parameters
    ----------
    filename : string
        Filename of file to parse

    Returns
    -------
    file : :class:`gerber.rs274x.GerberFile`
        A GerberFile created from the specified file.
    """
    return GerberParser().parse(filename, bbox)


class GerberFile(CamFile):
    """ A class representing a single gerber file

    The GerberFile class represents a single gerber file.

    Parameters
    ----------
    statements : list
        list of gerber file statements

    settings : dict
        Dictionary of gerber file settings

    filename : string
        Filename of the source gerber file

    Attributes
    ----------
    comments: list of strings
        List of comments contained in the gerber file.

    size : tuple, (<float>, <float>)
        Size in [self.units] of the layer described by the gerber file.

    bounds: tuple, ((<float>, <float>), (<float>, <float>))
        boundaries of the layer described by the gerber file.
        `bounds` is stored as ((min x, max x), (min y, max y))

    """
    def __init__(self, statements, settings, primitives, filename=None):
        super(GerberFile, self).__init__(statements, settings, primitives, filename)


    @property
    def comments(self):
        return [comment.comment for comment in self.statements
                if isinstance(comment, CommentStmt)]

    @property
    def size(self):
        xbounds, ybounds = self.bounds
        return (xbounds[1] - xbounds[0], ybounds[1] - ybounds[0])

    @property
    def bounds(self):
        xbounds = [0.0, 0.0]
        ybounds = [0.0, 0.0]
        for stmt in [stmt for stmt in self.statements
                     if isinstance(stmt, CoordStmt)]:
            if stmt.x is not None:
                if stmt.x < xbounds[0]:
                    xbounds[0] = stmt.x
                elif stmt.x > xbounds[1]:
                    xbounds[1] = stmt.x
            if stmt.y is not None:
                if stmt.y < ybounds[0]:
                    ybounds[0] = stmt.y
                elif stmt.y > ybounds[1]:
                    ybounds[1] = stmt.y
        return (xbounds, ybounds)


    def write(self, filename):
        """ Write data out to a gerber file
        """
        with open(filename, 'w') as f:
            for statement in self.statements:
                f.write(statement.to_gerber() + "\n")


class GerberParser(object):
    """ GerberParser
    """
    NUMBER = r"[\+-]?\d+"
    DECIMAL = r"[\+-]?\d+([.]?\d+)?"
    STRING = r"[a-zA-Z0-9_+\-/!?<>”’(){}.\|&@# :]+"
    NAME = r"[a-zA-Z_$][a-zA-Z_$0-9]+"
    FUNCTION = r"G\d{2}"

    COORD_OP = r"D[0]?[123]"

    FS = r"(?P<param>FS)(?P<zero>(L|T))?(?P<notation>(A|I))X(?P<x>[0-7][0-7])Y(?P<y>[0-7][0-7])"
    MO = r"(?P<param>MO)(?P<mo>(MM|IN))"
    IP = r"(?P<param>IP)(?P<ip>(POS|NEG))"
    LP = r"(?P<param>LP)(?P<lp>(D|C))"
    AD_CIRCLE = r"(?P<param>AD)D(?P<d>\d+)(?P<shape>C)[,]?(?P<modifiers>[^,]*)?"
    AD_RECT = r"(?P<param>AD)D(?P<d>\d+)(?P<shape>R)[,](?P<modifiers>[^,]*)"
    AD_OBROUND = r"(?P<param>AD)D(?P<d>\d+)(?P<shape>O)[,](?P<modifiers>[^,]*)"
    AD_POLY = r"(?P<param>AD)D(?P<d>\d+)(?P<shape>P)[,](?P<modifiers>[^,]*)"
    AD_MACRO = r"(?P<param>AD)D(?P<d>\d+)(?P<shape>{name})[,]?(?P<modifiers>[^,]*)?".format(name=NAME)
    AM = r"(?P<param>AM)(?P<name>{name})\*(?P<macro>.*)".format(name=NAME)

    # begin deprecated
    OF = r"(?P<param>OF)(A(?P<a>{decimal}))?(B(?P<b>{decimal}))?".format(decimal=DECIMAL)
    IN = r"(?P<param>IN)(?P<name>.*)"
    LN = r"(?P<param>LN)(?P<name>.*)"
    # end deprecated

    PARAMS = (FS, MO, IP, LP, AD_CIRCLE, AD_RECT, AD_OBROUND, AD_POLY, AD_MACRO, AM, OF, IN, LN)
    PARAM_STMT = [re.compile(r"%{0}\*%".format(p)) for p in PARAMS]

    COORD_STMT = re.compile((
        r"(?P<function>{function})?"
        r"(X(?P<x>{number}))?(Y(?P<y>{number}))?"
        r"(I(?P<i>{number}))?(J(?P<j>{number}))?"
        r"(?P<op>{op})?\*".format(number=NUMBER, function=FUNCTION, op=COORD_OP)))

    APERTURE_STMT = re.compile(r"(?P<deprecated>G54)?D(?P<d>\d+)\*")

    COMMENT_STMT = re.compile(r"G04(?P<comment>[^*]*)(\*)?")

    EOF_STMT = re.compile(r"(?P<eof>M02)\*")

    REGION_MODE_STMT = re.compile(r'(?P<mode>G3[67])\*')
    QUAD_MODE_STMT = re.compile(r'(?P<mode>G7[45])\*')
    
    # Visibility constants used when clipping against a bounding box.
    VIS_NONE = 0
    VIS_START = 1
    VIS_MIDDLE = 2
    VIS_END = 3
    VIS_ALL = 4

    def __init__(self):
        self.settings = FileSettings()
        self.statements = []
        self.primitives = []
        self.apertures = {}
        self.current_region = None
        self.x = 0
        self.y = 0

        self.aperture = 0
        self.interpolation = 'linear'
        self.direction = 'clockwise'
        self.image_polarity = 'positive'
        self.level_polarity = 'dark'
        self.region_mode = 'off'
        self.quadrant_mode = 'multi-quadrant'
        self.step_and_repeat = (1, 1, 0, 0)

    def parse(self, filename, bbox=None):
        fp = open(filename, "r")
        data = fp.readlines()

        for stmt in self._parse(data):
            stmts = self.evaluate(stmt, bbox)
            self.statements.extend(stmts)

        return GerberFile(self.statements, self.settings, self.primitives, filename)

    def dump_json(self):
        stmts = {"statements": [stmt.__dict__ for stmt in self.statements]}
        return json.dumps(stmts)

    def dump_str(self):
        s = ""
        for stmt in self.statements:
            s += str(stmt) + "\n"
        return s

    def _parse(self, data):
        oldline = ''

        for i, line in enumerate(data):
            line = oldline + line.strip()

            # skip empty lines
            if not len(line):
                continue

            # deal with multi-line parameters
            if line.startswith("%") and not line.endswith("%"):
                oldline = line
                continue

            did_something = True  # make sure we do at least one loop
            while did_something and len(line) > 0:
                did_something = False

                # Region Mode
                (mode, r) = _match_one(self.REGION_MODE_STMT, line)
                if mode:
                    yield RegionModeStmt.from_gerber(line)
                    line = r
                    did_something = True
                    continue

                # Quadrant Mode
                (mode, r) = _match_one(self.QUAD_MODE_STMT, line)
                if mode:
                    yield QuadrantModeStmt.from_gerber(line)
                    line = r
                    did_something = True
                    continue

                # coord
                (coord, r) = _match_one(self.COORD_STMT, line)
                if coord:
                    yield CoordStmt.from_dict(coord, self.settings)
                    line = r
                    did_something = True
                    continue

                # aperture selection
                (aperture, r) = _match_one(self.APERTURE_STMT, line)
                if aperture:
                    yield ApertureStmt(**aperture)

                    did_something = True
                    line = r
                    continue

                # comment
                (comment, r) = _match_one(self.COMMENT_STMT, line)
                if comment:
                    yield CommentStmt(comment["comment"])
                    did_something = True
                    line = r
                    continue

                # parameter
                (param, r) = _match_one_from_many(self.PARAM_STMT, line)
                if param:
                    if param["param"] == "FS":
                        stmt = FSParamStmt.from_dict(param)
                        self.settings.zero_suppression = stmt.zero_suppression
                        self.settings.format = stmt.format
                        self.settings.notation = stmt.notation
                        yield stmt
                    elif param["param"] == "MO":
                        stmt = MOParamStmt.from_dict(param)
                        self.settings.units = stmt.mode
                        yield stmt
                    elif param["param"] == "IP":
                        yield IPParamStmt.from_dict(param)
                    elif param["param"] == "LP":
                        yield LPParamStmt.from_dict(param)
                    elif param["param"] == "AD":
                        yield ADParamStmt.from_dict(param)
                    elif param["param"] == "AM":
                        yield AMParamStmt.from_dict(param)
                    elif param["param"] == "OF":
                        yield OFParamStmt.from_dict(param)
                    elif param["param"] == "IN":
                        yield INParamStmt.from_dict(param)
                    elif param["param"] == "LN":
                        yield LNParamStmt.from_dict(param)
                    else:
                        yield UnknownStmt(line)
                    did_something = True
                    line = r
                    continue

                # eof
                (eof, r) = _match_one(self.EOF_STMT, line)
                if eof:
                    yield EofStmt()
                    did_something = True
                    line = r
                    continue

                if False:
                    print self.COORD_STMT.pattern
                    print self.APERTURE_STMT.pattern
                    print self.COMMENT_STMT.pattern
                    print self.EOF_STMT.pattern
                    for i in self.PARAM_STMT:
                        print i.pattern

                if line.find('*') > 0:
                    yield UnknownStmt(line)
                    did_something = True
                    line = ""
                    continue

            oldline = line

    def evaluate(self, stmt, bbox=None):
        """ Evaluate Gerber statement and update image accordingly.

        This method is called once for each statement in the file as it
        is parsed.

        Parameters
        ----------
        statement : Statement
            Gerber/Excellon statement to evaluate.

        """
        if isinstance(stmt, CoordStmt):
            return self._evaluate_coord(stmt, bbox)

        elif isinstance(stmt, ParamStmt):
            return self._evaluate_param(stmt)

        elif isinstance(stmt, ApertureStmt):
            return self._evaluate_aperture(stmt)

        elif isinstance(stmt, (RegionModeStmt, QuadrantModeStmt)):
            return self._evaluate_mode(stmt)

        elif isinstance(stmt, (CommentStmt, UnknownStmt, EofStmt)):
            return [stmt,]

        else:
            raise Exception("Invalid statement to evaluate")


    def _define_aperture(self, d, shape, modifiers):
        aperture = None
        if shape == 'C':
            diameter = float(modifiers[0][0])
            aperture = Circle(position=None, diameter=diameter)
        elif shape == 'R':
            width = float(modifiers[0][0])
            height = float(modifiers[0][1])
            aperture = Rectangle(position=None, width=width, height=height)
        elif shape == 'O':
            width = float(modifiers[0][0])
            height = float(modifiers[0][1])
            aperture = Obround(position=None, width=width, height=height)
        self.apertures[d] = aperture

    def _evaluate_aperture(self, stmt):
        self.aperture = stmt.d
        return [stmt,]

    def _evaluate_mode(self, stmt):
        if stmt.type == 'RegionMode':
            if self.region_mode == 'on' and stmt.mode == 'off':
                self.primitives.append(Region(self.current_region, self.level_polarity))
                self.current_region = None
            self.region_mode = stmt.mode
        elif stmt.type == 'QuadrantMode':
            self.quadrant_mode = stmt.mode
        return [stmt,]

    def _evaluate_param(self, stmt):
        if stmt.param == "FS":
            self.settings.zero_suppression = stmt.zero_suppression
            self.settings.format = stmt.format
            self.settings.notation = stmt.notation
        elif stmt.param == "MO":
            self.settings.units = stmt.mode
        elif stmt.param == "IP":
            self.image_polarity = stmt.ip
        elif stmt.param == "LP":
            self.level_polarity = stmt.lp
        elif stmt.param == "AD":
            self._define_aperture(stmt.d, stmt.shape, stmt.modifiers)
        return [stmt,]

    def _evaluate_coord(self, stmt, bbox=None):
        stmts = [stmt,]

        # import pprint
        # pp = pprint.PrettyPrinter(indent=4)
        
        x = self.x if stmt.x is None else stmt.x
        y = self.y if stmt.y is None else stmt.y

        if stmt.function in ("G01", "G1"):
            self.interpolation = 'linear'
        elif stmt.function in ('G02', 'G2', 'G03', 'G3'):
            self.interpolation = 'arc'
            self.direction = ('clockwise' if stmt.function in ('G02', 'G2') else 'counterclockwise')

        if stmt.op == "D01":
            if self.region_mode == 'on':
                if in_bbox((x,y), bbox):
                    if self.current_region is None:
                        self.current_region = [(self.x, self.y), ]
                    self.current_region.append((x, y,))
                else:
                    stmts = []
            else:
                start = (self.x, self.y)
                end = (x, y)
                width = self.apertures[self.aperture].stroke_width
                if self.interpolation == 'linear':
                    self.primitives.append(Line(start, end, width, self.level_polarity))
                    visibility, clipped_start, clipped_end = self.clip(start, end, bbox)
                    new_startpoint_stmt = copy.deepcopy(stmt)
                    new_startpoint_stmt.x = clipped_start[0]
                    new_startpoint_stmt.y = clipped_start[1]
                    new_startpoint_stmt.op = "D02"
                    new_endpoint_stmt = copy.deepcopy(stmt)
                    new_endpoint_stmt.x = clipped_end[0]
                    new_endpoint_stmt.y = clipped_end[1]
                    if visibility == self.VIS_ALL:
                        pass 
                    elif visibility == self.VIS_START:
                        stmts = [new_endpoint_stmt,]
                        # print('> ', stmt.to_gerber())
                        # for s in stmts:
                            # print('<< ', s.to_gerber())
                    elif visibility == self.VIS_MIDDLE:
                        stmts = [new_startpoint_stmt, new_endpoint_stmt,]
                        # print('> ', stmt.to_gerber())
                        # for s in stmts:
                            # print('<< ', s.to_gerber())
                    elif visibility == self.VIS_END:
                        stmts = [new_startpoint_stmt, stmt,]
                        # pp.pprint(clipped_start)
                        # pp.pprint(clipped_end)
                        # print('> ', stmt.to_gerber())
                        # for s in stmts:
                            # print('<< ', s.to_gerber())
                    elif visibility == self.VIS_NONE:
                        stmts = []
                        # print('> ', stmt.to_gerber())
                        # for s in stmts:
                            # print('<< ', s.to_gerber())
                    else:
                        raise exception('Unknown visibility')
                else:
                    center = (start[0] + stmt.i, start[1] + stmt.j)
                    self.primitives.append(Arc(start, end, center, self.direction, width, self.level_polarity))

        elif stmt.op == "D02":
            if not in_bbox((x,y), bbox):
                stmts = []

        elif stmt.op == "D03":
            if not in_bbox((x,y), bbox):
                stmts = []
            primitive = copy.deepcopy(self.apertures[self.aperture])
            # XXX: temporary fix because there are no primitives for Macros and Polygon
            if primitive is not None:
                primitive.position = (x, y)
                primitive.level_polarity = self.level_polarity
                self.primitives.append(primitive)

        self.x, self.y = x, y
        return stmts
        
    def clip(self, start, end, bbox):
        '''Return the endpoints of a segment of line from start to end that are visible within the given bbox.'''
            
        if bbox is None:
            return self.VIS_ALL, start, end
            
        # Create a list containing the starting and ending points of the line and
        # where the line crosses all the sides of the bounding box.
        t = [0.0, 1.0]
        t.append(solve_t(start, end, x=bbox[0][0])) # bbox left-side.
        t.append(solve_t(start, end, x=bbox[0][1])) # bbox right-side.
        t.append(solve_t(start, end, y=bbox[1][0])) # bbox bottom.
        t.append(solve_t(start, end, y=bbox[1][1])) # bbox top.
        t = [t_i for t_i in t if 0 <= t_i <= 1] # Remove points not between start and end.
        assert(len(t) >= 2) # At least the start and end points must be in the list.
        t.sort() # Sort the points in increasing distance from the starting point.

        start_vis, end_vis = start, end
        visibility = self.VIS_NONE
        for i, t_i in list(enumerate(t))[:-1]:
            if abs(t_i - t[i+1]) > 0.0001:
                if in_bbox(solve_xy(start, end, t_i), bbox):
                    if in_bbox(solve_xy(start, end, t[i+1]), bbox):
                        start_vis = solve_xy(start, end, t_i)
                        end_vis = solve_xy(start, end, t[i+1])
                        if t_i < 0.0001:
                            if t[i+1] > 1.0-0.0001:
                                visibility = self.VIS_ALL
                            else:
                                visibility = self.VIS_START
                        else:
                            if t[i+1] > 1.0-0.0001:
                                visibility = self.VIS_END
                            else:
                                visibility = self.VIS_MIDDLE
                        break
        return visibility, start_vis, end_vis

def _match_one(expr, data):
    match = expr.match(data)
    if match is None:
        return ({}, None)
    else:
        return (match.groupdict(), data[match.end(0):])


def _match_one_from_many(exprs, data):
    for expr in exprs:
        match = expr.match(data)
        if match:
            return (match.groupdict(), data[match.end(0):])

    return ({}, None)
        
def in_bbox(pt, bbox):
    '''Return true if the given point is inside the given bounding box.'''
    e = 0.000001 # Small value to fudge points near bbox boundary.
    return (bbox is None) or (bbox[0][0]-e <= pt[0] <= bbox[0][1]+e and bbox[1][0]-e <= pt[1] <= bbox[1][1]+e)
    
def solve_t(start, end, x=None, y=None):
    '''Find the fraction of the x or y along the line from start to end.'''
    x0, y0 = start
    x1, y1 = end
    if x is not None:
        try:
            t = (x-x0)/float(x1-x0)
        except:
            t = None
    elif y is not None:
        try:
            t = (y-y0)/float(y1-y0)
        except:
            t = None
    else:
        t = None
    return t
    
def solve_xy(start, end, t):
    '''Find the point (x,y) given the fraction along the line from start to end.'''
    x0, y0 = start
    x1, y1 = end
    if t is not None:
        x = x0 + t * (x1-x0)
        y = y0 + t * (y1-y0)
    else:
        x, y = None, None
    return x, y
