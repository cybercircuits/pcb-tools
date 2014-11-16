import string
import logging
from argparse import ArgumentParser
import gerber.rs274x as RS274X
import gerber.primitives as PRIM

VERSION = "0.1"

p = ArgumentParser(description='Remove elements from a Gerber file that fall outside the bounding box of another Gerber file.')
p.add_argument('-i', '--input', type=str, required=True, help='Gerber file to be clipped.')
p.add_argument('-c', '--clip', type=str, required=True, help='Gerber file which defines the clipping bounding box.')
p.add_argument('-o', '--output', type=str, required=True, help='Clipped Gerber file.')
p.add_argument('-l', '--logfile', type=str, default='./bound.log',
               help='Name of the log file to fill with all the nasty output for debugging and tracing what this program is doing: [%(default)s]'
               )
p.add_argument('-v', '--version', action='version', version='%(prog)s ' + VERSION,
               help='Print the version number of this program and exit.'
               )
args = p.parse_args()

bbox = RS274X.read(args.clip).bounds

import pprint
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(bbox)

clipped_layer = RS274X.read(args.input, bbox)

pp.pprint(clipped_layer.bounds)

clipped_layer.write(args.output)
