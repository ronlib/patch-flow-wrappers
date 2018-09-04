import sys
import struct
import os
import numpy as np

class p2vParser:
    """
    This class parses .p2v files, and exposes an index access to the parsed file.

    p = p2vParser('path')
    p[y,x] or p[y][x] returns a 2-tuple, where the first value is the vertical NN,
    location (y value), and the second is the horizontal location (x value).
    """

    P2V_VECTOR_SIZE = 128

    def __init__ (self, filep, h, w):
        with open(filep, 'rb') as f:
            self.h_ = h
            self.w_ = w
            computed_size = self.w_*self.h_*2*4+12
            # if computed_size != os.stat(f.name).st_size:
            #     raise IndexError('According to header, file size should be %d' % computed_size)

            self.flow = []

            for y in range(self.h_):
                self.flow.append([])
                for _ in range(self.w_):
                    vec = struct.unpack('<%df' % (p2vParser.P2V_VECTOR_SIZE,),
                                        f.read(4*p2vParser.P2V_VECTOR_SIZE))
                    self.flow[y].append(vec)
            self.flow = np.array(self.flow).astype('float')

    def __getitem__ (self, index):
        return self.flow[index]
