import sys
import struct
import os
import numpy as np

class FloParser:
    """
    This class parses .flo files, and exposes an index access to the parsed file.

    p = FloParser('path')
    p[y,x] or p[y][x] returns a 2-tuple, where the first value is the vertical NN,
    location (y value), and the second is the horizontal location (x value).
    """
    def __init__ (self, filep):
        with open(filep, 'rb') as f:
            header = struct.unpack('<f', f.read(4))[0]
            if (header != 202021.25):
                raise ValueError("Incorrect header value")

            self.w_ = struct.unpack('<i', f.read(4))[0]
            self.h_ = struct.unpack('<i', f.read(4))[0]
            computed_size = self.w_*self.h_*2*4+12
            if computed_size != os.stat(f.name).st_size:
                raise IndexError('According to header, file size should be %d' % computed_size)

            self.flow = []

            for y in range(self.h_):
                self.flow.append([])
                for x in range(self.w_):
                    u, v = struct.unpack('<2f', f.read(8))
                    self.flow[y].append([v, u])
            self.flow = np.array(self.flow).round().astype('int')
            grid1, grid2 = np.meshgrid(range(self.w_), range(self.h_))
            # .flo format contains distance from the current pixel, so we
            # convert it to an absolute location
            self.flow[:,:,0] = self.flow[:,:,0] + grid2
            self.flow[:,:,1] = self.flow[:,:,1] + grid1


    def __getitem__ (self, index):
        return self.flow[index]
