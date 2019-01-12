import struct
import numpy as np
from PIL import Image

class BmpNNFParser:
    """
    This class parses flow .bmp files, and exposes an index access to the parsed
    file.

    p = FloParser('path')
    p[y,x] or p[y][x] returns a 2-tuple, where the first value is the vertical NN,
    location (y value), and the second is the horizontal location (x value).
    """
    def __init__(self, filep):
        self.image_ = Image.open(filep)
        self.h_ = self.image_.height
        self.w_ = self.image_.width
        self.bytes_ = self.image_.tobytes()
        computed_size = self.h_*self.w_*4
        if computed_size != len(self.bytes_):
            raise IndexError('According to header, file size should be %d' % computed_size)
        self.flow = np.empty([self.h_, self.w_, 2], dtype=np.int32)
        for y in range(self.h_):
            for x in range(self.w_):
                index = (y*self.w_ + x)*4
                b = self.bytes_[index : index+4]
                # Extracting one integer
                pixel_v = struct.unpack('<i', b)[0]
                # Parsing according to nn.h:50 macros. [vertical, horizontal]
                self.flow[y][x] = [pixel_v>>12 & 0x0fff, pixel_v & 0x0fff]

    def __getitem__(self, index):
        return self.flow[index]

    @property
    def height(self):
        return self.h_

    @property
    def width(self):
        return self.w_
