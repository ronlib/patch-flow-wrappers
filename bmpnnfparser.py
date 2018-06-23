import struct
import numpy as np
from PIL import Image

class BmpNNFParser:
    def __init__(self, imagep):
        self.image_ = Image.open(imagep)
        self.h_ = self.image_.height
        self.w_ = self.image_.width
        self.bytes_ = self.image_.tobytes()
        computed_size = self.h_*self.w_*4
        if computed_size != len(self.bytes_):
            raise IndexError('According to header, file size should be %d' % computed_size)
        self.flow_ = np.empty([self.h_, self.w_, 2], dtype=np.int32)
        for y in range(self.h_):
            for x in range(self.w_):
                index = (y*self.w_ + x)*4
                b = self.bytes_[index : index+4]
                pixel_v = struct.unpack('<i', b)[0]
                self.flow_[y][x] = [pixel_v & 0x0fff, pixel_v>>12 & 0x0fff]

    def __getitem__(self, index):
        return self.flow_[index]

    @property
    def height(self):
        return self.h_

    @property
    def width(self):
        return self.w_
