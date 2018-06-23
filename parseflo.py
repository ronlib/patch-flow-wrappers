import sys
import struct
import os
import numpy as np

class FloParser:
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
                    self.flow[y].append([u, v])
            self.flow = np.array(self.flow).round().astype('int')

    def __getitem__ (self, index):
        return self.flow[index]


def main():
    path = '/home/ron/studies/project/compare_PM_performance/other-gt-flow/Dimetrodon/flow10.flo'

    return 0

if __name__ == '__main__':
    sys.exit(main())
