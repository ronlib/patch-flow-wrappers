import numpy as np

def image_from_nnf(nnf, im, patch_w):
    ret = np.zeros(np.append(np.array(nnf.flow.shape[:-1])+[patch_w,]*2, [3]), dtype=np.uint64)
    counter = np.zeros(np.array(nnf.flow.shape[:-1])+[patch_w,]*2)
    im = np.array(im, dtype=np.uint64)
    x = []
    for h in range(nnf.flow.shape[0]):
        for w in range(nnf.flow.shape[1]):
            ret[h:h+patch_w, w:w+patch_w] += im[nnf.flow[h,w][0]:nnf.flow[h,w][0]+patch_w, nnf.flow[h,w][1]:nnf.flow[h,w][1]+patch_w][:,:,:3]
            counter[h:h+patch_w, w:w+patch_w] += 1

    counter = counter.clip(1, None)
    counter = np.repeat(counter[:,:], 3).reshape(counter.shape+(3,))
    ret = np.divide(ret, counter)
    return ret
