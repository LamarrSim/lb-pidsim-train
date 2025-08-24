import os 
from glob import glob
import re
from tempfile import NamedTemporaryFile


__GPU_REQUEST__ = None

def request_gpu(gpu_needed: bool):
    global __GPU_REQUEST__
    
    if __GPU_REQUEST__ is not None:
        if gpu_needed: 
            return __GPU_REQUEST__.gpu_id
        else:
            raise RuntimeError("request_gpu cannot be used to dismiss a GPU")
            
    if not gpu_needed:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
        return None
    
    __GPU_REQUEST__ = GpuRequest()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(__GPU_REQUEST__.gpu_id)
    
    return __GPU_REQUEST__.gpu_id
 
    


class GpuRequest:
    def __init__(self):
        files = glob(self.make_filename('*') + "*")
        gpu_occupied = [int(re.findall(self.make_filename(r'(\d+)'), f)[0]) for f in files]
        self._filename = None
        for i_gpu in range(16):
            if i_gpu not in gpu_occupied:
                self._gpu = i_gpu
                dirname, fname = os.path.split(self.make_filename(i_gpu))
                os.makedirs(dirname, exist_ok=True)
                self._file = NamedTemporaryFile(dir=dirname, prefix=fname, suffix="")
                break

    def make_filename(self, gpu: str | int):
        dirname = os.environ.get("GPU_PICKER_DIR_PATH", "/tmp")
        return os.path.join(dirname, rf'.gpu-picker-{str(gpu)}.json')
    
    @property
    def gpu_id(self):
        return self._gpu


if __name__ == '__main__':
    print ("Selected GPU:", request_gpu(True))
    import time
    time.sleep (30)
    
    import tensorflow as tf
    tf.test.gpu_device_name()
    
    