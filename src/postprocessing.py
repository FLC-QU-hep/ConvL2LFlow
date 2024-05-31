import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline


class Postprocessor:

    def __init__(self,
            num_nodes:int = 128,
            threshold:float=1e-4,
            noise_width:float=0.) -> None:
        self.num_nodes = num_nodes
        self.threshold = threshold
        self.noise_width = noise_width
        self.nodes = None
        self.spline = None

    def __call__(self, x:npt.NDArray) -> npt.NDArray:
        ret = np.zeros_like(x)
        mask = x>self.threshold
        ret[mask] = self.p(self.spline(self.p(x[mask])),inverse=True)
        ret[ret<self.threshold] = 0.
        return ret

    def p(self, x:npt.NDArray, inverse:bool=False) -> npt.NDArray:
        if not inverse:
            x = np.log(x)
            x = x - np.log(self.threshold)
            x = x/(np.log(self.maxv)-np.log(self.threshold))
        else:
            x = x*(np.log(self.maxv)-np.log(self.threshold))
            x = x + np.log(self.threshold)
            x = np.exp(x)
        return x

    def init_spline(self,
            reference:npt.NDArray,
            generated:npt.NDArray) -> None:
        reference = reference + self.noise_width*np.random.random(reference.shape)
        reference = reference[reference>self.threshold]
        generated = generated[generated>self.threshold]

        reference = np.random.permutation(reference)
        generated = np.random.permutation(generated)

        length = min(len(reference), len(generated))
        reference = reference[:length]
        generated = generated[:length]

        reference = np.sort(reference)
        generated = np.sort(generated)

        self.maxv = np.max(generated)

        reference = self.p(reference)
        generated = self.p(generated)

        nodes = [[0.,0.]]
        for i in range(self.num_nodes):
            x1 = i/self.num_nodes
            x2 = (i+1)/self.num_nodes
            if i != self.num_nodes-1:
                mask = (generated>=x1)&(generated<x2)
            else:
                mask = (generated>=x1)
            if np.count_nonzero(mask)==0:
                continue
            nodes.append([
                np.mean(generated[mask], axis=0),
                np.mean(reference[mask], axis=0)
            ])
        self.nodes = np.array(nodes)
        self.spline = CubicSpline(self.nodes[:,0], self.nodes[:,1])

    def load_spline(self, spline_file:str):
        data = np.load(spline_file)
        self.nodes = data[:-1].reshape(-1,2)
        self.maxv = data[-1]
        self.spline = CubicSpline(self.nodes[:,0], self.nodes[:,1])

    def save_spline(self, spline_file:str):
        data = np.zeros((2*len(self.nodes)+1,))
        data[:-1] = self.nodes.flatten()
        data[-1] = self.maxv
        np.save(spline_file, data)
