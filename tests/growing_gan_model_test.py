import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras.utils import custom_object_scope
from tensorflow.python.framework import test_util
from tensorflow_core.python.keras.api._v2.keras import layers as tfkl

import dynastes as d
from dynastes.models.growing_gan_models import GrowingGanGenerator

class NOOPLayer(tf.keras.layers.Layer):

    def call(self, inputs, **kwargs):
        return inputs

class Simple2DGrowingGanGenerator(GrowingGanGenerator):

    def interpolate_domain(self, x, y, interp):
        return x + (y - x) * interp

    def get_gan_layer(self, lod) -> tfkl.Layer:
        return tfkl.UpSampling2D(interpolation='bilinear')

    def get_to_domain_layer(self, lod) -> tfkl.Layer:
        return NOOPLayer()

    def get_upscale_domain_layer(self, input_lod, output_lod) -> tfkl.Layer:
        if output_lod == input_lod:
            return NOOPLayer()
        levels = [2] * (output_lod - input_lod)
        scale = np.cumprod(levels)[-1]
        return tfkl.UpSampling2D(size=(scale, scale))

    def get_conform_to_output_layer(self, input_lod) -> tfkl.Layer:
        return self.get_upscale_domain_layer(input_lod, self.n_lods-1)


class GrowingGanGeneratorTest(tf.test.TestCase):
    @test_util.use_deterministic_cudnn
    def test_simple(self):
        with custom_object_scope(d.object_scope):
            z = np.random.random(size=(1,2,2,1))
            gen = Simple2DGrowingGanGenerator(n_lods=3)
            y = gen(z, lod_in=0.)
            print(y)