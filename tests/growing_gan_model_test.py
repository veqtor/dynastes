import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope
from tensorflow.python.framework import test_util
from tensorflow_core.python.keras.api._v2.keras import layers as tfkl

import dynastes as d
from dynastes.models.growing_gan_models import GrowingGanGenerator, GrowingGanClassifier


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
        return self.get_upscale_domain_layer(input_lod, self.n_lods - 1)


class Simple2DGrowingGanClassifier(GrowingGanClassifier):

    def interpolate_domain(self, x, y, interp):
        return x + (y - x) * interp

    def get_gan_layer(self, lod) -> tfkl.Layer:
        return tfkl.AveragePooling2D((2, 2), strides=(2, 2))

    def get_from_domain_layer(self, lod) -> tfkl.Layer:
        return NOOPLayer()

    def get_downscale_domain_layer(self, input_lod, output_lod) -> tfkl.Layer:
        print('input_lod, output_lod', input_lod, output_lod)
        if output_lod == input_lod:
            return NOOPLayer()
        levels = [2] * (input_lod - output_lod)
        if len(levels) == 0:
            return NOOPLayer()
        scale = np.cumprod(levels)[-1]
        return tfkl.AveragePooling2D((scale, scale), strides=(scale, scale))

    def get_input_transform_layer(self, lod) -> tfkl.Layer:
        return self.get_downscale_domain_layer(self.n_lods - 1, lod)


class GrowingGanGeneratorTest(tf.test.TestCase):
    @test_util.use_deterministic_cudnn
    def test_simple(self):
        with custom_object_scope(d.object_scope):
            z = np.random.random(size=(1, 2, 2, 1)).astype(np.float32)
            gen = Simple2DGrowingGanGenerator(n_lods=3)
            y = gen(z, lod_in=0.)
            print(y)


class GrowingGanClassifierTest(tf.test.TestCase):
    @test_util.use_deterministic_cudnn
    def test_simple(self):
        with custom_object_scope(d.object_scope):
            in_hw = 16
            n_lods = 3
            z = np.random.random(size=(1, in_hw, in_hw, 1)).astype(np.float32)
            cls = Simple2DGrowingGanClassifier(n_lods=n_lods)
            y = cls(z, lod_in=1.)
            ex_hw = 16 // (2 ** n_lods)
            self.assertShapeEqual(np.ones(shape=(1, ex_hw, ex_hw, 1)), y)


class GrowingEndToEndTest(tf.test.TestCase):
    @test_util.use_deterministic_cudnn
    def test_simple(self):
        with custom_object_scope(d.object_scope):
            n_lods = 3
            z = np.random.random(size=(1, 64, 48, 1)).astype(np.float32)
            z = tf.convert_to_tensor(z)
            gen = Simple2DGrowingGanGenerator(n_lods=n_lods)
            cls = Simple2DGrowingGanClassifier(n_lods=n_lods)
            _z = z
            for i in range(int(n_lods * 2.)):
                _z = gen(cls(_z, lod_in=i / 2.), lod_in=i / 2.)
            self.assertLess(tf.reduce_mean(tf.abs(z - _z)).numpy(), 0.26)
