import tensorflow as tf
import tensorflow.keras as tfk
import abc


class GanLayer(abc.ABC):

    @abc.abstractmethod
    def call(self, inputs, training=None, mask=None, context=None):
        """ This method is called by the generator """


class LayerFactory(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_layer(self, lod: int) -> GanLayer:
        """ This method should return a layer for given LOD """


class GrowingGanGenerator(tfk.Model):
    def __init__(self,
                 layer_factory: LayerFactory,
                 scale_factory: LayerFactory,
                 output_layer_factory: LayerFactory,
                 min_lod=0,
                 max_lod=0,
                 **kwargs):
        super(GrowingGanGenerator, self).__init__(**kwargs)
        self.layer_factory = layer_factory
        self.scale_factory = scale_factory
        self.output_layer_factory = output_layer_factory
        self.min_lod = min_lod
        self.max_lod = max_lod
        self.process_layers = []
        self.output_layers = []

    def _get_layers(self):
        for i in range(self.min_lod, self.max_lod + 1):
            self.process_layers.append(self.layer_factory.get_layer(i))
            self.output_layers.append(self.output_layer_factory.get_layer(i))

    """
    @tf.function
    def get_images_ag(x, lod_in):

        def grow(x, res, lod):
            print("grow, res:", 2 ** res, ' lods left: ', lod)
            y = block(res, x)

            def get_lod_out():
                y_rgb = torgb(res, y)  # This stage's output in rgb
                if lod_in > lod:  # Currently lerping
                    x_rgb = upscale2d(torgb(res - 1, x))
                    z = lerp(y_rgb, x_rgb, lod_in - lod)
                else:  # No need to lerp at this stage
                    z = y_rgb
                return upscale2d(z, 2 ** lod)  ##Conform to output

            if lod > 0:
                if lod_in < lod:  # Lod decreases per rec until we're at lod_in
                    return grow(y, res + 1, lod - 1)
                else:  # If this stage should output
                    return get_lod_out()
            else:  # We're at lod 0 so we return the "lod-stack", this would be the innermost call of the recursion loop
                return get_lod_out()

        # Stylegan starts at lod 3 = 16x16
        print('resolution_log2', resolution_log2)
        images_out = grow(x, 3, resolution_log2 - 3)
        return images_out
    """

    def call(self, inputs, lod=None, training=None, mask=None, context=None):
        """ Implement this """
