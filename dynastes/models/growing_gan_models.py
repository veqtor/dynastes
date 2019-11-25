import abc

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl


def _call_masked(layer, inputs, training=None, mask=None, **kwargs):
    if layer.supports_masking:
        out = layer(inputs, training=training, mask=mask, **kwargs)
        out_mask = layer.compute_mask(inputs, mask, **kwargs)
    else:
        out = layer(inputs, training=training, **kwargs)
        out_mask = mask
    return out, out_mask


class GrowingGanGenerator(tfk.Model, abc.ABC):
    def __init__(self,
                 n_lods=0,
                 **kwargs):
        super(GrowingGanGenerator, self).__init__(**kwargs)
        self.n_lods = n_lods

    """
    @tf.function
    def get_images_ag(x, lod_in):

        def grow(x, res, lod_in):
            print("grow, res:", 2 ** res, ' lods left: ', lod_in)
            y = block(res, x)

            def get_lod_out():
                y_rgb = torgb(res, y)  # This stage's output in rgb
                if lod_in > lod_in:  # Currently lerping
                    x_rgb = upscale2d(torgb(res - 1, x))
                    z = lerp(y_rgb, x_rgb, lod_in - lod_in)
                else:  # No need to lerp at this stage
                    z = y_rgb
                return upscale2d(z, 2 ** lod_in)  ##Conform to output

            if lod_in > 0:
                if lod_in < lod_in:  # Lod decreases per rec until we're at lod_in
                    return grow(y, res + 1, lod_in - 1)
                else:  # If this stage should output
                    return get_lod_out()
            else:  # We're at lod_in 0 so we return the "lod_in-stack", this would be the innermost call of the recursion loop
                return get_lod_out()

        # Stylegan starts at lod_in 3 = 16x16
        print('resolution_log2', resolution_log2)
        images_out = grow(x, 3, resolution_log2 - 3)
        return images_out
    """

    @abc.abstractmethod
    def interpolate_domain(self, x, y, interp):
        """
        Return result of interpolating between domains
        """

    @staticmethod
    def interpolate_mask(x_mask, y_mask, interp):

        @tf.function
        def interp_masks(_x_mask, _y_mask, _interp):

            if _interp <= 0.:
                return _x_mask
            elif _interp >= 1.:
                return _y_mask
            else:
                return tf.math.logical_and(_x_mask, _y_mask)

        if x_mask is None:
            return y_mask
        else:
            if y_mask is None:
                return x_mask
            else:
                return interp_masks(x_mask, y_mask, interp)

    @abc.abstractmethod
    def get_gan_layer(self, lod) -> tfkl.Layer:
        """ Return processing block here """

    @abc.abstractmethod
    def get_to_domain_layer(self, lod) -> tfkl.Layer:
        """
        Return a layer that transforms output of
        gan_layer at this lod_in into a tensor that
        is compatible with your output domain
        The domain should be the same for every
        lod_in, but doesn't need to be RGB etc, if
        such a conversion is handled by the layer returned
        by get_conform_to_output_layer()
        Lerping between lods is handled in this "domain"
        """

    @abc.abstractmethod
    def get_upscale_domain_layer(self, input_lod, output_lod) -> tfkl.Layer:
        """
        Return a layer that scales input from input_lod
        to output lod_in dimensions, this happens in
        the "domain" space

        caveats:
        Possibly you might have to perform cumsum
        on hparams strides, depending on your architecture
        if it's a simple one, you might just return Upsampling2D here

        """

    @abc.abstractmethod
    def get_conform_to_output_layer(self, input_lod) -> tfkl.Layer:
        """
        Return a layer that scales/transforms input @ input_lod
        to conform exactly to targets
        """

    @tf.function
    def _get_output(self, inputs, lod_in=None, training=None, **kwargs):

        def grow(gen: GrowingGanGenerator, x, current_lod):

            y_layer = gen.get_gan_layer(current_lod)
            y = y_layer(x, training=training, **kwargs)

            lods_left = gen.n_lods - (current_lod + 1)

            def get_lod_output(x, y):
                y_domain = gen.get_to_domain_layer(current_lod)(y, training=training, **kwargs)

                if lod_in > lods_left:

                    x_domain = gen.get_to_domain_layer(current_lod - 1)(x, training=training, **kwargs)

                    x_as_y_domain = gen.get_upscale_domain_layer(current_lod - 1, current_lod)(x_domain,
                                                                                               training=training,
                                                                                               **kwargs)

                    z = gen.interpolate_domain(y_domain, x_as_y_domain, lod_in - lods_left)
                else:
                    z = y_domain

                return gen.get_conform_to_output_layer(current_lod)(z, training=training, **kwargs)

            if lods_left > 0:
                if lod_in < lods_left:
                    return grow(gen, y, current_lod=current_lod + 1)
                else:
                    return get_lod_output(x, y)
            else:
                return get_lod_output(x, y)

        return grow(self, x=inputs, current_lod=0)

    @tf.function
    def _get_output_masked(self, inputs, lod_in=None, training=None, mask=None, **kwargs):

        def grow(gen: GrowingGanGenerator, x, current_lod, mask=None):

            y_layer = gen.get_gan_layer(current_lod)
            y, y_mask = _call_masked(y_layer, x, training=training, mask=mask, **kwargs)

            lods_left = gen.n_lods - (current_lod + 1)

            def get_lod_output(x, y, y_mask=None, x_mask=None):
                y_domain_layer = gen.get_to_domain_layer(current_lod)
                y_domain, y_domain_mask = _call_masked(y_domain_layer, y, training=training, mask=y_mask, **kwargs)

                if lod_in > lods_left:

                    x_domain_layer = gen.get_to_domain_layer(current_lod - 1)
                    x_domain, x_domain_mask = _call_masked(x_domain_layer, x, training=training, mask=x_mask, **kwargs)

                    x_to_y_layer = gen.get_upscale_domain_layer(current_lod - 1, current_lod)
                    x_as_y_domain, x_as_y_mask = _call_masked(x_to_y_layer, x_domain, training=training,
                                                              mask=x_domain_mask, **kwargs)

                    z = gen.interpolate_domain(y_domain, x_as_y_domain, lod_in - lods_left)
                    z_mask = gen.interpolate_mask(y_domain_mask, x_as_y_mask, lod_in - lods_left)
                else:
                    z = y_domain
                    z_mask = y_domain_mask

                r, r_mask = _call_masked(gen.get_conform_to_output_layer(current_lod), z, training=training,
                                         mask=z_mask, **kwargs)
                return r

            if lods_left > 0:
                if lod_in < lods_left:
                    return grow(gen, y, current_lod=current_lod + 1, mask=y_mask)
                else:
                    return get_lod_output(x, y, y_mask=y_mask, x_mask=mask)
            else:
                return get_lod_output(x, y, y_mask=y_mask, x_mask=mask)

        return grow(self, x=inputs, current_lod=0, mask=mask)

    def call(self, inputs, lod_in=None, training=None, mask=None, **kwargs):
        """
        @param lod_in: value between 0. and self.n_lods-1
        @param kwargs: optional arguments passed to every layer on call
        """

        lod_in = tf.maximum(0., tf.minimum(self.n_lods - 1, (self.n_lods-1) - tf.convert_to_tensor(lod_in)))

        if mask is None:
            return self._get_output(inputs=inputs,
                                    lod_in=lod_in,
                                    training=training,
                                    **kwargs)
        else:
            return self._get_output_masked(inputs=inputs,
                                           lod_in=lod_in,
                                           training=training,
                                           mask=mask, **kwargs)


"""
    Discriminator pseudo-tf-function:
    @tf.function
    def get_scores_ag(images_in, lod_in):
    
      def grow(res, lod_in):
        x = fromrgb(res, downscale2d(images_in, 2**lod_in))
        if lod_in > 0:
          if lod_in < lod_in:
            x = grow(res+1, lod_in-1)
        x = disc_block(x, res)
        if res > 2:
          if lod_in > lod_in:
            return lerp(x, fromrgb(downscale2d(images_in, 2**(lod_in+1)), res-1), lod_in - lod_in)
          else:
            return x
        else:
          return x
    
      return grow(2, resolution_log_2 - 2)
"""
