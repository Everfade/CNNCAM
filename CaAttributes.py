from enum import Enum


class RuleTypes(Enum):
    OuterTotalistic = 'outer_totalistic'
    InnerTotalistic = 'inner_totalistic'
    Default = 'default'


class CaNeighbourhoods(Enum):
    Von_Neumann = 'von_Neumann'
    Moore = 'moore'


def periodic_padding(image, padding=1):
    '''
    Create a periodic padding (wrap) around an image stack, to emulate periodic boundary conditions
    Adapted from https://github.com/tensorflow/tensorflow/issues/956

    If the image is 3-dimensional (like an image batch), padding occurs along the last two axes


    '''
    if len(image.shape) == 2:
        upper_pad = image[-padding:, :]
        lower_pad = image[:padding, :]

        partial_image = tf.concat([upper_pad, image, lower_pad], axis=0)

        left_pad = partial_image[:, -padding:]
        right_pad = partial_image[:, :padding]

        padded_image = tf.concat([left_pad, partial_image, right_pad], axis=1)
        return padded_image
    elif len(image.shape) == 3:
        upper_pad = image[:, -padding:, :]
        lower_pad = image[:, :padding, :]

        partial_image = tf.concat([upper_pad, image, lower_pad], axis=1)

        left_pad = partial_image[:, :, -padding:]
        right_pad = partial_image[:, :, :padding]

        padded_image = tf.concat([left_pad, partial_image, right_pad], axis=2)
        return padded_image

    else:
        assert True, "Input data shape not understood."

    return []

class MemoryTypes(Enum):
    Default = 'default'
    Most_Frequent = 'most_frequent'
    Average = 'average'



