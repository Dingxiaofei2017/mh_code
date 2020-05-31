import numpy
import tensorflow as tf

def convert_and_cast(value, name, dtype):
    return tf.cast(tf.convert_to_tensor(value, name=name), dtype=dtype)


def _prepare_labels_logits_weights(labels, logits, weights):
    logits = tf.convert_to_tensor(logits, name='logits')
    labels = convert_and_cast(labels, 'labels', logits.dtype.base_dtype)
    weights = convert_and_cast(
        weights, 'weights', logits.dtype.base_dtype)

    try:
        labels.get_shape().merge_with(logits.get_shape())
    except ValueError:
        raise ValueError(
            'logits and labels must have the same shape (%s vs %s)' %
            (logits.get_shape(), labels.get_shape()))

    original_shape = labels.get_shape().as_list()
    if labels.get_shape().ndims > 0:
        original_shape[0] = -1
    if labels.get_shape().ndims <= 1:
        labels = tf.reshape(labels, [-1, 1])
        logits = tf.reshape(logits, [-1, 1])

    if weights.get_shape().ndims == 1:
        # Weights has shape [batch_size]. Reshape to [batch_size, 1].
        weights = tf.reshape(weights, [-1, 1])
    if weights.get_shape().ndims == 0:
        # Weights is a scalar. Change shape of weights to match logits.
        weights *= tf.ones_like(logits)

    return labels, logits, weights, original_shape

def expand_outer(tensor, rank):
    if tensor.get_shape().ndims is None:
        raise ValueError('tensor dimension must be known.')
    if len(tensor.get_shape()) > rank:
        raise ValueError(
            '`rank` must be at least the current tensor dimension: (%s vs %s).' %
            (rank, len(tensor.get_shape())))
    while len(tensor.get_shape()) < rank:
        tensor = tf.expand_dims(tensor, 0)
    return tensor

def prepare_loss_args(labels, logits, positive_weights, negative_weights):
    logits = tf.convert_to_tensor(logits, name='logits')
    labels = convert_and_cast(labels, 'labels', logits.dtype)
    if len(labels.get_shape()) == 2 and len(logits.get_shape()) == 3:
        labels = tf.expand_dims(labels, [2])

    positive_weights = convert_and_cast(positive_weights, 'positive_weights',
                                        logits.dtype)
    positive_weights = expand_outer(positive_weights, logits.get_shape().ndims)
    negative_weights = convert_and_cast(negative_weights, 'negative_weights',
                                        logits.dtype)
    negative_weights = expand_outer(negative_weights, logits.get_shape().ndims)
    return labels, logits, positive_weights, negative_weights

def weighted_hinge_loss(labels,
                        logits,
                        positive_weights=1.0,
                        negative_weights=1.0,
                        name=None):
    with tf.name_scope(
            name, 'weighted_hinge_loss',
            [logits, labels, positive_weights, negative_weights]) as name:
        labels, logits, positive_weights, negative_weights = prepare_loss_args(
            labels, logits, positive_weights, negative_weights)

        positives_term = positive_weights * \
            labels * tf.maximum(1.0 - logits, 0)
        negatives_term = (negative_weights * (1.0 - labels)
                          * tf.maximum(1.0 + logits, 0))
        return positives_term + negatives_term

def weighted_sigmoid_cross_entropy_with_logits(labels,
                                               logits,
                                               positive_weights=1.0,
                                               negative_weights=1.0,
                                               name=None):
    with tf.name_scope(
        name,
        'weighted_logistic_loss',
            [logits, labels, positive_weights, negative_weights]) as name:
        labels, logits, positive_weights, negative_weights = prepare_loss_args(
            labels, logits, positive_weights, negative_weights)

        softplus_term = tf.add(tf.maximum(-logits, 0.0),
                               tf.log(1.0 + tf.exp(-tf.abs(logits))))
        weight_dependent_factor = (
            negative_weights + (positive_weights - negative_weights) * labels)
        return (negative_weights * (logits - labels * logits) +
                weight_dependent_factor * softplus_term)

def weighted_surrogate_loss(labels,
                            logits,
                            surrogate_type='xent',
                            positive_weights=1.0,
                            negative_weights=1.0,
                            name=None):
    with tf.name_scope(
        name, 'weighted_loss',
        [logits, labels, surrogate_type, positive_weights,
         negative_weights]) as name:
        if surrogate_type == 'xent':
            return weighted_sigmoid_cross_entropy_with_logits(
                logits=logits,
                labels=labels,
                positive_weights=positive_weights,
                negative_weights=negative_weights,
                name=name)
        elif surrogate_type == 'hinge':
            return weighted_hinge_loss(
                logits=logits,
                labels=labels,
                positive_weights=positive_weights,
                negative_weights=negative_weights,
                name=name)
        raise ValueError('surrogate_type %s not supported.' % surrogate_type)



def roc_auc_loss(
        labels,
        logits,
        weights=1.0,
        surrogate_type='xent',
        scope=None):
    with tf.name_scope(scope, 'roc_auc', [labels, logits, weights]):
        # Convert inputs to tensors and standardize dtypes.
        labels, logits, weights, original_shape = _prepare_labels_logits_weights(
            labels, logits, weights)

        # Create tensors of pairwise differences for logits and labels, and
        # pairwise products of weights. These have shape
        # [batch_size, batch_size, num_labels].
        logits_difference = tf.expand_dims(
            logits, 0) - tf.expand_dims(logits, 1)
        labels_difference = tf.expand_dims(
            labels, 0) - tf.expand_dims(labels, 1)
        weights_product = tf.expand_dims(
            weights, 0) * tf.expand_dims(weights, 1)

        signed_logits_difference = labels_difference * logits_difference
        raw_loss = weighted_surrogate_loss(
            labels=tf.ones_like(signed_logits_difference),
            logits=signed_logits_difference,
            surrogate_type=surrogate_type)
        weighted_loss = weights_product * raw_loss

        # Zero out entries of the loss where labels_difference zero (so loss is only
        # computed on pairs with different labels).
        loss = tf.reduce_mean(tf.abs(labels_difference)
                              * weighted_loss, 0) * 0.5
        loss = tf.reshape(loss, original_shape)
        return loss