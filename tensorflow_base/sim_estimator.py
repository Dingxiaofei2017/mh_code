from __future__ import division
from __future__ import print_function
import tensorflow as tf
import pandas as pd


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=10000, type=int,
                    help='number of training steps')

"""
data reader
"""

feature_name = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "captital_gain",
    "captital_loss",
    "hours_per_week",
    "native_country",
    "income"]


data_train = pd.read_table(
    '../data/adult.data',
    names=feature_name,
    sep=",",
    skipinitialspace=True,
    header=None)
data_test = pd.read_table(
    '../data/adult.test',
    names=feature_name,
    sep=',',
    skipinitialspace=True,
    header=0)
train = data_train.replace(">50K", 1)
train = train.replace("<=50K", 0)
test = data_test.replace(">50K.", 1)
test = test.replace("<=50K.", 0)

train_x, train_y = train, train.pop("income")
test_x, test_y = test, test.pop("income")


# 预留数据input_fn
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


# 特征工程
def create_feature_columns():
    age_seg = [20, 30, 40, 50, 60]
    age = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column(
            "age", dtype=tf.int32), boundaries=age_seg)
    workclass = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_hash_bucket(
            'workclass', hash_bucket_size=9, dtype=tf.string))
    education = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_hash_bucket(
            'education', hash_bucket_size=16, dtype=tf.string))
    education_num = tf.feature_column.numeric_column(
        'education_num', default_value=1)
    marital_status = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_hash_bucket(
            'marital_status', hash_bucket_size=7, dtype=tf.string))
    occupation = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_hash_bucket(
            'occupation', hash_bucket_size=7, dtype=tf.string))
    relationship = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_hash_bucket(
            'relationship', hash_bucket_size=6, dtype=tf.string))
    race = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_hash_bucket(
            'race', hash_bucket_size=6, dtype=tf.string))
    gender = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_hash_bucket(
            'gender', hash_bucket_size=3, dtype=tf.string
        )
    )
    captital_gain = tf.feature_column.numeric_column(
        'captital_gain', default_value=0)
    captital_loss = tf.feature_column.numeric_column(
        'captital_loss', default_value=0)
    hours_per_week = tf.feature_column.numeric_column(
        'hours_per_week', default_value=0
    )
    native_country_fn = tf.feature_column.categorical_column_with_hash_bucket(
        'native_country', hash_bucket_size=42, dtype=tf.string)
    native_country = tf.feature_column.embedding_column(
        native_country_fn, dimension=12, combiner='sum')

    deep_column = [
        age,
        workclass,
        education,
        education_num,
        marital_status,
        occupation,
        relationship,
        race,
        gender,
        captital_gain,
        captital_loss,
        hours_per_week,
        native_country
    ]
    return deep_column


def main(argv):
    args = parser.parse_args(argv[1:])
    classifier = tf.estimator.DNNClassifier(
        feature_columns=create_feature_columns(),
        hidden_units=[50, 20],
        n_classes=2
    )
    classifier.train(
        input_fn=lambda: train_input_fn(
            train_x, train_y, args.batch_size), steps=args.train_steps)

    eval_result = classifier.predict(
        input_fn=lambda: eval_input_fn(test_x, test_y, args.batch_size)
    )



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)


