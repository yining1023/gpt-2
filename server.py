#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

import runway
from runway.data_types import number, text


sess = None

@runway.setup(options={'seed': number(default=0, max=999)})
def setup(opts):
    global sess
    global output
    global enc
    model_name='117M'
    seed=opts.get('seed', 0)
    length=None
    temperature=1
    top_k=0

    np.random.seed(seed)
    tf.set_random_seed(seed)

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    sess = tf.Session()
    context = tf.placeholder(tf.int32, [1, None])
    output = sample.sample_sequence(
        hparams=hparams, 
        length=length,
        context=context,
        batch_size=1,
        temperature=temperature,
        top_k=top_k
    )
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
    saver.restore(sess, ckpt)

    return sess, enc, context


@runway.command('generate', inputs=[text(name='prompt')], outputs=[text])
def generate(model, prompt):
    sess, enc, context = model
    context_tokens = enc.encode(prompt)
    out = sess.run(output, feed_dict={context: [context_tokens]})[:, len(context_tokens):]
    result = enc.decode(out[0])
    return result


if __name__ == '__main__':
    runway.run()
