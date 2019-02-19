#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

from runway import RunwayModel


gtp2 = RunwayModel()
sess = None

@gtp2.setup
def setup(alpha=0.5):

    global sess
    global output
    global enc
    model_name='117M'
    seed=None
    nsamples=0
    batch_size=1
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
        length = hparams.n_ctx
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    sess = tf.Session()#graph=tf.Graph())
    output = sample.sample_sequence(
        hparams=hparams, length=length,
        start_token=enc.encoder['<|endoftext|>'],
        batch_size=batch_size,
        temperature=temperature, top_k=top_k
    )[:, 1:]
    print('loading at batchsize',batch_size)
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
    saver.restore(sess, ckpt)

    print('bottom1')
    return sess


@gtp2.command('generate', inputs={'input': 'string'}, outputs={'output': 'string'})
def generate(sess, inp):
    input = inp['input']
    print('generate from input', input)
    out = sess.run(output)
    text = enc.decode(out[0])
    print('created text', text)
    return dict(output=text)



if __name__ == '__main__':
    gtp2.run()
