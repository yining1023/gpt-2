import sys
sys.path.insert(0, 'src')

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

import runway


sess = None
g = None

@runway.setup(options={'checkpoint_dir': runway.file(is_directory=True)})
def setup(opts):
    global sess
    global output
    global enc
    global g
    length=None
    temperature=1
    top_k=0

    enc = encoder.get_encoder(opts['checkpoint_dir'])
    hparams = model.default_hparams()
    with open(os.path.join(opts['checkpoint_dir'], 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    sess = tf.Session()
    context = tf.placeholder(tf.int32, [1, None])
    output, length_ph = sample.sample_sequence(
        hparams=hparams,
        context=context,
        batch_size=1,
        temperature=temperature,
        top_k=top_k
    )
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(opts['checkpoint_dir'])
    saver.restore(sess, ckpt)

    g = tf.get_default_graph()
    g.finalize()
    return sess, enc, context, length_ph


command_inputs = {
    'prompt': runway.text,
    'seed': runway.number(default=0, min=0, max=999, step=1),
    'sequence_length': runway.number(default=128, min=1, max=256, step=1)
}

@runway.command('generate', inputs=command_inputs, outputs={'text': runway.text})
def generate(model, inputs):
    with g.as_default():
        sess, enc, context, length_ph = model
        seed = inputs['seed']
        np.random.seed(seed)
        tf.set_random_seed(seed)
        context_tokens = enc.encode(inputs['prompt'])
        out = sess.run(output, feed_dict={
            context: [context_tokens],
            length_ph: inputs['sequence_length']
        })
        result = enc.decode(out[0])
        return result.split('<|endoftext|>')[0]


if __name__ == '__main__':
    runway.run(model_options={'checkpoint_dir': './models/1558M'})