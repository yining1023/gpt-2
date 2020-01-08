# import os

# class TrainingContext(object):
#     def __init__(self, queue):
#         self._queue = queue
#         self._state = {
#             'step': 0,
#             'history': {},
#             'metrics': {},
#             'checkpoints': {},
#             'samples': {},
#             'paths': {}
#         }
#         self.metrics = MetricsProxy(self)
#         self.checkpoints = ArtifactsProxy('checkpoints', self, keep_max=10)
#         self.samples = ArtifactsProxy('samples', self, keep_max=100)

#     def refresh(self):
#         self._queue.put(self._state)

#     def __setattr__(self, key, value):
#         super().__setattr__(key, value)
#         if key == 'step' or key == 'training_status':
#             self._state[key] = value
#             self.refresh()

# def train(datasets, options, ctx):
#     print('Creating training config')
#     result_dir = generate_id()
#     os.makedirs(result_dir)
#     # kwargs = create_training_config(tmp_dataset.name, options['checkpoint'], result_dir, **options)
#     # kwargs.update(max_steps=options['max_steps'])
#     # gen = training_loop(**kwargs)

#     ctx.training_status = 'TRAINING'

#     # for (step, metrics, samples, checkpoints) in gen:
#     #     ctx.step = step
#     #     for k, v in metrics.items():
#     #         ctx.metrics[k] = v
#     #     for k, v in samples.items():
#     #         ctx.samples.add(k, v)
#     #     for k, v in checkpoints.items():
#     #         ctx.checkpoints.add(k, v)

#     print('Training succeeded!')

# def run_training(fn, datasets, options, queue):
#     ctx = TrainingContext(queue)
#     ctx.training_status = 'PREPARING'
#     try:
#         fn(datasets, options, ctx)
#         # Provide some time for the sidecar to upload artifacts
#         time.sleep(60 * 5)
#         ctx.training_status = 'SUCCEEDED'
#     except Exception as e:
#         ctx.training_status = 'FAILED'
#         import traceback
#         print('Training failed with exception:', repr(e))
#         traceback.print_exc()

# from flask import send_file

# import logging
# log = logging.getLogger('werkzeug')
# log.setLevel(logging.ERROR)

# def start_server(q):
#     state = {
#         'step': 0,
#         'trainingStatus': 'PREPARING',
#         'history': {},
#         'metrics': {},
#         'checkpoints': {},
#         'samples': {}
#     }

#     app = Flask('GPT2')

#     metrics = []

#     time_started = time.time()

#     def refresh_state():
#         nonlocal state
#         try:
#             while not q.empty():
#                 state = q.get_nowait()
#         except:
#             return

#     @app.route('/healthcheck', methods=['GET'])
#     def healthcheck_route():
#         return jsonify(dict(success=True))

#     @app.route('/status')
#     def status():
#         refresh_state()
#         return jsonify({
#             'step': state['step'],
#             'trainingStatus': state['training_status'],
#             'metrics': state['metrics'],
#             'timeElapsed': time.time() - time_started
#         })

#     @app.route('/history')
#     def history():
#         refresh_state()
#         return jsonify(state['history'])

#     @app.route('/samples')
#     def samples():
#         refresh_state()
#         return jsonify(state['samples'])
    
#     @app.route('/checkpoints')
#     def checkpoints():
#         refresh_state()
#         return jsonify(state['checkpoints'])

#     @app.route('/samples/<key>/<file_id>')
#     def download_sample(key, file_id):
#         refresh_state()
#         path = state['paths'][file_id]
#         return send_file(path, as_attachment=True)

#     @app.route('/checkpoints/<key>/<file_id>')
#     def download_checkpoint(key, file_id):
#         refresh_state()
#         path = state['paths'][file_id]
#         return send_file(path, as_attachment=True)

#     http_server = WSGIServer(('0.0.0.0', int(os.getenv('RW_PORT', '8000'))), app, log=log)
#     http_server.serve_forever()

# from runway.utils import cast_to_obj

# import sys

# if __name__ == '__main__':
#     schema = {
#         'datasets': {
#             'text': runway.file(extension='.txt')
#         },
#         'options': {
#             'max_steps': runway.number(default=5000, min=1, max=25000, description='Number of training iterations')
#         },
#         'samples': {
#             'generated_text': runway.text,
#         },
#         'metrics': {
#         }
#     }

#     if os.getenv('RW_META', '0') == '1':
#         print(json.dumps(dict(
#             datasets=fields_to_dict(schema['datasets']),
#             options=fields_to_dict(schema['options']),
#             samples=fields_to_dict(schema['samples']),
#             metrics=fields_to_dict(schema['metrics'])
#         )))
#         sys.exit(0)

#     dataset = json.loads(os.getenv('RW_DATASETS', '{}'))
#     dataset['text'] = extract_dataset(dataset['text'])
#     options = json.loads(os.getenv('RW_MODEL_OPTIONS', '{}'))

#     deserialized_opts = {}
#     for name, opt in schema['options'].items():
#         opt = cast_to_obj(opt)
#         opt.name = name
#         if name in options:
#             deserialized_opts[name] = opt.deserialize(options[name])
#         elif hasattr(opt, 'default'):
#             deserialized_opts[name] = opt.default
#         else:
#             raise MissingOptionError(name)
    
#     state = None
#     q = Queue()
#     proc = Process(target=start_server, args=(q,))
#     proc.start()
#     run_training(train, dataset, deserialized_opts, q)