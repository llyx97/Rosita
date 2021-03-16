import json

task_names = ['CoLA', 'SST-2', 'MNLI', 'QNLI', 'QQP']
configs = {}

"""
    Configuration for setting1
"""
config = {}
for task_name in task_names[:3]:
    config[task_name.lower()] = {'teacher_model': 'models/bert_ft/%s'%task_name,
                       'student_model': 'models/prun_bert/%s/a2_l8_f512_e128'%task_name,
                       'data_dir': '../../tinybert/glue/%s'%task_name,
                       'output_dir': 'models/OUTPUT_DIR_FOR_SET1/%s/a2_l8_f512_e128'%task_name,
                       'lr_schedule': 'warmup_linear',
                       'warmup_proportion': 0.1,
                       'eval_step': 500,
                       'learning_rate': 5e-5,
                       'repr_distill': True,
                       'pred_distill': True}
config['cola']['num_train_epochs'] = 50
config['cola']['patience'] = 10
config['sst-2']['num_train_epochs'] = 10
config['mnli']['num_train_epochs'] = 5
configs['setting1'] = config


"""
    Configuration for bert-student
"""
config = {}
for task_name in task_names:
    config[task_name.lower()] = {'teacher_model': 'models/bert_ft/%s'%task_name,
                       'student_model': 'models/bert_pt',
                       'data_dir': '../../tinybert/glue/%s'%task_name,
                       'output_dir': 'models/kd_bert_ft/%s'%task_name,
                       'lr_schedule': 'warmup_linear',
                       'warmup_proportion': 0.1,
                       'eval_step': 500,
                       'learning_rate': 2e-5,
                       'repr_distill': False,
                       'pred_distill': True}
config['cola']['num_train_epochs'] = 5
config['cola']['eval_step'] = 100
config['sst-2']['num_train_epochs'] = 5 
config['mnli']['num_train_epochs'] = 3
config['mnli']['learning_rate'] = 5e-5
configs['bert-student'] = config


"""
    Configuration for setting2
"""
config = {}
for task_name in task_names[:3]:
    config[task_name.lower()] = {'teacher_model': 'models/kd_bert_ft/%s'%task_name,
                       'student_model': 'models/kd_prun_bert/%s/a2_l8_f512_e128'%task_name,
                       'data_dir': '../../tinybert/glue/%s'%task_name,
                       'output_dir': 'models/OUTPUT_DIR_FOR_SET2/%s/a2_l8_f512_e128'%task_name,
                       'lr_schedule': 'warmup_linear',
                       'warmup_proportion': 0.1,
                       'eval_step': 500,
                       'learning_rate': 5e-5,
                       'repr_distill': True,
                       'pred_distill': True}
config['cola']['num_train_epochs'] = 50
config['cola']['patience'] = 10
config['sst-2']['num_train_epochs'] = 10
config['mnli']['num_train_epochs'] = 5
configs['setting2'] = config


"""
    Configuration for setting3
"""
config = {}
for task_name in task_names[:3]:
    config[task_name.lower()] = {'teacher_model': 'models/kd_bert_ft/%s'%task_name,
                       'student_model': 'models/kd_prun_bert/%s/bert-8layer'%task_name,
                       'data_dir': '../../tinybert/glue/%s'%task_name,
                       'output_dir': 'models/OUTPUT_DIR_FOR_SET3/%s/a2_l8_f512_e128'%task_name,
                       'lr_schedule': 'none',
                       'warmup_proportion': 0.,
                       'eval_step': 500,
                       'learning_rate': 1e-5,
                       'prun_period_proportion':0.1,
                       'keep_heads':2,
                       'keep_layers':8,
                       'emb_hidden_dim':128,
                       'ffn_hidden_dim':512,
                       'depth_or_width': 'width',
                       'repr_distill': True,
                       'pred_distill': True}
config['cola']['num_train_epochs'] = 50
config['cola']['patience'] = 10
config['sst-2']['num_train_epochs'] = 10
config['mnli']['num_train_epochs'] = 5
config['mnli']['lr_schedule'] = 'warmup_linear'
config['mnli']['learning_rate'] = 5e-5
configs['setting3'] = config


"""
    Configuration for bert-8layer
"""
config = {}
for task_name in task_names:
    config[task_name.lower()] = {'teacher_model': 'models/kd_bert_ft/%s'%task_name,
                       'student_model': 'models/kd_bert_ft/%s'%task_name,
                       'data_dir': '../../tinybert/glue/%s'%task_name,
                       'output_dir': 'models/kd_prun_bert/%s/bert-8layer_iter_depth'%task_name,
                       'lr_schedule': 'none',
                       'eval_step': 500,
                       'learning_rate': 2e-5,
                       'prun_period_proportion':0.2,
                       'keep_heads':12,
                       'keep_layers':8,
                       'emb_hidden_dim':-1,
                       'ffn_hidden_dim':3072,
                       'depth_or_width': 'depth',
                       'repr_distill': False,
                       'pred_distill': True}
config['cola']['num_train_epochs'] = 10
config['cola']['learning_rate'] = 1e-5
config['sst-2']['num_train_epochs'] = 5
config['sst-2']['prun_period_proportion'] = 0.5
config['mnli']['num_train_epochs'] = 2
config['qnli']['learning_rate'] = 3e-5
config['qnli']['num_train_epochs'] = 3
config['qqp']['num_train_epochs'] = 2
configs['bert-8layer'] = config


"""
    Configuration for setting4
"""
config = {}
for task_name in task_names:
    config[task_name.lower()] = {'teacher_model': 'models/kd_prun_bert/%s/bert-8layer_iter_depth'%task_name,
                       'student_model': 'models/kd_prun_bert/%s/bert-8layer_iter_depth'%task_name,
                       'data_dir': '../../tinybert/glue/%s'%task_name,
                       'output_dir': 'models/setting4/%s'%task_name,
                       'lr_schedule': 'none',
                       'warmup_proportion': 0.,
                       'eval_step': 500,
                       'learning_rate': 1e-5,
                       'prun_period_proportion':0.1,
                       'keep_heads':2,
                       'keep_layers':8,
                       'emb_hidden_dim':128,
                       'ffn_hidden_dim':512,
                       'depth_or_width': 'width',
                       'repr_distill': True,
                       'pred_distill': True}
config['cola']['num_train_epochs'] = 50
config['cola']['patience'] = 10
config['cola']['prun_period_proportion'] = 0.01
config['sst-2']['num_train_epochs'] = 10
config['sst-2']['prun_period_proportion'] = 0.01
config['mnli']['num_train_epochs'] = 5
config['mnli']['lr_schedule'] = 'warmup_linear'
config['mnli']['learning_rate'] = 5e-5
config['qnli']['learning_rate'] = 3e-5
config['qnli']['num_train_epochs'] = 7
config['qqp']['learning_rate'] = 3e-5
config['qqp']['num_train_epochs'] = 5
configs['setting4'] = config

for key, config in configs.items():
    file = open('config_%s.json'%key, 'w')
    json.dump(config, file, indent=1)
