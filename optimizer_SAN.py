import logging
import sys
import optuna

import numpy as np
import time

from train_optuna import train_test_SAN_model

work_dir = ''
#log_file_name = 'optimize_HME_100k_reduced_SAN'
log_file_name = 'optimize_Base_Soma_Subtr_SAN'

def evaluation_trial(trial):
    model = 'SAN'

    if model == 'SAN':
        decoder_input_size = trial.suggest_categorical('decoder_input_size', [64, 128, 256]) # default: 256

        params = dict(experiment='SAN', epoches=500,
                      #batch_size=trial.suggest_categorical('batch_size', [4, 8, 16]), # default: 8
                      batch_size=trial.suggest_categorical('batch_size', [8]),  # default: 8
                      workers=0,

                      optimizer='Adadelta',
                      lr=1,
                      lr_decay='cosine',
                      eps='1e-6',
                      weight_decay='1e-4',

                      image_width=200, image_height=200, image_channel=1, dropout=True, dropout_ratio=0.5, relu=True,
                      gradient=100, gradient_clip=True, use_label_mask=False,
                      train_image_path='data/train_image.pkl',
                      train_label_path='data/train_label.pkl',
                      eval_image_path='data/test_image.pkl',
                      eval_label_path='data/test_label.pkl',
                      word_path='data/word.txt',
                      encoder={'net': 'DenseNet', 'input_channels': 1, 'out_channels': 684}, resnet={'conv1_stride': 1},
                      densenet={'ratio': 16,
                                #'three_layers': trial.suggest_categorical('three_layers', [True, False]), # default: True
                                'three_layers': trial.suggest_categorical('three_layers', [True]),
                                # default: True
                                'nDenseBlocks': trial.suggest_categorical('nDenseBlocks', [4, 8, 16]),  # default: 16
                                'growthRate': trial.suggest_categorical('growthRate', [8, 16, 24]), # default: 24
                                'reduction': trial.suggest_categorical('reduction', [0.1, 0.2, 0.5]), # default: 0.5
                                'bottleneck': trial.suggest_categorical('bottleneck', [True, False]), # default: True
                                'use_dropout': trial.suggest_categorical('use_dropout', [True, False]) # default: True
                                },
                      decoder={'net': 'SAN_decoder', 'cell': 'GRU', 'input_size': decoder_input_size, 'hidden_size': decoder_input_size},
                      # attention={'attention_dim': trial.suggest_categorical('attention_dim', [128, 256, 512]), # default: 512
                      #            'attention_ch': trial.suggest_categorical('attention_ch', [8, 16, 32]), # default: 32
                      #            },
                      attention={'attention_dim': trial.suggest_categorical('attention_dim', [128, 256, 512]),
                                 # default: 512
                                 #'attention_ch': trial.suggest_categorical('attention_ch', [8, 16, 32]),  # default: 32
                                 'attention_ch': trial.suggest_categorical('attention_ch', [32]),  # default: 32
                                 },
                      hybrid_tree={'threshold': 0.5}, optimizer_save=True,
                      checkpoint_dir='checkpoints', finetune=False,
                      checkpoint='',
                      data_augmentation=trial.suggest_categorical('data_augmentation', [10, 100]), # default 0
                      log_dir='logs')

        print(params)

        trial.set_user_attr("params", params)

        test_exp_rate, train_exp_rate = train_test_SAN_model(params=params)

        trial.set_user_attr("train_exp_rate", train_exp_rate)

    return test_exp_rate


optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = log_file_name
storage_name = "sqlite:///{}.db".format(work_dir + study_name)
study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction='maximize',
                            sampler=optuna.samplers.TPESampler())

print('Trials:', len(study.trials))

study.optimize(evaluation_trial, n_trials=1000)

print("Best params: ", study.best_params)
print("Best value: ", study.best_value)
print("Best Trial: ", study.best_trial)
print("Trials: ", study.trials)
