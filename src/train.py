import yaml
import os
from datetime import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from sklearn.model_selection import StratifiedKFold
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
from optuna import visualization
from optuna.integration import TFKerasPruningCallback
from optuna.trial import TrialState
from tqdm import tqdm

from module.model import vgg16Model, resnet50Model, xceptionModel, nesnatlargeModel, inceptionV3Model, mobileNetModel, cnnModel
from module.util import load_df, load_x_data, load_y_data, f1_m, precision_m, recall_m
# from inference import plot_loss_graph, plot_confusion_matrix, plot_roc_curve, export_train_result
import warnings
warnings.filterwarnings(action='ignore')

TARGET_SIZE = [400, 400, 1]
global x_train
global y_train
global model_name
global epoch
global batch_size
    

def create_optimizer(trial):
    # We optimize the choice of optimizers as well as their parameters.
    kwargs = {}
    optimizer_options = ["RMSprop", "Adam", "SGD"]
    optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
    if optimizer_selected == "RMSprop":
        kwargs["learning_rate"] = trial.suggest_float(
            "rmsprop_learning_rate", 1e-5, 1e-4, log=True
        )
        kwargs["weight_decay"] = trial.suggest_float("rmsprop_weight_decay", 0.85, 0.99)
        kwargs["momentum"] = trial.suggest_float("rmsprop_momentum", 1e-5, 1e-3, log=True)
    elif optimizer_selected == "Adam":
        kwargs["learning_rate"] = trial.suggest_float("adam_learning_rate", 1e-5, 1e-4, log=True)
        kwargs["weight_decay"] = trial.suggest_float("adam_weight_decay", 0.85, 0.99)
    elif optimizer_selected == "SGD":
        kwargs["learning_rate"] = trial.suggest_float(
            "sgd_opt_learning_rate", 1e-5, 1e-4, log=True
        )
        kwargs["momentum"] = trial.suggest_float("sgd_opt_momentum", 1e-5, 1e-3, log=True)

    optimizer = getattr(tf.optimizers, optimizer_selected)(**kwargs)
    return optimizer

def create_model(optimizer, model_name, target_size=TARGET_SIZE):
    
    if model_name == "VGG16":
        model = vgg16Model(target_size)
    elif model_name == "InceptionV3":
        model = inceptionV3Model(target_size)
    elif model_name == "ResNet50":
        model = resnet50Model(target_size)
    elif model_name == 'CNN':
        model = cnnModel(target_size)
    elif model_name == 'Xception':
        model = xceptionModel(target_size)
    elif model_name == 'Nesnatlarge':
        model = nesnatlargeModel(target_size)
    elif model_name == 'Mobile':
        model = mobileNetModel(target_size)
    else:
        print("Invalid model_name")

    # Compile model.
    model.compile(
        optimizer= optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['acc', f1_m, precision_m, recall_m],
    )

    return model

def objective(trial):
    global x_train
    global y_train
    global model_name
    global epoch
    global batch_size
    
    tf.keras.backend.clear_session()
    # os.makedirs(model_save_path + f"/plot/{str(fold_number)}", exist_ok=True)
    optimizer = create_optimizer(trial)
        
    model = create_model(optimizer, model_name, target_size=TARGET_SIZE)
    
    earlystopping = EarlyStopping(monitor = 'val_loss', patience = 10)
    pruning = TFKerasPruningCallback(trial, monitor='val_loss')
    model.fit(x=(x_train[0], x_train[1], x_train[2]),
              y= y_train, validation_split=0.2,
              epochs = epoch, batch_size= batch_size,
              callbacks = [earlystopping, pruning])
    loss, accuracy, f1, precision, recall = model.evaluate(x=(x_train[0], x_train[1], x_train[2]), y=y_train)

    return accuracy

def show_result(study):
    df = study.trials_dataframe()
    df.to_csv('../experiments.csv')
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    with open('../best_params.yaml', 'w') as f:
        yaml.dump(trial.params, f)

def main():
    global x_train
    global y_train
    global model_name
    global epoch
    global batch_size
    
    methods = ["origin", "median_blur", "sobel_masking"]
    with open("../config.yaml", "r") as f:
        data = yaml.full_load(f)
        
    train_path = data["train_path"]
    epoch = data["epoch"]
    batch_size = data["batch_size"]
    model_name = data["model_name"]
    seed = data["seed"]
    
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    
    train_df = load_df(train_path)
    
    x_train = load_x_data(train_path, methods, target_size=TARGET_SIZE, df = train_df)
    
    y_train = load_y_data(train_df)
    
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed), pruner=SuccessiveHalvingPruner())
    study.optimize(objective, n_trials=50, timeout=100)
    show_result(study)
    return 0    

if __name__ == '__main__':
    # config = parse_opt()
    main()
    