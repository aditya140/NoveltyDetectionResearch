from os.path import join as osj
import os
import torch
import joblib
import shutil
import pickle


MODELS_FOLDER_PATH = "./models"
def save_model(model_type, name, model_data):
    if not os.path.exists(osj(MODELS_FOLDER_PATH,model_type)):
        os.makedirs(osj(MODELS_FOLDER_PATH,model_type))
    model_path = osj(MODELS_FOLDER_PATH,model_type,name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    model = model_data["model"]
    model_conf = model_data["model_conf"]
    Lang = model_data["Lang"]

    torch.save(model.state_dict(), osj(model_path, "weights.pt"))
    with open(osj(model_path , "model_conf.pkl"), "wb") as f:
        pickle.dump(model_conf, f)
    with open(osj(model_path , "lang.pkl"), "wb") as f:
        joblib.dump(Lang, f)
    shutil.make_archive(model_path, "zip", model_path)
    
    return model_path

def save_model_neptune(model_path,logger):
    logger.experiment.log_artifact(model_path+".zip")