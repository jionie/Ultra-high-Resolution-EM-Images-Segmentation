import shutil
import json
import os

def backup_project_as_zip(project_dir, zip_file):
    assert(os.path.isdir(project_dir))
    assert(os.path.isdir(os.path.dirname(zip_file)))


def write_exp(readme=None, cfg=None, score=None, files=None, save_folder='../exp/tmp'):
    if readme:
        with open(os.path.join(save_folder,'readme.txt'), 'w') as f:
            f.write(readme)
    if cfg:
        with open(os.path.join(save_folder,'cfg.json'), 'w') as f:
            json.dump(cfg, f)
    if score:
        with open(os.path.join(save_folder,'score.json'), 'w') as f:
            json.dump(score, f)
    if files:
        for file in files:
            shutil.copy(file,os.path.join(save_folder,file))
