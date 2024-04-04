import joblib
import os

def get_dump_file(filename=""):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
    path = os.path.join(project_dir, f'src/dump_model/{filename}')

    return path


def get_dump_model(filename):
    filename = get_dump_file("logistic_model.joblib")
    model = joblib.load(filename)
    
    return model