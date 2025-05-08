import importlib
import yaml

def load_config():
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)

def dynamic_import(module_path, function_names):
    """
    YAML 설정을 기반으로 동적으로 모듈과 함수를 임포트합니다.
    """
    module = importlib.import_module(module_path)
    functions = {name: getattr(module, name) for name in function_names}
    return functions
