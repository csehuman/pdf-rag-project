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

def load_modules_from_config():
    """
    config.yaml에 정의된 모든 모듈을 동적으로 임포트합니다.
    """
    config = load_config()
    modules = config.get('modules', {})
    imported_modules = {}

    for key, value in modules.items():
        module_path = value['module']
        functions = value['functions']
        imported_modules[key] = dynamic_import(module_path, functions)

    return imported_modules

