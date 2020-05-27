from azureml.core import Workspace
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
from azureml.core.webservice import Webservice
from azureml.core.model import Model
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies

state = 1

if (state == 1):
    ws = Workspace.from_config(path="K:\AGorize\RIGA-dataset\config.json")
    # Register model
    model = Model.register(workspace = ws,
                            model_path ="Model.pkl",
                            model_name = "Model",
                            tags = {"CNN": "Classifier"},
                            description = "Retinal Glaucoma image CNN classifier",)

    myenv = CondaDependencies.create(pip_packages=['azureml-defaults', 'torch', 'torchvision>=0.5.0'])

    with open("myenv.yml", "w") as f:
        f.write(myenv.serialize_to_string())
    myenv = Environment.from_conda_specification(name="myenv", file_path="myenv.yml")
    inference_config = InferenceConfig(entry_script="pytorch_score.py", environment=myenv)

    aciconfig = AciWebservice.deploy_configuration(cpu_cores=1,
                                                   memory_gb=1,
                                                   tags={'data': 'Retinal scans',  'method':'Supervised Leanrning', 'framework':'pytorch'},
                                                   description='Retinal Glaucoma image CNN classifier')

    service = Model.deploy(workspace=ws,
                               name='Model',
                               models=[model],
                               inference_config=inference_config,
                               deployment_config=aciconfig)
    service.wait_for_deployment(True)
    print(service.state)
    state = 2

elif (state == 2):
    pass