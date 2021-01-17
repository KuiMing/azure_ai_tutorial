"""
Hello on Azure machine learning.
"""

from azureml.core import Workspace, Experiment, ScriptRunConfig
import azureml
from azureml.core.authentication import InteractiveLoginAuthentication


def main():
    """
    Hello on Azure machine learning.
    """
    interactive_auth = InteractiveLoginAuthentication(
        tenant_id="9ae3a071-d4ec-4cca-bcbd-2f8d2fa92981"
    )
    work_space = Workspace.from_config(auth=interactive_auth)
    # work_space = Workspace.from_config()
    experiment = Experiment(workspace=work_space, name="hello-experiment")

    config = ScriptRunConfig(
        source_directory=".", script="hello.py", compute_target="cpu-cluster"
    )
    # azureml._restclient.snapshots_client.SNAPSHOT_MAX_SIZE_BYTES = 2000000000

    run = experiment.submit(config)
    aml_url = run.get_portal_url()
    print(aml_url)
    run.wait_for_completion(show_output=True)


if __name__ == "__main__":
    main()