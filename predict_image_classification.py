"""
Image classification with Azure Custom Vision
"""
import json
import argparse
from azure.cognitiveservices.vision.customvision.training import (
    CustomVisionTrainingClient,
)

from azure.cognitiveservices.vision.customvision.prediction import (
    CustomVisionPredictionClient,
)
from msrest.authentication import ApiKeyCredentials

# Now there is a trained endpoint that can be used to make a prediction


def parse_args():
    """
    Parse argument
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="image path", type=str)
    parser.add_argument(
        "-c",
        "--config",
        help="cofigure file path",
        type=str,
        default="image_classification_config.json",
    )
    args = parser.parse_args()
    return args


def get_project_id(config):
    """
    Get project ID list
    """
    credentials = ApiKeyCredentials(in_headers={"Training-key": config["training_key"]})
    trainer = CustomVisionTrainingClient(config["ENDPOINT"], credentials)
    project_list = trainer.get_projects()
    project_id = {}
    for i in project_list:
        temp = i.as_dict()
        project_id[temp["name"]] = temp["id"]
        project_id[temp["name"]] = temp["id"]
    return project_id


def main():
    """
    Image classification
    """
    args = parse_args()
    config = json.load(open(args.config, "r"))

    prediction_credentials = ApiKeyCredentials(
        in_headers={"Prediction-key": config["prediction_key"]}
    )
    predictor = CustomVisionPredictionClient(config["ENDPOINT"], prediction_credentials)
    project_id = get_project_id(config)
    with open(args.image, "rb") as image_contents:
        results = predictor.classify_image(
            project_id[config["project_name"]],
            config["publish_iteration_name"],
            image_contents.read(),
        )

        # Display the results.
        for prediction in results.predictions:
            print(
                "{0}: {1:.2f}%".format(
                    prediction.tag_name, prediction.probability * 100
                )
            )


if __name__ == "__main__":
    main()
