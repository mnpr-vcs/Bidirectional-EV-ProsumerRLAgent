#!/usr/bin/env python
# encoding: utf-8
from flask import Flask, request, jsonify
import onnxruntime as ort
import onnx
import json
import numpy as np

app = Flask(__name__)

onnx_path_ppo = "checkpoints/ppo_model.onnx"
onnx_path_sac = "checkpoints/sac_model.onnx"
onnx_path_td3 = "checkpoints/td3_model.onnx"

onnx_model_ppo = onnx.load(onnx_path_ppo)
onnx_model_sac = onnx.load(onnx_path_sac)
onnx_model_td3 = onnx.load(onnx_path_td3)

onnx.checker.check_model(onnx_model_ppo)
onnx.checker.check_model(onnx_model_sac)
onnx.checker.check_model(onnx_model_td3)

sess_ppo = ort.InferenceSession(onnx_path_ppo)
sess_sac = ort.InferenceSession(onnx_path_sac)
sess_td3 = ort.InferenceSession(onnx_path_td3)


def _get_io_shapes(onnx_model: onnx.ModelProto) -> tuple[str, str]:
    input_shape = (
        onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value,
        onnx_model.graph.input[0].type.tensor_type.shape.dim[1].dim_value,
    )
    output_shape = (
        onnx_model.graph.output[0].type.tensor_type.shape.dim[0].dim_value,
        onnx_model.graph.output[0].type.tensor_type.shape.dim[1].dim_value,
    )

    return input_shape, output_shape


def _scale_action_output(action):
    max_charge_rate = 11
    max_discharge_rate = -11
    return (action + 1) * (
        max_charge_rate - max_discharge_rate
    ) / 2 + max_discharge_rate


@app.route("/", methods=["GET"])
def index():
    """
    Handles the root route ("/") for GET requests.
    Returns:
        - If the request method is "GET", returns a JSON response with information about the available endpoints:
            - "info endpoint": The URL for retrieving model information with the model name as a query parameter.
            - "predict endpoint": The URL for making predictions with the model name as a query parameter.
        - If the request method is not "GET", returns a JSON response with an "error" key indicating the invalid request method.
    """
    if request.method == "GET":
        return jsonify(
            {
                "info endpoint": "/get_model_info?model_name=<model_name>",
                "predict endpoint": "/predict?model_name=<model_name>",
            }
        )
    return jsonify({"error": "Invalid request method"})


@app.route("/get_model_info", methods=["GET"])
def get_model_info():
    """
    Retrieves information about a specific model.
    This function handles the GET request to the "/get_model_info" endpoint. It expects a query parameter
    "model_name" which specifies the name of the model to retrieve information about. The function checks the
    value of "model_name" and returns the corresponding model information if it is valid. If the "model_name"
    is not valid, it returns an error message. The function returns a JSON response with the input shape,
    output shape, and the model name. If the request method is not GET, it returns an error message.
    Parameters:
        - None
    Returns:
        - A JSON response containing the input shape, output shape, and the model name if the "model_name" is
        valid.
        - A JSON response containing an error message if the "model_name" is not valid or if the request method is
        not GET.
    """
    if request.method == "GET":
        model_name = request.args.get("model_name")
        if "ppo" == model_name:
            onnx_model = onnx_model_ppo
        elif "sac" == model_name:
            onnx_model = onnx_model_sac
        elif "td3" == model_name:
            onnx_model = onnx_model_td3
        else:
            return jsonify({"error": "Invalid model name"})
        return jsonify(
            {
                "input_shape": _get_io_shapes(onnx_model)[0],
                "output_shape": _get_io_shapes(onnx_model)[1],
                "model_name": model_name,
            }
        )
    return jsonify({"error": "Invalid request method"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predicts an action based on the input observations and the specified model name.
    This function handles the "/predict" route with the "POST" method. It expects a JSON payload in the request body
    with the "observations" key. The function also expects a query parameter "model_name" in the request URL.
    Parameters:
        None
    Returns:
        - If the request method is "POST" and the model name is valid:
            - A JSON response with the predicted action. The action is obtained by running the input
            observations through the specified model and scaling the output using the `_scale_action_output` function.
        - If the request method is not "POST":
            - A JSON response with an "error" key indicating the invalid request method.
        - If the model name is invalid:
            - A JSON response with an "error" key indicating the invalid model name.
    """
    if request.method == "POST":
        input_data = request.json["observations"]
        model_name = request.args.get("model_name")
        if "ppo" == model_name:
            sess = sess_ppo
        elif "sac" == model_name:
            sess = sess_sac
        elif "td3" == model_name:
            sess = sess_td3
        else:
            return jsonify({"error": "Invalid model name"})

        observation = np.array([input_data]).astype(np.float32)
        outputs = sess.run(None, {"observations": observation})
        action = _scale_action_output(outputs[0].item())

        return jsonify({"action": action})
    return jsonify({"error": "Invalid request method"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
