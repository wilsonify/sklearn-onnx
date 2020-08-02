import logging
import os
import pickle

import numpy as np
import onnx
import pandas as pd
import pytest
from onnxruntime import backend
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

from skl2onnx import convert_sklearn, get_latest_tested_opset_version
from skl2onnx.common.data_types import (
    FloatTensorType,
)

TARGET_OPSET = get_latest_tested_opset_version()


def test_smoke():
    logging.info("is anything on fire?")


@pytest.fixture(name='iris_df')
def iris_fixture():
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    target_names_map = dict(enumerate(iris.target_names))
    df['plant_type'] = pd.Series(iris.target).map(target_names_map)
    return df


def test_knn(iris_df):
    folder = os.environ.get("ONNXTESTDUMP", "tests_dump")
    basename = "SklearnKNeighborsClassifierMulti"
    os.makedirs(folder, exist_ok=True)

    x_columns = [
        'sepal length (cm)',
        'sepal width (cm)',
        'petal length (cm)'
    ]
    y_columns = ['plant_type']
    x_nda = iris_df[x_columns].values.astype(np.float32)
    y_1da = iris_df[y_columns].values.astype("<U10")
    model = KNeighborsClassifier()

    model.fit(x_nda, y_1da)
    prediction = [model.predict(x_nda), model.predict_proba(x_nda)]

    model_onnx = convert_sklearn(
        model=model,
        name="KNN classifier multi-class",
        initial_types=[("input", FloatTensorType([None, x_nda.shape[1]]))],
        # target_opset=TARGET_OPSET
    )

    dest_data = os.path.join(folder, f"{basename}.data.pkl")
    with open(dest_data, "wb") as data_file:
        pickle.dump(x_nda, data_file)

    dest_expected = os.path.join(folder, f"{basename}.expected.pkl")
    with open(dest_expected, "wb") as expected_file:
        pickle.dump(prediction, expected_file)

    dest_pkl = os.path.join(folder, f"{basename}.model.pkl")
    with open(dest_pkl, "wb") as pickle_file:
        pickle.dump(model, pickle_file)

    dest_onnx = os.path.join(folder, f"{basename}.model.onnx")
    with open(dest_onnx, "wb") as onnx_file:
        logging.info(f"created {onnx_file}")
        onnx_file.write(model_onnx.SerializeToString())

    onnx_graph = onnx.load(dest_onnx)

    print("doc_string={}".format(onnx_graph.doc_string))
    print("domain={}".format(onnx_graph.domain))
    print("ir_version={}".format(onnx_graph.ir_version))
    print("metadata_props={}".format(onnx_graph.metadata_props))
    print("model_version={}".format(onnx_graph.model_version))
    print("producer_name={}".format(onnx_graph.producer_name))
    print("producer_version={}".format(onnx_graph.producer_version))

    rep = backend.prepare(onnx_graph, 'CPU')
    prediction_from_saved = rep.run(x_nda)

    prediction_from_saved_df = pd.DataFrame(prediction_from_saved[1])
    prediction_from_saved_df.columns = prediction_from_saved_df.columns.map("plant_type_is_{}".format)

    prediction_from_saved_df['plant_type_pred'] = pd.Series(prediction_from_saved[0])
    prediction_from_saved_df
    assert prediction_from_saved_df.shape == (x_nda.shape[0], 4)
