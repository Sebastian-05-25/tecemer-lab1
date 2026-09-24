import os
import sys

# Agregar el directorio raíz de semana04 a la ruta de búsqueda de módulos
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import numpy as np
import pytest
from preparar_dataset import calcular_dia_lluvioso


def test_calcular_dia_lluvioso():
    assert calcular_dia_lluvioso(0.0) == 0
    assert calcular_dia_lluvioso(0.1) == 1
    assert calcular_dia_lluvioso(15.5) == 1


def test_dataset_preparado_existencia():
    datos = np.load("dataset_preparado.npz")
    assert "X_train" in datos
    assert "X_test" in datos
    assert "y_train" in datos
    assert "y_test" in datos
    assert "media" in datos
    assert "desviacion" in datos


def test_dimensiones_dataset():
    datos = np.load("dataset_preparado.npz")
    X_train = datos["X_train"]
    X_test = datos["X_test"]
    assert X_train.shape[1] == 3
    assert X_test.shape[1] == 3
