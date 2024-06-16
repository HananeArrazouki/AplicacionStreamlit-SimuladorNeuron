import numpy as np
import inspect
import re

"""
   Clase representa una neurona en una red neuronal.

   Atributos:
   - weights (list): Lista de pesos para la neurona.
   - bias (float): Término de sesgo para la neurona.
   - func (str): Función de activación a utilizar ("sigmoid", "relu", "tanh", "binary_step").
"""
class GenericNeuron:

   """
      Inicializa una neurona con pesos, sesgo y función de activación.

      Parámetros:
      - weights (list): Lista de pesos para la neurona.
      - bias (float): Término de sesgo para la neurona.
      - func (str): Función de activación a utilizar ("sigmoid", "relu", "tanh", "binary_step").
   """
   def __init__(self, weights, bias, func):
      self.weights = weights
      self.bias = bias
      self.func = func

   """
      Devuelve una representación en cadena de la neurona.
   """
   def __str__(self):
      return f"Neurona con pesos: {self.weights}, sesgo = {self.bias} y usando la función {self.func}"

   """
      Calcula la función de activación sigmoide.

      Parámetros:
      - x (float): Valor de entrada.

      Retorna:
      - float: Resultado de la función sigmoide.
   """
   @staticmethod
   def _sigmoid(x):
      return 1 / (1 + np.exp(-x))

   """
      Calcula la función de activación ReLU (Rectified Linear Unit).

      Parámetros:
      - x (float): Valor de entrada.

      Retorna:
      - float: Resultado de la función ReLU.
   """
   @staticmethod
   def _relu(x):
      return max(0, x)

   """
      Calcula la función de activación tangente hiperbólica (tanh).

      Parámetros:
      - x (float): Valor de entrada.

      Retorna:
      - float: Resultado de la función tanh.
   """
   @staticmethod
   def _tanh(x):
      return np.tanh(x)

   """
      Calcula la función de activación escalón binario.

      Parámetros:
      - x (float): Valor de entrada.

      Retorna:
      - int: Resultado de la función escalón binario.
   """
   @staticmethod
   def _binary_step(x):      
      return 1 if x > 0 else 0

   """
      Aplica la función de activación especificada al valor de entrada 'y'.

      Parámetros:
      - y (float): Valor de entrada a transformar.
      - activation_function (str): Nombre de la función de activación.

      Retorna:
      - float: Valor transformado después de aplicar la función de activación.
   """
   @staticmethod
   def _apply_activation(y, activation_function):
      for name, func in inspect.getmembers(GenericNeuron):
         if re.search(activation_function, name):
               return func(y)

   """
      Realiza una predicción basada en los datos de entrada.

      Parámetros:
      - input_data (list): Lista de valores de entrada.

      Retorna:
      - float: Resultado de la predicción.
   """
   def predict(self, input_data):
      y = sum(np.multiply(input_data, self.weights)) + self.bias
      return GenericNeuron._apply_activation(y, self.func)

   """
      Cambia el término de sesgo de la neurona.

      Parámetros:
      - bias (float): Nuevo valor de sesgo.
   """
   def update_bias(self, bias):
      self.bias = bias

   """
      Devuelve un diccionario que contiene los pesos y el sesgo de la neurona.

      Retorna:
      - dict: Un diccionario con las claves 'Pesos' y 'Sesgo'.
   """
   def get_parameters(self):
      return {'Pesos': self.weights, 'Sesgo': self.bias}
