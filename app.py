from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json

app = Flask(__name__)
CORS(app)

# Cargar el modelo de TensorFlow
modelo = tf.keras.models.load_model('modelo/modeloV1')

# Cargar mapeo de índices de clase a etiquetas de aves desde un archivo JSON
with open('decripcion.json', 'r', encoding='utf-8') as file:
    indice_a_etiqueta = json.load(file)

# Cargar información adicional sobre las especies de aves desde un archivo JSON
with open('decripcion.json', 'r', encoding='utf-8') as file:
    informacion_especies = json.load(file)

def preparar_imagen(imagen, tamaño_objetivo):
    """Preprocesar la imagen para que sea adecuada para el modelo."""
    if imagen.mode != "RGB":
        imagen = imagen.convert("RGB")
    imagen = imagen.resize(tamaño_objetivo)
    imagen = tf.keras.preprocessing.image.img_to_array(imagen)
    imagen = np.expand_dims(imagen, axis=0)
    return imagen

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró el archivo'}), 400

    archivo = request.files['file']
    if archivo.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400

    imagen = Image.open(io.BytesIO(archivo.read()))
    imagen_preparada = preparar_imagen(imagen, (224, 224))  # Ajustar según el tamaño de entrada del modelo
    predicciones = modelo.predict(imagen_preparada)
    clase_predicha = np.argmax(predicciones, axis=1)[0]  # Obtener la clase con mayor probabilidad


    especie_info = informacion_especies.get(str(clase_predicha), {})
    
    response_data = {
            'especieInfo': especie_info
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
