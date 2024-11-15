from flask import Flask, render_template, request, jsonify
from scripts.Evaluacion import run_evaluation
#from scripts.Visualizacion import generate_visualizations
from scripts.Logistica import SpamDetector
from scripts.Creacion import generate_tables
from scripts.Preparacion import preparar_dataset
from scipy.io import arff


import os

app = Flask(__name__)

# Configuración de rutas
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_PATH_EVALUACION = os.path.join(BASE_DIR, "datasets/datasets/NSL-KDD/KDDTrain+.arff")
PLOTS_DIR = os.path.join(BASE_DIR, "static/plots")
DATASET_PATH_SPAM = os.path.join(BASE_DIR, "datasets/datasets/trec07p")
INDEX_PATH_SPAM = os.path.join(DATASET_PATH_SPAM, "full/index")

# Instancia del detector de SPAM
spam_detector = SpamDetector(DATASET_PATH_SPAM, INDEX_PATH_SPAM)

# Página principal
@app.route('/')
def index():
    return render_template('index.html')

# Evaluación de resultados
@app.route('/result', methods=['POST'])
def result():
    metrics, plots = run_evaluation(DATASET_PATH_EVALUACION)
    return render_template('result.html', metrics=metrics, plots=plots)

# Visualización de datos
@app.route('/evaluacionDeResultados')
def resultados2():
    return render_template('visualizar.html')

# Modelo de detección de SPAM
@app.route('/run_model', methods=['GET'])
def run_model():
    # Entrenar el modelo
    spam_detector.train(15000)  # Entrena con 15,000 correos
    # Realizar predicciones
    result = spam_detector.predict(5000)  # Predice con 5,000 correos

    # Combinar predicciones y valores reales en una lista de diccionarios
    combined_results = [
        {"pred": pred, "true": true} 
        for pred, true in zip(result['y_pred'], result['y_true'])
    ]
    
    # Pasar datos procesados a la plantilla
    return render_template(
        'logistica.html', 
        accuracy=result['accuracy'], 
        combined_results=combined_results
    )


    # Ruta para la página de creación de transformadores y pipelines
# Ruta para la página de creación de transformadores y pipelines
@app.route("/creacion", methods=["GET"])
def creacion():
    num_rows = request.args.get("num_rows", default=10, type=int)
    tables = generate_tables("datasets/datasets/NSL-KDD/KDDTrain+.arff", num_rows)
    
    return render_template("creacion.html", **tables)

# Ruta para la preparación del dataset
@app.route('/preparacion', methods=['GET'])
def preparacion():
    # Obtener el número de filas a mostrar, por defecto 10
    num_rows = request.args.get('num_rows', default=10, type=int)

    # Preparar los datos con el límite de filas especificado
    tablas = preparar_dataset(num_rows)
    
    return render_template('preparacion.html', tablas=tablas, num_rows=num_rows)

if __name__ == '__main__':
    # Crear directorio para gráficos si no existe
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    app.run(debug=True)