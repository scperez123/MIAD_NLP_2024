from flask import Flask
from flask_restx import Api, Resource
import joblib
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)
api = Api(app)

parser = api.parser()
parser.add_argument('year', type=int, required=True, help='year', location='args')
parser.add_argument('title', type=str, required=True, help='title', location='args')
parser.add_argument('plot', type=str, required=True, help='Plot', location='args')

# Cargar los modelos entrenados y el vectorizador
vectorizer = joblib.load('vectorizer.pkl')
scaler = joblib.load('scaler.pkl')
modelo_regresion = joblib.load('regresion.pkl')
label_binarizer = joblib.load('label_binarizer.pkl')

def process_request(year, title, plot):
    # Vectorizar el plot
    plot_vectorized = vectorizer.transform([plot])
    
    # Normalizar el año
    year_normalized = scaler.transform([[year]])
    
    # Concatenar las características
    features = np.hstack((plot_vectorized.toarray(), year_normalized))
    
    # Predecir los géneros
    prediccion = modelo_regresion.predict(features)
    
    # Decodificar la predicción
    prediccion_decoded = label_binarizer.inverse_transform(prediccion)
    
    return prediccion_decoded[0]

class GenresInfo(Resource):
    @api.expect(parser)
    def get(self):
        args = parser.parse_args()
        year = args['year']
        title = args['title']
        plot = args['plot']
        
        result = process_request(year, title, plot)
        return {
            'description': f"Los géneros son: {', '.join(result)}",
            'result': result
        }

api.add_resource(GenresInfo, '/genres_info')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)