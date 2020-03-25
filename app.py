# TODO: when running the app load the following resources:
#  nltk.download('punkt')
#  nltk.download('wordnet')
#  nltk.download('stopwords')
#  load the nn model

from flask import Flask
from flask_restful import reqparse, Api, Resource
import nltk

from src.models import IntentClassifier


app = Flask(__name__)
api = Api(app)

# download nltk data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Define model parameters and load the model
weights_file_path = 'model_params/intent_classifier_0012-1.05701441.h5'
input_tokenizer_file_path = 'model_params/input_tokenizer.pickle'
output_tokenizer_file_path = 'model_params/output_tokenizer.pickle'
training_set_file_path = 'dataset/train.csv'

model = IntentClassifier(
    weights_file_path=weights_file_path,
    training_set_file_path=training_set_file_path,
    input_tokenizer_file_path=input_tokenizer_file_path,
    output_tokenizer_file_path=output_tokenizer_file_path
)
model.load_weights()

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('text')


class PredictIntent(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['text']

        # prediction = model.predict(user_query)
        intent, confidence = model.predict(user_query)

        # round the predict proba value and set to new variable
        confidence = round(confidence, 3)

        # create JSON object
        output = {'intent': intent, 'confidence': str(confidence)}

        return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictIntent, '/')

if __name__ == '__main__':
    app.run(debug=True)