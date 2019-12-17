from flask import Flask, request, abort, Response
import learning as lrn
from bson.json_util import dumps
import pickle

app = Flask(__name__)


# localhost:8000/train
# used to manually train the model
# returns 'successful' on success, otherwise an error
@app.route('/train', methods=['GET'])
def train_user_model():
    try:
        df = lrn.generate_matrix(lrn.pull_all_ratings())
        lrn.train_model(df)
    except Exception as e:
        return abort(Response(e))
    return 'successful'


# localhost:8000/predict?username=username&i_id=item_id
# used to predict a user's rating of a specific item
# returns [rating]
@app.route('/predict', methods=['GET'])
def get_prediction():
    try:
        username = request.args.get('username', default='')
        i_id = request.args.get('i_id', default='1')
        model_file = open('svd_model', 'rb')
        model = pickle.load(model_file)
        model_file.close()
        prediction = lrn.make_item_prediction(username, i_id, model)
        return dumps(prediction)
    except Exception as e:
        return abort(Response(e))


# localhost:8000/outfit_predict?username=username
# used to generate an outfit that a user hasn't voted on, but will very likely ... like
# returns {_id: ???, gender: m, ... }
@app.route('/outfit_predict', methods=['GET'])
def get_outfit_prediction():
    try:
        username = request.args.get('username', default='')
        limit = request.args.get('limit', default='10')
        model_file = open('svd_model', 'rb')
        model = pickle.load(model_file)
        model_file.close()
        prediction = lrn.make_outfit_prediction(username, model, limit)
        return dumps(prediction)
    except Exception as e:
        return abort(Response(e))


@app.route('/', methods=['GET'])
def home():
    return 'hello stranger'


if __name__ == "__main__":
    app.run('localhost', 8000, debug=True)
