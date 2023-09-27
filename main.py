from flask import Flask, request, jsonify
import pickle
import torch

# loading model here
model = pickle.load(open('T05_InsuranceModel.pkl', 'rb'))

app = Flask(__name__)

str = "HELLO THIS IS ARYAN PRAJAPATI\nDeployed DL model to PREDICT Health Insurance COST"


@app.route('/')
def home():
    predictInsuranceCost()
    return str


@app.route('/predictInsuranceCost', methods=['GET'])
def predictInsuranceCost():
    age = request.form.get('age')
    sex = request.form.get('sex')
    height = request.form.get('height')
    weight = request.form.get('weight')
    children = request.form.get('children')
    smoker = request.form.get('smoker')

    input_list = [int(age)]
    if sex.lower() == 'female':
        input_list.append(0)
    else:
        input_list.append(1)
    input_list.append(bodyMassIndex(float(height), float(weight)))
    input_list.append(int(children))
    if smoker.lower() == 'yes':
        input_list.append(1)
    else:
        input_list.append(0)

    result = model(torch.tensor(input_list, dtype=torch.float64))[0]
    return jsonify(int(round(result.item(), 0)))


def bodyMassIndex(height, weight):
    return round((weight / height ** 2), 2)


if __name__ == '__main__':
    app.run(debug=True)
