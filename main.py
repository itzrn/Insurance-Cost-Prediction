from flask import Flask, request, jsonify
import pickle
import torch

# loading model here
model = pickle.load(open('T05_InsuranceModel.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return "HELLO WORLD : THIS IS ARYAN PRAJAPATI\nDEPLOYED A DL MODEL FOR PRACTICE\nTHIS MODEL PREDICT HEALTH " \
           "INSURANCE COST\n WHERE INPUT FEATURES IS -> [AGE, SEX, BMI, No OF CHILDREN'S HAVE, SMOKER]"


@app.route('/predictInsuranceCost', methods=['POST'])
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
