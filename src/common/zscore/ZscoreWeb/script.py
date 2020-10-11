from flask import Flask, request, render_template

from cgmzscore import Calculator


app = Flask(__name__)


@app.route('/')
def customer():
    file_path = '/Users/prajwalsingh/Desktop/ZScore/ZscoreWeb/image/roob_rbpp_151005.jpg'
    return render_template('first.html', user_image=file_path)


@app.route('/success', methods=['POST', 'GET'])
def print_data():
    if request.method == 'POST':
        result = request.form
        print(result['age'])
        v = Calculator().zScore_wfa(weight=str(result['weight']), muac=str(result['muca']), age_in_days=str(
            result['age']), sex=str(result['gender']), height=str(result['height']))
        print(v)
        return render_template("result.html", result=v)


if __name__ == '__main__':
    app.run(debug=True)
