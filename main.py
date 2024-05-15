import user_interface
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    # Render the HTML template without the button initially
    return render_template('index.html')

@app.route('/gui')
def gui():
    # Render the Tkinter GUI
    ui_html = user_interface.create_ui()
    return ui_html

if __name__ == "__main__":
    app.run(debug=True)
