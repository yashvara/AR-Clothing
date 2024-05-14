from flask import Flask, render_template
import user_interface

app = Flask(__name__)

@app.route('/')
def index():
    # Render the Tkinter GUI and convert it to a string
    ui_html = user_interface.create_ui()

    # Render the HTML template with the Tkinter GUI included
    return render_template('index.html', ui_html=ui_html)

if __name__ == "__main__":
    app.run(debug=True)
