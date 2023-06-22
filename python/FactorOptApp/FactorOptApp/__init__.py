# The Flask application

from flask import Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'd923sdj#KS9*d2xkww'  # Used for CSRF

# Import views
from . import views
