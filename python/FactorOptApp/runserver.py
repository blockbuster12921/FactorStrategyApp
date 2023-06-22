"""
This script runs the application using a development server.
"""

from FactorOptApp import app
import logging

if __name__ == '__main__':
    app.logger.setLevel(logging.DEBUG)
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host='0.0.0.0', port=5001, debug=True)
