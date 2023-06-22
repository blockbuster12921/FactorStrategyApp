from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, IntegerField, SubmitField, RadioField, SelectField, BooleanField, DecimalField, HiddenField, FieldList, FormField, DateField, PasswordField
from wtforms import validators, ValidationError

class AddProjectForm(FlaskForm):
    new_name = StringField("Name", validators=[validators.Required()], render_kw={"placeholder": "Name of Project"})
    submit = SubmitField("Add")

class DeleteProjectForm(FlaskForm):
    project_id = HiddenField("ProjectID")
    submit = SubmitField("Delete")

class RenameProjectForm(FlaskForm):
    new_name = StringField("Name", validators=[validators.Required()])
    submit = SubmitField("Save changes")
