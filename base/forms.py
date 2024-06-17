from django.forms import (
    DateTimeInput,
    ModelForm,
    ModelChoiceField,
    Form,
    TextInput,
    PasswordInput,
    EmailInput,
    Select,
)
from django.contrib.auth.forms import UserCreationForm
from . import models


class StudentRegistrationForm(ModelForm):
    class Meta:
        model = models.Student
        fields = "__all__"
        exclude = ["user"]
        widgets = {
            "program": TextInput(
                attrs={
                    "class": "form-control form-control-lg",
                    "placeholder": "Program",
                }
            ),
            "department": TextInput(
                attrs={
                    "class": "form-control form-control-lg",
                    "placeholder": "Department",
                }
            ),
            "level": Select(
                attrs={
                    "class": "form-control form-control-lg",
                    "placeholder": "Select Level",
                }
            ),
        }


class TeacherRegistrationForm(ModelForm):
    class Meta:
        model = models.Teacher
        fields = "__all__"
        exclude = ["user"]
        widgets = {
            "department": TextInput(
                attrs={
                    "class": "form-control form-control-lg",
                    "placeholder": "Department",
                }
            ),
        }


class CourseCreationForm(ModelForm):
    class Meta:
        model = models.Course
        fields = "__all__"
        exclude = ["teacher", "students"]
        widgets = {
            "name": TextInput(
                attrs={
                    "class": "form-control form-control-lg",
                    "placeholder": "Name",
                }
            ),
            "description": TextInput(
                attrs={
                    "class": "form-control form-control-lg",
                    "placeholder": "Description",
                }
            ),
        }


class AssignmentCreationForm(ModelForm):
    class Meta:
        model = models.Assignment
        fields = "__all__"
        exclude = ["teacher", "course"]
        widgets = {
            "name": TextInput(
                attrs={
                    "class": "form-control form-control-lg",
                    "placeholder": "Name",
                }
            ),
            "description": TextInput(
                attrs={
                    "class": "form-control form-control-lg",
                    "placeholder": "Description",
                }
            ),
            "due_date": DateTimeInput(
                attrs={
                    "class": "form-control form-control-lg",
                    "placeholder": "Due Date",
                }
            ),
        }


class SubmissionForm(ModelForm):
    class Meta:
        model = models.Submission
        fields = "__all__"
        exclude = ["assignment", "student"]


class EnrollmentForm(Form):
    course = ModelChoiceField(
        queryset=models.Course.objects.all(), label="Select Course"
    )


class RegisterUserForm(UserCreationForm):
    class Meta:
        model = models.User
        fields = [
            "first_name",
            "last_name",
            "email",
            "password1",
            "password2",
            "user_type",
        ]
        widgets = {
            "first_name": TextInput(
                attrs={
                    "class": "form-control form-control-lg",
                    "placeholder": "First Name",
                }
            ),
            "last_name": TextInput(
                attrs={
                    "class": "form-control form-control-lg",
                    "placeholder": "Last Name",
                }
            ),
            "email": EmailInput(
                attrs={"class": "form-control form-control-lg", "placeholder": "Email"}
            ),
            "password1": PasswordInput(
                attrs={
                    "class": "form-control form-control-lg",
                    "placeholder": "Password",
                }
            ),
            "password2": PasswordInput(
                attrs={
                    "class": "form-control form-control-lg",
                    "placeholder": "Confirm Password",
                }
            ),
            "user_type": Select(attrs={"class": "form-control form-control-lg"}),
        }
