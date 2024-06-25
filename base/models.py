from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from django.utils.timezone import localtime, is_aware
from datetime import timedelta

# Create your models here.


class UserType(models.TextChoices):
    STUDENT = "ST", _("Student")
    TEACHER = "TE", _("Teacher")

    @staticmethod
    def get_link(data: str):
        match data:
            case "TE":
                return "teacher/course"
            case _:
                # By default return the student home
                return "student/course"

    @staticmethod
    def get_register_link(data: str):
        match data:
            case "TE":
                return "teacher/register"
            case _:
                # By default return the student home
                return "student/register"


class StudentLevel(models.IntegerChoices):
    LEVEL_100 = 100, _("100 Level")
    LEVEL_200 = 200, _("200 Level")
    LEVEL_300 = 300, _("300 Level")
    LEVEL_400 = 400, _("400 Level")
    LEVEL_500 = 500, _("500 Level")


class User(AbstractUser):
    # username = models.CharField(max_length=200)
    username = models.CharField(max_length=200)
    email = models.EmailField(unique=True)
    user_type = models.CharField(
        max_length=2, choices=UserType, default=UserType.STUDENT
    )

    def get_student(self) -> type["Student"] | None:
        return Student.objects.get(user=self)

    def get_teacher(self) -> type["Teacher"] | None:
        return Teacher.objects.get(user=self)

    REQUIRED_FIELDS = ["user_type"]
    USERNAME_FIELD = "email"


class Student(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, unique=True)
    level = models.IntegerField(
        choices=StudentLevel,
    )
    program = models.CharField(max_length=200)
    department = models.CharField(max_length=200)

    def get_name(self):
        return f"{str(self.user.last_name)} {str(self.user.first_name)}"

    def __str__(self):
        return str(self.user)


class Teacher(models.Model):
    department = models.CharField(max_length=200)
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return str(self.user)


class Course(models.Model):
    teacher = models.ForeignKey(Teacher, on_delete=models.SET_NULL, null=True)
    name = models.CharField(max_length=200)
    description = models.TextField()
    students = models.ManyToManyField(Student, "enrollment_key")

    def __str__(self):
        return str(self.name)


# class Enrollment(models.Model):
#     course = models.ForeignKey(Course, on_delete=models.CASCADE)
#     student = models.ForeignKey(Student, on_delete=models.CASCADE)
#
#     def __str__(self):
#         return f"{str(self.student)} -> {str(self.course)}"


class Assignment(models.Model):
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    teacher = models.ForeignKey(Teacher, on_delete=models.CASCADE)
    due_date = models.DateTimeField()
    name = models.CharField(max_length=200)
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return str(self.name)

    def is_due(self) -> bool:
        time = timezone.now()
        due_date = self.due_date
        time = time + timedelta(hours=1) 
        val = time > due_date

        print(val, time, due_date)
        return val


class Submission(models.Model):
    # INFO: When we delete an assignment from a course, don't delete the submission
    assignment = models.ForeignKey(Assignment, on_delete=models.SET_NULL, null=True)
    # INFO: When we delete a student, delete the submission
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    file_name = models.CharField(max_length=200)
    file = models.FileField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{str(self.student)} -> {str(self.file_name)}"


class Plagarism(models.Model):
    submission = models.ForeignKey(Submission, on_delete=models.CASCADE)
    score = models.FloatField()
    sources = models.TextField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.score)


class Similarity(models.Model):
    submission = models.ForeignKey(Submission, on_delete=models.CASCADE)
    score = models.FloatField()
    sources = models.TextField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.score)

class Grading(models.Model):
    submission = models.ForeignKey(Submission, on_delete=models.CASCADE)
    predicted_score = models.FloatField()
    spelling_errors = models.FloatField()
    grammar_errors = models.FloatField()
    readability_score = models.FloatField()
    readability_level = models.TextField(max_length=256)
    topics_covered=models.TextField()
    description_match = models.TextField()
    total_score = models.FloatField()
    sources = models.TextField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)
