from django.urls import path
from . import views

urlpatterns = [
    # INFO: Student Paths
    path("student/register", views.student_register, name="student/register"),
    path("student/enrollment", views.student_enrollment, name="student/enrollment"),
    path("student/course/", views.student_course, name="student/course"),
    path(
        "student/course/<str:pk>/assignment",
        views.student_assignment,
        name="student/assignment",
    ),
    path(
        "student/course/<str:course_pk>/assignment/<str:ass_pk>/submission",
        views.student_submissions,
        name="student/submission",
    ),
    path(
        "student/submission/<str:pk>/report",
        views.student_report,
        name="student/report",
    ),
    # INFO: Teacher Paths
    path("teacher/register", views.teacher_register, name="teacher/register"),
    path("teacher/course", views.teacher_course, name="teacher/course"),
    path(
        "teacher/create_course",
        views.teacher_create_course,
        name="teacher/create_course",
    ),
    path(
        "teacher/course/<str:pk>/assignment",
        views.teacher_assignment,
        name="teacher/assignment",
    ),
    path(
        "teacher/course/<str:course_pk>/assignment/create",
        views.teacher_create_assignment,
        name="teacher/create-assignment",
    ),
    path(
        "teacher/course/<str:pk>/assignment/<str:ass_pk>/submission",
        views.teacher_submissions,
        name="teacher/submissions",
    ),
    path("teacher/submission/<str:pk>/report", views.teacher_report, name="teacher/report"),
    # INFO: Auth Paths
    path("login/", views.login_register, name="login"),
    path("signup/", views.signup, name="signup"),
    path("logout/", views.logout_route, name="logout"),
]
