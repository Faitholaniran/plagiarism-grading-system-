from django.contrib.auth import authenticate, login, logout
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.contrib import messages

from base.forms import (
    AssignmentCreationForm,
    CourseCreationForm,
    EnrollmentForm,
    RegisterUserForm,
    StudentRegistrationForm,
    SubmissionForm,
    TeacherRegistrationForm,
)

from . import models


# INFO: Student Routes
def student_course(request: HttpRequest):
    """
    Get the all the courses for a student and put it in a list
    """
    user: models.Student = request.user.get_student()
    if not user:
        return HttpResponse(b"You are not logged in")

    courses = models.Course.objects.filter(students__id=user.id)

    ctx = {"courses": courses}
    return render(request, "base/students/course.html", ctx)


def student_register(request: HttpRequest):
    if not request.user.is_authenticated:
        return HttpResponse("You need to be logged in to get here")

    try:
        user = request.user.get_student()
        return HttpResponse("You already are a student")
    except:
        if request.method == "POST":
            form = StudentRegistrationForm(request.POST)
            if form.is_valid():
                student = form.save(commit=False)
                student.user = request.user
                student.save()
                form.save_m2m()
                return redirect("student/course")
            else:
                messages.error(request, "An error occured trying to register the user")
        form = StudentRegistrationForm()
        ctx = {"form": form}
        return render(request, "base/students/register.html", ctx)


def student_enrollment(request: HttpRequest):
    student = request.user.get_student()
    if not student:
        return HttpResponse("Not a valid student")

    if request.method == "POST":
        enrollment_form = EnrollmentForm(request.POST)
        if enrollment_form.is_valid():
            course = models.Course.objects.get(id=request.POST.get("course"))

            course.students.add(student)
            return redirect("student/course")

    enrollment_form = EnrollmentForm()
    all_courses = models.Course.objects.all()
    courses_enrolled_in = models.Course.objects.filter(students=student)
    valid_courses = all_courses.difference(courses_enrolled_in)

    enrollment_form.fields["course"].queryset = valid_courses

    return render(
        request,
        "base/students/enrollment_form.html",
        {"valid_courses": valid_courses, "form": enrollment_form},
    )


def student_assignment(request: HttpRequest, pk: str):
    """
    Get all the assignment for course using the course_id(`pk`)
    """

    course = models.Course.objects.get(id=int(pk))
    assignments = models.Assignment.objects.filter(course__id=int(pk))
    submission = models.Submission.objects.filter(student__user=request.user)
    submitted_assignments = [submission.assignment for submission in submission]

    return render(
        request,
        "base/students/assignment.html",
        {
            "assignments": assignments,
            "course": course,
            "submitted_assignments": submitted_assignments,
        },
    )


def student_submissions(request: HttpRequest, course_pk: int, ass_pk: int):
    """
    Upload the submission
    """
    student = models.Student.objects.get(user=request.user)
    if not student:
        return HttpResponse(b"An error occured, the user was not defined")

    course = models.Course.objects.get(id=int(course_pk))
    assignment = models.Assignment.objects.get(id=int(ass_pk))

    try:
        prev_submission = models.Submission.objects.get(
            student=student, assignment=assignment
        )
        if prev_submission:
            return HttpResponse(b"You have already submitted before")
    except:
        if request.method == "POST":
            submission_form = SubmissionForm(request.POST, request.FILES)
            if submission_form.is_valid():
                submission = submission_form.save(commit=False)
                submission.assignment = assignment
                submission.student = student

                submission.save()
                submission_form.save_m2m()
                return redirect("student/course")
            else:
                return HttpResponse(b"There was an error in submitting the form")

        submission_form = SubmissionForm()
        ctx = {"course": course, "assignment": assignment, "form": submission_form}
        return render(request, "base/students/submission.html", ctx)


def student_report(request: HttpRequest, pk: int):
    """
    Get the submission report for a student assignment submission
    """
    return HttpResponse(b"Not yet implemented")


# INFO: Teacher Routes
def teacher_register(request: HttpRequest):
    if not request.user.is_authenticated:
        return HttpResponse("You need to be logged in to get here")

    try:
        user = request.user.get_teacher()
        return HttpResponse("You already are a teacher")
    except:
        if request.method == "POST":
            form = TeacherRegistrationForm(request.POST)
            if form.is_valid():
                teacher = form.save(commit=False)
                teacher.user = request.user
                teacher.save()
                form.save_m2m()
                return redirect("teacher/course")
            else:
                messages.error(request, "An error occured trying to register the user")
        form = TeacherRegistrationForm()
        ctx = {"form": form}
        return render(request, "base/teachers/register.html", ctx)


def teacher_course(request: HttpRequest):
    teacher = request.user.get_teacher()
    if teacher is None:
        return HttpResponse(b"User must be a teacher to use this route")
    courses = models.Course.objects.filter(teacher__id=teacher.id)
    ctx = {
        "courses": courses,
    }
    return render(request, "base/teachers/course.html", ctx)


def teacher_create_course(request: HttpRequest):
    teacher = request.user.get_teacher()
    if not teacher:
        return HttpResponse("Not a valid teacher")
    if request.method == "POST":
        course_form = CourseCreationForm(request.POST)
        if course_form.is_valid():
            course = course_form.save(commit=False)
            course.teacher = teacher
            course.save()
            course_form.save_m2m()
            return redirect("teacher/course")
        else:
            messages.error(request, "There was an error in creating a new course")

    course_form = CourseCreationForm()
    ctx = {"form": course_form}
    return render(request, "base/teachers/course_creation.html", ctx)


def teacher_create_assignment(request: HttpRequest, course_pk: str):
    teacher = request.user.get_teacher()
    if not teacher:
        return HttpResponse("Not a valid teacher")
    course = models.Course.objects.get(id=int(course_pk))
    if request.method == "POST":
        assignment_form = AssignmentCreationForm(request.POST)
        if assignment_form.is_valid():
            assignment = assignment_form.save(commit=False)
            assignment.course = course
            assignment.teacher = teacher
            assignment.save()
            assignment_form.save_m2m()
            return redirect("teacher/course")
        else:
            messages.error(request, "There was an error in creating a new course")

    assignment_form = AssignmentCreationForm()
    ctx = {"form": assignment_form, "course": course}
    return render(request, "base/teachers/assignment_creation.html", ctx)
    pass


def teacher_assignment(request: HttpRequest, pk: str):
    course = models.Course.objects.get(id=int(pk))
    print(course.assignment_set.count())

    return render(
        request,
        "base/teachers/assignment.html",
        {"course": course, "assignments": course.assignment_set.all()},
    )
    return HttpResponse(b"Not yet implemented")


def teacher_submissions(request: HttpRequest, pk: str, ass_pk: str):
    course = models.Course.objects.get(id=int(pk))
    assignment = models.Assignment.objects.get(id=int(ass_pk))
    ctx = {
        "course": course,
        "assignment": assignment,
        "submissions": assignment.submission_set.all(),
    }
    return render(request, "base/teachers/submissions.html", ctx)

    return HttpResponse(b"Not yet implemented")


# INFO: Auth Routes
def login_register(request: HttpRequest):
    """
    This is the page that a user would be used to log in
    """

    if request.method == "POST":
        email = request.POST.get("email", "")
        password = request.POST.get("password", "")
        user = authenticate(request, email=email, password=password)
        if user is not None:
            login(request, user)
            return redirect(models.UserType.get_link(request.user.user_type))
        else:
            messages.error(request, "Username or Password does not exist")

    ctx = {
        "is_login": True,
    }
    return render(request, "base/register.html", ctx)


def signup(request: HttpRequest):
    """
    This is the page that a user would be used to register
    """
    if request.user.is_authenticated:
        return redirect(models.UserType.get_link(request.user.user_type))
    if request.method == "POST":
        form = RegisterUserForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect(models.UserType.get_register_link(user.user_type))
        else:
            print(form.errors)
            messages.error(request, "There was an error in validating the form")
    form = RegisterUserForm()
    ctx = {"form": form, "is_login": False}
    return render(request, "base/register.html", ctx)


def logout_route(request: HttpRequest):
    logout(request)
    return redirect("login")
