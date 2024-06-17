from django.contrib import admin
from . import models

# Register your models here.
admin.site.register(models.User)
admin.site.register(models.Student)
admin.site.register(models.Course)
admin.site.register(models.Teacher)
admin.site.register(models.Submission)
admin.site.register(models.Similarity)
admin.site.register(models.Plagarism)
admin.site.register(models.Assignment)
