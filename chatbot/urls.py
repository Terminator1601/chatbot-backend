from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from .views import chatbot_view, feedback_view, get_feedback_logs, delete_log, add_new_data, reject_log, add_media,add_new_media, get_table_data,delete_data

urlpatterns = [
    path('', views.chatbot_view, name='chatbot'),
    path('feedback/', feedback_view, name='feedback_view'),
    path('feedback-logs/', get_feedback_logs, name='feedback_logs'),  # Correct URL path
    path('delete-log/', delete_log, name='delete_log'),
    path('reject-log/', reject_log, name='reject_log'),
    path('add-media/', add_media, name='add_media'),
    path('add-new-media/', add_new_media, name='add_new_media'),
    path('add-new-data/', add_new_data, name='add_new_data'),
    path('delete-data/', delete_data, name='delete_data'),
    path('get-table-data/', get_table_data, name='get_table_data'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
