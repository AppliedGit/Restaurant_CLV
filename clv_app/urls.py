from operator import index
from django.urls import path
from . import views

urlpatterns = [
    path('index/',views.index,name="index"),
    path('get_time_series_data/',views.get_time_series_data,name="get_time_series_data"),    
    path('get_location/',views.get_location,name="get_location"),
    path('location_based_segment_data/',views.location_based_segment_data,name="location_based_segment_data"),
    path('age_based_location_data/',views.age_based_location_data,name="age_based_location_data"),
    path('revenue_based_location_data/',views.revenue_based_location_data,name="revenue_based_location_data"),
    path('location_based_excutive_summary/',views.location_based_excutive_summary,name="location_based_excutive_summary"),
    #path('',views.login,name="login"),   
    path('',views.home,name="home"),   
    ]