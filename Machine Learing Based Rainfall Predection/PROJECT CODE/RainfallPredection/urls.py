"""RainfallPredection URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from RainfallPredection import views as mainView
from users import views as usr
from admins import views as admns

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', mainView.index, name='index'),
    path('Logout/',mainView.Logout,name='Logout'),

    ### User Side all urs
    path('UserLogin/',usr.UserLogin,name='UserLogin'),
    path('UserRegister/',usr.UserRegister,name='UserRegister'),
    path('GetWeatherInfo/',usr.GetWeatherInfo,name='GetWeatherInfo'),
    path('UserRegisterAction/',usr.UserRegisterAction,name='UserRegisterAction'),
    path('UserLoginCheck/',usr.UserLoginCheck, name='UserLoginCheck'),
    path('GetWeatherInfo/',usr.GetWeatherInfo, name='GetWeatherInfo'),
    path('SearchByCity/',usr.SearchByCity, name='SearchByCity'),
    path('UserUploadData/',usr.UserUploadData,name='UserUploadData'),
    path('UploadCSVToDataBase/', usr.UploadCSVToDataBase, name='UploadCSVToDataBase'),
    path('UserDataPreProcess/',usr.UserDataPreProcess, name='UserDataPreProcess'),
    path('UserMLRCode/',usr.UserMLRCode,name='UserMLRCode'),
    path('TestMlR/',usr.TestMlR, name='TestMlR'),


    ### Admins side urls
    path('AdminLogin/',admns.AdminLogin,name='AdminLogin'),
    path('AdminLoginCheck/',admns.AdminLoginCheck,name='AdminLoginCheck'),
    path('AdminViewUsers/', admns.AdminViewUsers, name='AdminViewUsers'),
    path('AdminActivaUsers/',admns.AdminActivaUsers,name='AdminActivaUsers'),
    path('AdminAddData/',admns.AdminAddData,name='AdminAddData'),
    path('AdminViewData/',admns.AdminViewData,name='AdminViewData'),

]
