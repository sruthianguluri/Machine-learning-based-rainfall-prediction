import math

from django.shortcuts import render,HttpResponse
from sklearn.model_selection import train_test_split

from .forms import UserRegistrationForm
import requests
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import RainFallUserRegistrationModel,IndiaRainFallDataModel
import io,csv
from django_pandas.io import read_frame
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.preprocessing import StandardScaler
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import metrics
import os
from .GenGraphCode import GeneratePltGraph
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Create your views here.

def UserLogin(request):
    return render(request,'UserLogin.html',{})

def UserRegister(request):
    form = UserRegistrationForm()
    return render(request,'UserRegisterForm.html',{'form':form})


def GetWeatherInfo(request):
    if request.method == 'POST':
        city = request.POST.get('cityname')
        accesKey = '9c5781db8bb4ee7c96f7dee77728e353'
        url = 'http://api.openweathermap.org/data/2.5/weather?q={}&units=imperial&appid=9c5781db8bb4ee7c96f7dee77728e353'
        #city = 'Las Vegas'
        city_weather = requests.get( url.format(city)).json()  # request the API data and convert the JSON to Python data types
        print("Weather info ",city_weather)
        weather = {
            'city': city,
            'temperature': city_weather['main']['temp'],
            'description': city_weather['weather'][0]['description'],
            'icon': city_weather['weather'][0]['icon'],
            #'message':city_weather['message']
        }

    context = {'weather': weather}
    return render(request, 'weatherinfo.html', context)

def UserRegisterAction(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            # return HttpResponseRedirect('./CustLogin')
            form = UserRegistrationForm()
            return render(request, 'UserRegisterForm.html', {'form': form})
        else:
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegisterForm.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = RainFallUserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
            # return render(request, 'user/userpage.html',{})
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})

def GetWeatherInfo(request):
    if request.method == 'POST':
        city = request.POST.get('cityname')
        accesKey = '9c5781db8bb4ee7c96f7dee77728e353'
        url = 'http://api.openweathermap.org/data/2.5/weather?q={}&units=imperial&appid=9c5781db8bb4ee7c96f7dee77728e353'

        city_weather = requests.get( url.format(city)).json()  # request the API data and convert the JSON to Python data types
        print("Weather info ",city_weather)
        code = city_weather['cod']
        if code==200:
            print('Data valid')
        else:
            messages.success(request, 'Data Not Found')
            return render(request, 'users/GetWeatherInfo.html', {})
        print('Status code ',code)
        weather = {
            'city': city,
            'temperature': city_weather['main']['temp'],
            'description': city_weather['weather'][0]['description'],
            'icon': city_weather['weather'][0]['icon'],
            #'message':city_weather['message']
        }

    context = {'weather': weather}
    return render(request, 'users/GetWeatherInfo.html', context)

def SearchByCity(request):
    return render(request,'users/SearchByCity.html',{})

def UserUploadData(request):
    return render(request,'users/uploaddata.html',{})


def UploadCSVToDataBase(request):
    # declaring template
    template = "users/UserHomePage.html"
    data = IndiaRainFallDataModel.objects.all()
    # prompt is a context variable that can have different values      depending on their context
    prompt = {
        'order': 'Order of the CSV should be name, email, address,    phone, profile',
        'profiles': data
    }
    # GET request returns the value of the data with the specified key.
    if request.method == "GET":
        return render(request, template, prompt)
    csv_file = request.FILES['file']
    # let's check if it is a csv file
    if not csv_file.name.endswith('.csv'):
        messages.error(request, 'THIS IS NOT A CSV FILE')
    data_set = csv_file.read().decode('UTF-8')

    # setup a stream which is when we loop through each line we are able to handle a data in a stream
    io_string = io.StringIO(data_set)
    next(io_string)
    for column in csv.reader(io_string, delimiter=',', quotechar="|"):
        _, created = IndiaRainFallDataModel.objects.update_or_create(
            SUBDIVISION=column[0],
            YEAR=column[1],
            JAN=column[2],
            FEB=column[3],
            MAR=column[4],
            APR=column[5],
            MAY=column[6],
            JUN=column[7],
            JUL=column[8],
            AUG=column[9],
            SEP=column[10],
            OCT=column[11],
            NOV=column[12],
            DEC=column[13],
            ANNUAL=column[14],
            JanToFeb=column[15],
            MarToMay=column[16],
            JunToSep=column[17],
            OctToDec=column[18]

        )
    context = {}

    return render(request, 'users/UserHomePage.html', context)

def UserDataPreProcess(request):
    qs = IndiaRainFallDataModel.objects.all()
    data = read_frame(qs)
    g = GeneratePltGraph()
    g.preProcessGraphs(data)
    return render(request,'users/PreProcessedData.html',{'data':qs})

def UserMLRCode(request):
    qs = IndiaRainFallDataModel.objects.all()
    data = read_frame(qs)
    gf = GeneratePltGraph()
    gf.genMlrCodes(data)
    return render(request,'users/UsersMachineLearningGraphs.html',{})




def TestMlR(request):

    qs = IndiaRainFallDataModel.objects.all()
    data = read_frame(qs)
    gh = GeneratePltGraph()
    rsltdict = gh.testMltMSE(data)
    return render(request,"users/MlrTestResult.html",rsltdict)


