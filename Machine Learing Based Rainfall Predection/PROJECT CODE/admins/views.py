from django.shortcuts import render,HttpResponse
from django.contrib import messages
# Create your views here.
from users.models import RainFallUserRegistrationModel,IndiaRainFallDataModel
from .forms import IndiaRainFallDataForm
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

def AdminLogin(request):
    return render(request,'AdminLogin.html',{})

def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/AdminHome.html')

        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'AdminLogin.html', {})


def AdminViewUsers(request):
    data = RainFallUserRegistrationModel.objects.all()
    return render(request,'admins/AdminViewUsers.html',{'data':data})

def AdminActivaUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        RainFallUserRegistrationModel.objects.filter(id=id).update(status=status)
        data = RainFallUserRegistrationModel.objects.all()
        return render(request,'admins/AdminViewUsers.html',{'data':data})

def AdminAddData(request):
    if request.method == 'POST':
        form = IndiaRainFallDataForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'Data Added Successfull')
            form = IndiaRainFallDataForm()
            return render(request, 'admins/AddRainFallData.html', {'form': form})
        else:
            print("Invalid form")
    else:
        form = IndiaRainFallDataForm()
    return render(request, 'admins/AddRainFallData.html', {'form': form})

def AdminViewData(request):
    data_list = IndiaRainFallDataModel.objects.all()
    page = request.GET.get('page', 1)

    paginator = Paginator(data_list, 60)
    try:
        users = paginator.page(page)
    except PageNotAnInteger:
        users = paginator.page(1)
    except EmptyPage:
        users = paginator.page(paginator.num_pages)

    return render(request, 'admins/AdminViewWeather.html', {'users': users})








