from django.db import models

# Create your models here.

class RainFallUserRegistrationModel(models.Model):
    name = models.CharField(max_length=100)
    loginid = models.CharField(unique=True,max_length=100)
    password = models.CharField(max_length=100)
    mobile = models.CharField(max_length=100)
    email = models.CharField(max_length=100)
    locality = models.CharField(max_length=100)
    address = models.CharField(max_length=1000)
    city = models.CharField(max_length=100)
    state = models.CharField(max_length=100)
    status  = models.CharField(max_length=100)

    def __str__(self):
        return self.loginid
    class Meta:
        db_table='RainFallUsers'


class IndiaRainFallDataModel(models.Model):
    SUBDIVISION = models.CharField(max_length=100)
    YEAR = models.IntegerField()
    JAN = models.FloatField(default=0.0);
    FEB = models.FloatField(default=0.0);
    MAR = models.FloatField(default=0.0);
    APR = models.FloatField(default=0.0);
    MAY = models.FloatField(default=0.0);
    JUN = models.FloatField(default=0.0);
    JUL = models.FloatField(default=0.0);
    AUG = models.FloatField(default=0.0);
    SEP = models.FloatField(default=0.0);
    OCT = models.FloatField(default=0.0);
    NOV = models.FloatField(default=0.0);
    DEC = models.FloatField(default=0.0);
    ANNUAL = models.FloatField(default=0.0);
    JanToFeb = models.FloatField(default=0.0);
    MarToMay = models.FloatField(default=0.0);
    JunToSep = models.FloatField(default=0.0);
    OctToDec  = models.FloatField(default=0.0);
    def __str__(self):
        return self.SUBDIVISION

    class Meta:
        db_table = 'IndianRainfall'
