from django import forms
from users.models import IndiaRainFallDataModel

class IndiaRainFallDataForm(forms.ModelForm):
    SUBDIVISION = forms.CharField(max_length=100)
    YEAR = forms.IntegerField()
    JAN = forms.FloatField()
    FEB = forms.FloatField()
    MAR = forms.FloatField()
    APR = forms.FloatField()
    MAY = forms.FloatField()
    JUN = forms.FloatField()
    JUL = forms.FloatField()
    AUG = forms.FloatField()
    SEP = forms.FloatField()
    OCT = forms.FloatField()
    NOV = forms.FloatField()
    DEC = forms.FloatField()
    ANNUAL = forms.FloatField()
    JanToFeb = forms.FloatField()
    MarToMay = forms.FloatField()
    JunToSep = forms.FloatField()
    OctToDec = forms.FloatField()


    class Meta():
        model = IndiaRainFallDataModel
        fields = '__all__'