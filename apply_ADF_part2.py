#apply_ADF_part2.py
#part 2 uses as input files the output csv files created in part 1

#===============================================================================
#strdate='20170119'
#strdate='20170124'
#strdate='20170125_CHECK'
#strdate='20170125_ALL'
#strdate='20170126_FRAC1'
#strdate='20170127_FRAC_WOLF'
#strdate='20170127_FRAC_OBS'
#strdate='20170131_FRAC_WOLFER_SPEC'
#strdate='20170131_FRAC_QUIMBY_SPEC'
#strdate='20170206_FRAC_QUIMBY_SPEC'
#strdate='20170209_FRAC_WOLFER_1902_1923'
#strdate='20170531_instance0_Daily'
#strdate='20170531_instance0_Daily_broger'
#strdate='20170531_instance0_Daily_winkler'
#strdate='20170531_instance0_Daily_wolfer_2'
#strdate='20170531_instance0_Daily_quimby'
#strdate='20170601_instance2_ADF2_Daily'
#strdate='20170603_instance2_ADF2_Daily'
#strdate='20170604_ADF2_quimby'
#strdate='20170607_ADF2_shea_weber_special'

#strdate='20170607_ADF2'
#strdate='20170608_ADF2'
#strdate='20170612_ADF2_special'
#strdate='20170613_ADF2'
strdate='20170911_ADF2'

import re
import random
import itertools
import csv
import datetime as dt
import calendar
import sys,os,numpy, scipy, glob
import math as mt
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import pylab as py
from scipy.stats import sigmaclip
from scipy.stats import kstest
from scipy.stats import anderson
from scipy.stats import mode
from scipy import stats
import pandas as pd
import scipy.optimize as optimization
from scipy.optimize import curve_fit
from datetime import date, timedelta as td
import matplotlib as mpl

dirIn='/Users/Laure/Desktop/SSNWork/corr/'
dirBase='/Users/Laure/sunspots/SSN/SSNCALC/newSN/'
dircreate=dirBase+'results_ADF/'
if not os.path.exists(dircreate):
    os.makedirs(dircreate)

dirOut=dircreate
#fileObscsv=open(dirOut+'Observers_'+obs1+'_'+obs2+'.csv','w')
cmap = mpl.cm.spectral

#===============================================================================
#===========================FUNCTIONS===========================================
#===============================================================================
#===============================================================================
#===============================================================================
#===============================================================================
def num_days_in_month(year, month):
	NUM_DAYS_IN_YR = 365. + leapyr(year)
	if leapyr(year) > 0.:
		days = [31,29,31,30,31,30,31,31,30,31,30,31]
	else:
		days = [31,28,31,30,31,30,31,31,30,31,30,31]
	ndays=days[int(month-1)]
	return ndays
#===============================================================================
#===============================================================================
def find_ndaysobs(date1,gn1,d1,d2):
	ndaysobs=0.
	for i in range(0, len(date1)):
		if gn1[i] >=0. and date1[i] >= d1 and date1[i] <= d2:
			ndaysobs+=1.
	return ndaysobs
#===============================================================================
#===============================================================================
def return_subset(date1,gn1,d1,d2):
	date1s=[];gn1s=[]
	for i in range(0, len(date1)):
		if date1[i] >= d1 and date1[i] <= d2:
			date1s.append(date1[i])
			gn1s.append(gn1[i])
	return date1s,gn1s
#===============================================================================
#===============================================================================
def find_nmonthsobs(date1,gn1,d1,d2):
	monthold=0. ; nmonthsobs=0. ; yearold=0.
	for i in range(0, len(date1)):
		if date1[i] >= d1 and date1[i] <= d2:
			currentmonth=date1[i].month
			currentyear=date1[i].year
			if gn1[i] >=0. and monthold+yearold != currentmonth+currentyear:
				nmonthsobs+=1.
				monthold=currentmonth
				yearold=currentyear
	return nmonthsobs
#===============================================================================
#===============================================================================
def find_ndaysinmonthsobs(date1,gn1,d1,d2,limit):
	monthold=0. ; yearold=0.
	yeardd=[]; monthdd=[];ndmonthd=[];fracdmonthd=[]
	nmonthsobs=0.
	for i in range(0, len(date1)):
		if date1[i] >= d1 and date1[i] <= d2:
			currentmonth=date1[i].month
			currentyear=date1[i].year
			#print monthold, yearold, currentmonth,currentyear,nmonthsobs

			if monthold+yearold != currentmonth+currentyear and monthold != 0.:
				#if nmonthsobs == 0.:
				#	import pdb; pdb.set_trace()
				monthdd.append(monthold)
				yeardd.append(yearold)
				ndmonthd.append(nmonthsobs)
				ndaysmonth=num_days_in_month(yearold, monthold)
				fracdmonthd.append(nmonthsobs/ndaysmonth)
				#print 'NOBS',nmonthsobs
				nmonthsobs=0. 
				#print 'RAZ'
			if gn1[i] >=limit:
				#print 'TRUE', currentyear, currentmonth, gn1[i], limit 
				nmonthsobs+=1.
			if gn1[i] >= -99.: 
				monthold=currentmonth
				yearold=currentyear
	monthdd.append(monthold)
	yeardd.append(yearold)
	ndmonthd.append(nmonthsobs)
	ndaysmonth=num_days_in_month(yearold, monthold)
	fracdmonthd.append(nmonthsobs/ndaysmonth)

	return monthdd, yeardd, ndmonthd, fracdmonthd
#===============================================================================
#===============================================================================
def date_conv(year, month, day, hour, minutes, seconds):
	NUM_DAYS_IN_YR = 365. + leapyr(year)
	if leapyr(year) > 0.:
		days = [31,29,31,30,31,30,31,31,30,31,30,31]
	else:
		days = [31,28,31,30,31,30,31,31,30,31,30,31]
	nbdays=0.
	if month > 1:
		for i in range(1,int(month)):
			nbdays+=days[i-1]
			#print nbdays
		nbdays+=day-1.
		fracyear=year+nbdays/NUM_DAYS_IN_YR+hour/(NUM_DAYS_IN_YR*24)+minutes/(NUM_DAYS_IN_YR*24*60.)+seconds/(NUM_DAYS_IN_YR*24*60.*60.)
	else: #january
		nbdays=day-1.
		fracyear=year+nbdays/NUM_DAYS_IN_YR+hour/(NUM_DAYS_IN_YR*24)+minutes/(NUM_DAYS_IN_YR*24*60.)+seconds/(NUM_DAYS_IN_YR*24*60.*60.)
#	import pdb; pdb.set_trace()

	return fracyear
#===============================================================================
#===============================================================================
def leapyr(n):
    if n % 400 == 0:
        return True
    if n % 100 == 0:
        return False
    if n % 4 == 0:
        return 1.
    else:
        return 0.
#print leapyr(1900)
#===============================================================================
#===============================================================================
def select_fraction_days(datesR,dateR,areaR,fraction) :
	date2=[] ; gn2=[] ; dates2=[]
	date3=np.array(datesR) ; area3=np.array(areaR)
	nobs=len(datesR)
	#remove duplicates in datesR
	dates=sorted(list(set(datesR)))
	ndays=len(dates)
	d1=dateR[0] ; d2=dateR[-1]
	#determine actual number of days
	delta=d2-d1
	print 'DELTA', d1, d2
	ndaystot=delta.days
	nmonthtot=(d2.year - d1.year)*12 + (d2.month - d1.month)+1
	#import pdb; pdb.set_trace()

	num_to_select = int(round(fraction*ndays))
	list_of_select = random.sample(np.arange(len(dates)), num_to_select)
	tata=pd.Series(dates)
	selection=sorted(list(set(list_of_select)))	
	dates_selected=list(tata[selection])
	
	
	selected_groups=[]
	for i in range(0,len(dates_selected)) :
		curdate=dates_selected[i]
		titi=whereis(datesR,curdate,0.00001)
		if pd.Series(titi)[0] != -1.:
			#print curdate,datesR[titi[0]], len(titi)
			for j in range(0,len(titi)):
				selected_groups.append(titi[j])
	
	return selected_groups
#===============================================================================

#===============================================================================
def YearDec(thedate):
    nyear=thedate.year ; nmonth=thedate.month ; nday=thedate.day

    ndInYear=(dt.date(nyear+1,1,1)-dt.date(nyear,1,1)).days
    return float(nyear)+ ((dt.date(nyear,nmonth,1) - dt.date(nyear,1,1)).days + nday -0.5)/float(ndInYear)
#===============================================================================

def whereis(tableau, valeur, contrainte):
	table = np.asarray(tableau)
	#choice=(table == valeur)
	choice=((table >= valeur-contrainte) & (table < valeur+contrainte))
	r = np.array(range(len(choice)))
	index=np.asarray(r[choice])
	if len(index) :
		ind=index
	else:
		ind=-1
	return ind
#===============================================================================
#===============================================================================

def whereis2(tableau,tableau1, valeur, contrainte, tt):
	#print 'THRESHOLD whereis2', tt
	table = np.asarray(tableau)
	table1 = np.asarray(tableau1)
	choice=((table >= valeur-contrainte) & (table < valeur+contrainte) & (table1 >= tt))
	r = np.array(range(len(choice)))
	index=np.asarray(r[choice])
	if len(index) :
		ind=index
	else:
		ind=-1
	return ind
#===============================================================================


#===============================================================================

def apply_thres(datesR,dateR,areaR,TH) :
	#print 'THRESHOLD', TH
	Thres=str(TH).strip()

	date2=[] ; gn2=[] ; dates2=[]
	date3=np.array(datesR) ; area3=np.array(areaR)
	nobs=len(datesR)
	dates=sorted(list(set(datesR)))
	ndays=len(dates)
	lastdate=datesR[-1]
	
	#import pdb; pdb.set_trace()
	for i in range(0,ndays) :
		curdate=dates[i]
		if curdate <= lastdate:
			titi1=whereis(datesR,curdate,0.001)
			#print 'THRESHOLD', TH
			titi=whereis2(datesR,areaR,curdate,0.001,TH)
			#print titi
			#import pdb; pdb.set_trace()
			if pd.Series(titi)[0] != -1.:
				#print date3[titi], area3[titi], curdate, titi, len(pd.Series(titi))
				if float(i)%500. == 0.:
					print curdate,dateR[titi[0]], float(len(titi)), 'PROGRESSION :', (float(i)/float(ndays))*100., '%'
				comp=date_conv(dateR[titi[0]].year, dateR[titi[0]].month, dateR[titi[0]].day, 0.,0.,0.)
				if abs(curdate-comp) > 0.001 :
					import pdb; pdb.set_trace()
				dates2.append(curdate)
				date2.append(dateR[titi[0]])
				gn2.append(float(len(titi)))
			else:
				if float(i)%500. == 0.:
					print curdate,dateR[titi1[0]], 0., 'PROGRESSION :', (float(i)/float(ndays))*100., '%'
				comp=date_conv(dateR[titi1[0]].year, dateR[titi1[0]].month, dateR[titi1[0]].day, 0.,0.,0.)
				if abs(curdate-comp) > 0.001 and len(titi1) > 1:
					titi1=whereis(datesR,curdate,0.0002)
					
				if pd.Series(titi1)[0] == -1.:
					import pdb; pdb.set_trace()
	
				dates2.append(curdate)
				date2.append(dateR[titi1[0]])
				gn2.append(0.)
	
	
	
	#import pdb; pdb.set_trace()
	
	if len(set(dates2)) != len(set(date2)):
		import pdb; pdb.set_trace()
		
	
	return dates2,date2,gn2


#===============================================================================
#===============================================================================

def apply_thres2(datesR,dateR,areaR,TH) :
	print 'THRESHOLD', TH
	Thres=str(TH).strip()

	date2=[] ; gn2=[] ; dates2=[]
	date3=np.array(datesR) ; area3=np.array(areaR)
	nobs=len(datesR)
	dates=sorted(list(set(datesR)))
	ndays=len(dates)
	lastdate=datesR[-1]
	
	#import pdb; pdb.set_trace()
	for i in range(0,ndays) :
		curdate=dates[i]
		if curdate <= lastdate:
			
			alldates=np.asarray(datesR) ; area=np.asarray(areaR) 
			#condition = (alldates == curdate) & (area >= TH)
			#property_asel = list(itertools.compress(good_objects, property_a))
			#NG=len(np.where((alldates == curdate) & (area >= TH)))
			titi=np.where((alldates == curdate) & (area >= TH))
			titi1=np.where(alldates == curdate)
			NG=len(titi[0])
			#print 'TITI',titi, NG
			#import pdb; pdb.set_trace()
			if NG > 0.:
				#print date3[titi], area3[titi], curdate, titi, len(pd.Series(titi))
				if float(i)%500. == 0.:
					print curdate,dateR[titi[0][0]], float(NG), 'PROGRESSION :', (float(i)/float(ndays))*100., '%'
				#import pdb; pdb.set_trace()
				dates2.append(curdate)
				date2.append(dateR[titi[0][0]])
				gn2.append(float(NG))
			else:
				if float(i)%500. == 0.:
					print curdate,dateR[titi1[0][0]], 0., 'PROGRESSION :', (float(i)/float(ndays))*100., '%'
				dates2.append(curdate)
				date2.append(dateR[titi1[0][0]])
				gn2.append(0.)
	
	
	
	#import pdb; pdb.set_trace()
	
	if len(set(dates2)) != len(set(date2)):
		import pdb; pdb.set_trace()
		
	
	return dates2,date2,gn2


#===============================================================================

def add_values_holes(dates201,date201, gn201, value):
	print 'START ADD values holes'
	da1 = min(date201) ; da2=max(date201) 
	deltaa = da2 - da1

	date20=[] ; dates20=[] ; gn20=[]
	
	for ia in range(deltaa.days + 1):
		date=da1 + td(days=ia)
		curdate=date_conv(date.year, date.month,date.day, 0., 0., 0.)
		result = [abs(x-curdate)  for x in dates201]
		alldates=np.asarray(result)
		titi=np.where(alldates <= 0.001)
		NEL=len(titi[0])
		#print date_conv(date.year, date.month,date.day, 0., 0., 0.), date.year, date.month,date.day
		#import pdb; pdb.set_trace()
		if NEL > 0.:
			#print 'TRUE', dates201[titi[0]], gn201[titi[0]]
			if NEL >= 2.:
				for ib in range(0,NEL):
					gn20.append(gn201[titi[0][ib]])
					date20.append(da1 + td(days=ia))
					dates20.append(date_conv(date.year, date.month,date.day, 0., 0., 0.))
			else:
					gn20.append(gn201[titi[0][0]])
					date20.append(da1 + td(days=ia))
					dates20.append(date_conv(date.year, date.month,date.day, 0., 0., 0.))
				
		else:
			#print 'FALSE'
			gn20.append(value)
			date20.append(da1 + td(days=ia))
			dates20.append(date_conv(date.year, date.month,date.day, 0., 0., 0.))
	print 'END ADD values holes'

	return dates20,date20,gn20
#===============================================================================
def my_condition(x,a,b):
	return x >=a and x <= b
#===============================================================================
#===============================================================================
def my_thres(x,y,a,b):
	return x ==a and y >= b
#===============================================================================

#===============================================================================
#=============================DATA==============================================
#===============================================================================

#input2='/Users/Laure/sunspots/catalogs/RGO_1876_1981_corr.txt'
input2='/Users/Laure/sunspots/catalogs/RGO/RGOSOON_1874_2014.txt'
#firstdate=1909. ; lastdate=1924.
#firstdate=1914. ; lastdate=1924.
#firstdate=1900. ; lastdate=1976.
#wolfer
#firstdate=1900. ; lastdate=1928.
#quimby
#firstdate=1900. ; lastdate=1921.
firstdate=1900.
#wolfer
#lastdate=1928.
#quimby
#lastdate=1921.
#winkler
#lastdate=1910.
#broger
#lastdate=1935.

lastdate=1976.
#lastdate=1921.

#firstdate=1902; lastdate=1913.
#firstdate=1913; lastdate=1923.
#firstdate=1923; lastdate=1933.
#firstdate=1933; lastdate=1944.
#firstdate=1944; lastdate=1954.
#firstdate=1954; lastdate=1964.
#firstdate=1964; lastdate=1976.


fds=str(firstdate).strip()
lds=str(lastdate).strip()
#*********************************************************************
#***********************READ RGO**************************************
#*********************************************************************
#Declare all tables
yearRGO=[] ; monthRGO=[] ; dayRGO=[] ; areaRGO0=[]; dateRGO0=[] ; datesRGO0=[] 
fileTable=open(input2,'r') 
strlines=fileTable.readlines()
fileTable.close()
	
nblines=len(strlines)
for i in range(0,nblines) :
	readyear=float(strlines[i][0:4])
	if readyear >= firstdate and readyear <= lastdate:
		print '*',strlines[i][0:4],'*'
		print '*',strlines[i][4:6],'*'
		print '*',strlines[i][6:13],'*'
		print '*',strlines[i][30:34],'*'
		readmonth=float(strlines[i][4:6])
		readdays=strlines[i][6:13]
		readday=float(readdays.replace(" ", ""))
		hh=readday-mt.floor(readday)
		hour=mt.floor(hh*24.)
		mm=((hh*24.)-hour)*60.
		minutes=mt.floor(mm)
		seconds=(mm-minutes)*60.
		readarea=float(strlines[i][30:34])
		print readyear, readmonth, readday, hour, minutes, seconds, readarea
		yearRGO.append(readyear)
		monthRGO.append(readmonth)
		dayRGO.append(readday)
		datesRGO0.append(date_conv(readyear, readmonth, mt.floor(readday), 0., 0., 0.))
		dateRGO0.append(dt.date(int(readyear),int(readmonth),int(mt.floor(readday))))
		datecur=dt.date(int(readyear),int(readmonth),int(mt.floor(readday)))
		compare=date_conv(datecur.year, datecur.month, datecur.day, 0.,0.,0.)
		datecomp=date_conv(readyear, readmonth, mt.floor(readday), 0., 0., 0.)
		if abs(datecomp-compare) > 0.001 :
			import pdb; pdb.set_trace()
		areaRGO0.append(readarea)



datesRGO,dateRGO, areaRGO=add_values_holes(datesRGO0,dateRGO0, areaRGO0,0.)
#datesRGO=datesRGO0 ; dateRGO=dateRGO0 ; areaRGO=areaRGO0


#*********************************************************************
#***********************Vector of thresholds**************************
#*********************************************************************
#*********************************************************************
vectthres=numpy.arange(5.,100.,5.)
#vectthres=[5.,10.,20., 50.]
vectthres=[0.,10.,20., 50.]
sizethres=len(vectthres)



#Instance0
#observers=['quimby','wolfer','winkler','broger','tacchini','leppig','spoerer','weber','wolf','shea','schmidt','schwabe','pastorff']
#tcal1=[1900,1900,1889,1900,1879,1867,1865,1859,1860,1847,1841,1832,1824]
#tcal2=[1921,1928,1910,1935,1900,1880,1893,1883,1893,1866,1883,1866,1833]

#Instance0 - short
observers=['quimby','wolfer','winkler','broger']
tcal1=[1900,1900,1889,1900]
tcal2=[1921,1928,1910,1935]

#Instance0 - shea weber
observers=['weber','shea']
tcal1=[1859,1847]
tcal2=[1883,1866]


#QUIMBY
#observers=['quimby']
#tcal1=[1900]
#tcal2=[1921]



#WOLFER
#observers=['wolfer']
#tcal1=[1900]
#tcal2=[1928]

#WINKLER
#observers=['winkler']
#tcal1=[1900]
#tcal2=[1910]


#broger
#observers=['broger']
#tcal1=[1900]
#tcal2=[1935]

#spoerer
#observers=['spoerer']
#tcal1=[1865]
#tcal2=[1893]


#February 2017
#observers=['quimby','wolfer','winkler','tacchini','leppig','spoerer','weber','wolf','shea','schmidt','schwabe','pastorff']
#tcal1=[1900,1900,1889,1879,1867,1865,1859,1860,1847,1841,1832,1824]
#tcal2=[1921,1928,1910,1900,1880,1893,1883,1893,1866,1883,1867,1833]
#observers=['quimby','wolfer','wolf']
#tcal1=[1900,1900,1860]
#tcal2=[1921,1928,1893]
#observers=['RGO']
#tcal1=[1900]
#tcal2=[1976]

#WOLFER
#observers=['wolfer']

#tcal1=[1900]
#tcal2=[1928]

#tcal1=[1902]
#tcal2=[1923]



#observers=['quimby']
#tcal1=[1900]
#tcal2=[1921]
nbobs=len(observers)

#*********************************************************************
#***********************LOOP ON observers*****************************
#*********************************************************************
for observ in range(0,nbobs):
	obs=observers[observ]
	#obs='wolf'
	print 'STARTING OBSERVER ****************************', obs
#	fileTable=open(dirOut+'Observer_'+obs+'_parameters_20170119.csv','rb') 
#	fileTable=open(dirOut+'Observer_'+obs+'_parameters_20170126.csv','rb') 
#	fileTable=open(dirOut+'Observer_'+obs+'_parameters_20170209_SPEC_WOLFER.csv','rb') 
#	fileTable=open(dirOut+'Observer_'+obs+'_parameters_20170531_instance0_Daily.csv','rb') 
#	fileTable=open(dirOut+'Observer_'+obs+'_parameters_20170603_instance0_Daily.csv','rb') 
	fileTable=open(dirOut+'Observer_'+obs+'_parameters_20170606_instance0_Daily.csv','rb') 
	csvread= csv.reader(fileTable, delimiter=';')
	
	adfobs=[];cdfobs=[];errcdfobs=[];ncsv=0.
	for column in csvread:
		if ncsv == 0.:
			fraction_obs=float(column[0])
		else:
			readadf=float(column[0])
			readcdf=float(column[1])
			readerrcdf=float(column[2])
			adfobs.append(readadf)
			cdfobs.append(readcdf)
			errcdfobs.append(readerrcdf)
		ncsv+=1.
	#get calibration period from vectors
#	tc1=tcal1[observ] ; tc2=tcal2[observ]
	tc1=int(firstdate) ; tc2=int(lastdate)
	d1=dt.date(tc1,1,1) ; d2=dt.date(tc2,12,31)
	#determine actual number of days and months during the calibration period
	delta=d2-d1
	print 'DELTA', d1, d2
	ndaystot=delta.days
	nmonthtot=(d2.year - d1.year)*12 + (d2.month - d1.month)+1

	#RGO
	#obs='RGO'
	#fraction_obs=1.
	print 'OBSERVER ', obs, ' FRACTION ', fraction_obs

	#*********************************************************************
	#***********************LOOP ON thresholds**************************
	#*********************************************************************
	for i in range(0,int(sizethres)):
		color=iter(cm.rainbow(np.linspace(0,1,sizethres)))
		c=plt.cm.RdYlBu(i)
		print 'COLOR', c

		threshold=vectthres[i]
		thres=str(threshold).strip()
	#*********************************************************************
	#*********************LOOP ON fraction in MC**************************
	#*********************************************************************
		MCsize=1 # Monte Carlo size
		w, h = 11, MCsize 
		MatrixADF = [[0 for x in range(w)] for y in range(h)] 
		Matrixcdf = [[0 for x in range(w)] for y in range(h)] 
		Matrixerrcdf = [[0 for x in range(w)] for y in range(h)] 
		for j in range(0,MCsize):
			jmc=str(j).strip()

			filefraccsv=open(dirOut+obs+'_FracFile_'+thres+'_'+jmc+'_'+fds+'_'+lds+'_'+strdate+'.csv','w')
			fileRGO1csv=open(dirOut+obs+'_RGO_1_File_'+thres+'_'+jmc+'_'+fds+'_'+lds+'_'+strdate+'.csv','w')
			fileRGO2csv=open(dirOut+obs+'_RGO_2_File_'+thres+'_'+jmc+'_'+fds+'_'+lds+'_'+strdate+'.csv','w')
			filecheckcsv=open(dirOut+obs+'_CheckFile_'+thres+'_'+jmc+'_'+fds+'_'+lds+'_'+strdate+'.csv','w')
			filecheckcsv2=open(dirOut+obs+'_CheckFile2_'+thres+'_'+jmc+'_'+fds+'_'+lds+'_'+strdate+'.csv','w')
			fraction=fraction_obs
			for ia in range(0,len(areaRGO)):
				lineOutcsv=('{:8.3f};{:8.3f};{:8.3f};{:8.3f};{:8.3f}').format(float(datesRGO[ia]),float(dateRGO[ia].year),float(dateRGO[ia].month),float(dateRGO[ia].day),float(areaRGO[ia]))
				fileRGO1csv.write(lineOutcsv+'\n')
			fileRGO1csv.close()
			
			#put comments here for RGO with f=1
			print 'BEFORE fraction' , obs, fraction, thres, j 
			list_of_select=select_fraction_days(datesRGO,dateRGO,areaRGO,fraction)
			print  'AFTER fraction', obs, fraction, thres, j , len(list_of_select)
			titi=pd.Series(datesRGO)
			toto=pd.Series(dateRGO)
			tata=pd.Series(areaRGO)
			
			selection=sorted(list(set(list_of_select)))	
			dates22=list(titi[selection])
			date22=list(toto[selection])
			area22=list(tata[selection])
			
			#uncomment for f=1 for RGO
			#dates22=datesRGO
			#date22=dateRGO
			#area22=areaRGO
			
			for ia in range(0,len(area22)):
				lineOutcsv=('{:8.3f};{:8.3f};{:8.3f};{:8.3f};{:8.3f}').format(float(dates22[ia]),float(date22[ia].year),float(date22[ia].month),float(date22[ia].day),float(area22[ia]))
				fileRGO2csv.write(lineOutcsv+'\n')
			fileRGO2csv.close()
			#import pdb; pdb.set_trace()

			#*********************************************************************
			#*****************************FIGURE**********************************
			#*********************************************************************
			fig1=figure(1,figsize=(15.0,4.0),dpi=90)
			matplotlib.rcParams.update({'font.size': 20})
			matplotlib.rc('xtick', labelsize=15) 
			matplotlib.rc('ytick', labelsize=15) 
			plt.axes([0.1, 0.2, 0.8, 0.7]) 
			plt.plot(datesRGO,areaRGO,'.', color='r', linewidth=1, alpha=0.25, label='RGO') 
			plt.plot(dates22,area22,'.', color='b', linewidth=1, alpha=0.25, label='RGO random') 
			#plt.axis([1800., 2016.,0., 20.])
			legend(loc='upper right',prop={'size':11},labelspacing=0.2)
			
			#plt.title('Binned by '+str(binwin)+' days + bin Ri = '+ str(binsize))
			plt.xlabel('Time')
			plt.ylabel('Area of groups')
			savefig(dirOut+obs+'_TimeRandom_'+thres+'_'+fds+'_'+lds+'_'+strdate+'.png',dpi=72)
			#*********************************************************************
			
			if threshold <= 0.:
				threshold=1.
			dates201,date201,gn201=apply_thres2(dates22,date22,area22,threshold)
			dates0,date0,gn0=apply_thres2(dates22,date22,area22,1.)
			
			dates20,date20,gn20=add_values_holes(dates201,date201, gn201, -99.)
			
			
			
			print 'BEFORE/AFTER ADDED VALUES',len(dates201),'/',len(dates20), 'FRACTION:',fraction, '/',float(len(dates201))/float(len(dates20)), ndaystot
			
			#import pdb; pdb.set_trace()
			#it seems this program fills in values that do not exist before
			for ia in range(0,len(gn20)):
				lineOutcsv=('{:8.3f};{:8.3f};{:8.3f};{:8.3f};{:8.3f}').format(float(dates20[ia]),float(date20[ia].year),float(date20[ia].month),float(date20[ia].day),float(gn20[ia]))
				filecheckcsv.write(lineOutcsv+'\n')
			filecheckcsv.close()
			for ia in range(0,len(gn201)):
				lineOutcsv=('{:8.3f};{:8.3f};{:8.3f};{:8.3f};{:8.3f}').format(float(dates201[ia]),float(date201[ia].year),float(date201[ia].month),float(date201[ia].day),float(gn201[ia]))
				filecheckcsv2.write(lineOutcsv+'\n')
			filecheckcsv2.close()
			print 'ADDED VALUES FOR INEXISTENT DATES'
			#import pdb; pdb.set_trace()

			#*********************************************************************
			#*****************************FIGURE**********************************
			#*********************************************************************
			fig2=figure(2,figsize=(15.0,4.0),dpi=90)
			matplotlib.rcParams.update({'font.size': 20})
			matplotlib.rc('xtick', labelsize=15) 
			matplotlib.rc('ytick', labelsize=15) 
			plt.axes([0.1, 0.2, 0.8, 0.7]) 
			plt.plot(dates0,gn0,'.', color='blue', linewidth=1, alpha=0.25, label='RGO T=0') 
			plt.plot(dates201,gn201,'.', color='red', linewidth=1, alpha=0.25, label='RGO T='+thres) 
			#plt.plot(dates22,area22,'.', color='b', linewidth=1, alpha=0.25, label='RGO random') 
			#plt.axis([1800., 2016.,0., 20.])
			legend(loc='upper right',prop={'size':11},labelspacing=0.2)
			
			#plt.title('Binned by '+str(binwin)+' days + bin Ri = '+ str(binsize))
			plt.xlabel('Time')
			plt.ylabel('GN with threshold')
			savefig(dirOut+obs+'_TimeRandom_GN_'+thres+'_'+fds+'_'+lds+'_'+strdate+'.png',dpi=72)
			#*********************************************************************

			#limit for counting in stats 
			limitnd=3.
			#determine number of days with groups for each month within the calibration period
			monthd0, yeard0, ndmonth0, fracdmonth0=find_ndaysinmonthsobs(date20,gn20,d1,d2,0.)
			#import pdb; pdb.set_trace()
			monthd, yeard, ndmonth, fracdmonth1=find_ndaysinmonthsobs(date20,gn20,d1,d2,1.)
			nmonthsobs=find_nmonthsobs(date20,gn20,d1,d2)
			print 'NMONTHOBS',nmonthsobs 
			#import pdb; pdb.set_trace()
			datemonth=[] ; fracdmonth=[];datemonth0=[] ; fracdmonth0=[]
			print 'Count ADF for obs '+obs+' threshold '+thres
			#if obs == 'leppig':
			#	import pdb; pdb.set_trace()

			for k in range(0,len(monthd)):
				print k, yeard[k],monthd[k],ndmonth[k],fracdmonth1[k]#,ndmonth[k]/ndmonth0[k]
				datemonth0.append(dt.date(int(yeard[k]),int(monthd[k]),15))
				if ndmonth[k]==0. and ndmonth0[k] == 0.:
					fracdmonth0.append(0.) # non result should be -1.
				else:
					fracdmonth0.append(ndmonth[k]/ndmonth0[k])
			
				if ndmonth0[j] >= limitnd:
					#print yeard[k],monthd[k],ndmonth[k],fracdmonth1[k]#,ndmonth[k]/ndmonth0[k]
					datemonth.append(dt.date(int(yeard[k]),int(monthd[k]),15))
					if ndmonth[k]==0. and ndmonth0[k] == 0.:
						fracdmonth.append(0.) # non result should be -1.
					else:
						fracdmonth.append(ndmonth[k]/ndmonth0[k])
				#*********************************************************************
			for k in range(0,len(fracdmonth0)):
				if ndmonth0[k] < limitnd :
					lineOutcsv=('{:8.3f};{:8.3f};{:8.3f};{:8.3f};{:8.3f} *').format(float(datemonth0[k].year),float(datemonth0[k].month),float(fracdmonth0[k]),float(ndmonth[k]),float(ndmonth0[k]))
				else:
					lineOutcsv=('{:8.3f};{:8.3f};{:8.3f};{:8.3f};{:8.3f}').format(float(datemonth0[k].year),float(datemonth0[k].month),float(fracdmonth0[k]),float(ndmonth[k]),float(ndmonth0[k]))
				filefraccsv.write(lineOutcsv+'\n')
			filefraccsv.close()

			#*********************************************************************
			#*****************************FIGURE**********************************
			#*********************************************************************
			fig3=figure(3,figsize=(15.0,4.0),dpi=90)
			matplotlib.rcParams.update({'font.size': 20})
			matplotlib.rc('xtick', labelsize=15) 
			matplotlib.rc('ytick', labelsize=15) 
			plt.axes([0.1, 0.2, 0.8, 0.7]) 
			#plt.plot(date1,gn1,'.', color='r', linewidth=1, alpha=0.25, label='Wolf') 
			#plt.plot(date2,gn2,'.', color='b', linewidth=1, alpha=0.25, label='Wolfer') 
			plt.plot(datemonth,fracdmonth,'o', color='r', linewidth=2, alpha=0.25, label='RGO') 
			#plt.plot(dateintersect,gn2_inters,'.', color='b', linewidth=1, alpha=0.25, label=obs2) 
			#plt.axis([1800., 2016.,0., 20.])
			#legend(loc='upper right',prop={'size':11},labelspacing=0.2)
			
			#plt.title('Binned by '+str(binwin)+' days + bin Ri = '+ str(binsize))
			plt.xlabel('Time ('+obs+')')
			plt.ylabel('ADF in months')
			savefig(dirOut+obs+'_Fracmonth_RGO_'+thres+'_'+fds+'_'+lds+'_'+strdate+'.png',dpi=72)
			#*********************************************************************




			#count the number of months with A (fracdmonth) between 2 values:
			ADF=[0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7, 0.8,0.9,1.]
			ncdf=[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
			cdf=[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
			errcdf=[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
			for jj in range(0,len(ADF)):
				ncdf[jj]=sum(1 for k in fracdmonth if my_condition(k,ADF[0],ADF[jj]))
				nmonthsan=sum(1 for k in fracdmonth) #number of months analysed (removed months with less than 3 days observed)
				norm= nmonthsobs # nmonthsan,nmonthtot, nmonthsobs
				cdf[jj]=float(ncdf[jj])/float(norm) 
				errcdf[jj]=numpy.sqrt(ncdf[jj])/float(norm) 
				print 'RANGE ', ADF[0], ADF[jj],' CDF', ncdf[jj], norm, cdf[jj], errcdf[jj]
				MatrixADF[j][jj]=ADF[jj]
				Matrixcdf[j][jj]=cdf[jj]
				Matrixerrcdf[j][jj]=errcdf[jj]
			#print 'CHECK FILE ', filefraccsv
			#import pdb; pdb.set_trace()


		#*********************************************************************
		#fig2=figure(2,figsize=(15.0,4.0),dpi=90)
		add=200
		figname2='fig2'+obs
		figname2=figure(observ+add,figsize=(10.0,8.0),dpi=90)
		matplotlib.rcParams.update({'font.size': 20})
		matplotlib.rc('xtick', labelsize=15) 
		matplotlib.rc('ytick', labelsize=15) 
		plt.axes([0.1, 0.2, 0.8, 0.7]) 
		#plt.plot(date1,gn1,'.', color='r', linewidth=1, alpha=0.25, label='Wolf') 
		#plt.plot(date2,gn2,'.', color='b', linewidth=1, alpha=0.25, label='Wolfer') 
		#plt.plot(ADF,cdf,'o', color='g', linewidth=1, alpha=0.5, label=obs) 
		plt.plot(ADF[0:-1],cdf[0:-1], color=cmap((i+1.)/ float(sizethres)), linewidth=1, label=obs+'_'+thres) 
		plt.errorbar(ADF[0:-1],cdf[0:-1], yerr=errcdf[0:-1],color=cmap((i+1.)/ float(sizethres)), fmt='o')
		plt.plot(ADF[0:-1],cdf[0:-1],'o',color=cmap((i+1.)/ float(sizethres)), linewidth=1, alpha=0.5,markersize=2) 
		if i >= sizethres-1.:
			plt.plot(adfobs[0:-1],cdfobs[0:-1], color='red', linewidth=1, label=obs) 
			plt.errorbar(adfobs[0:-1],cdfobs[0:-1], yerr=errcdfobs[0:-1],color='red', fmt='o')
			plt.plot(adfobs[0:-1],cdfobs[0:-1],'o', color='red', linewidth=1, alpha=0.5,markersize=2) 
			
		#plt.plot(dateintersect,gn2_inters,'.', color='b', linewidth=1, alpha=0.25, label=obs2) 
		plt.axis([0.,0.95, 0.,2.*cdf[-2]])
		legend(loc='upper right',prop={'size':11},labelspacing=0.2)
		
		#plt.title('Binned by '+str(binwin)+' days + bin Ri = '+ str(binsize))
		plt.xlabel('ADF for RGO-obs '+obs)
		plt.ylabel('CDF of ADF ')
		savefig(dirOut+obs+'_CDF_'+thres+'_'+fds+'_'+lds+'_'+strdate+'.png',dpi=72)
		#*********************************************************************
	#import pdb; pdb.set_trace()

	

