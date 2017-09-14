#apply_ADF_part1.py

#===============================================================================
#strdate='20170117'
#strdate='20170118'
#strdate='20170119'
#strdate='20170123'
#strdate='20170126'
#strdate='20170130'
#strdate='20170130_check'
#strdate='20170131'
#strdate='20170209_SPEC_WOLFER'
#strdate='20170531_instance0_corr' #nmonthsan
#strdate='20170531_instance0_Daily'
#strdate='20170601_instance2_Daily'
#strdate='20170601_instance0_Daily'
#strdate='20170603_instance0_Daily' #nmonthobs
strdate='20170606_instance0_Daily' #nmonthobs

import re
import random
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


#dirIn='/Users/Laure/Desktop/SSNWork/corr/' #values from Levi 04/2016
dirIn='/Users/Laure/Desktop/SSNWork/Daily/' #new values Ilya 05/2017
dirBase='/Users/Laure/sunspots/SSN/SSNCALC/newSN/'
dircreate=dirBase+'results_ADF/'
if not os.path.exists(dircreate):
    os.makedirs(dircreate)

dirOut=dircreate
#fileObscsv=open(dirOut+'Observers_'+obs1+'_'+obs2+'.csv','w')
#filemediancsv=open(dirOut+'Observers_median_'+obs1+'_'+obs2+'.csv','w')



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
			if monthold+yearold != currentmonth+currentyear and monthold != 0.:
				monthdd.append(monthold)
				yeardd.append(yearold)
				ndmonthd.append(nmonthsobs)
				ndaysmonth=num_days_in_month(yearold, monthold)
				fracdmonthd.append(nmonthsobs/ndaysmonth)
				nmonthsobs=0. 
			if gn1[i] >=limit: 
				nmonthsobs+=1.
			if gn1[i] >=0.: 
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
def find_ngroupsinmonthsobs(date1,gn1,d1,d2,limit):
	monthold=0. ; yearold=0.
	yeardd=[]; monthdd=[];ndmonthd=[];fracdmonthd=[];ngmonthd=[];fracgrmonth=[]
	nmonthsobs=0. ; ngroupsobs=0.
	for i in range(0, len(date1)):
		if date1[i] >= d1 and date1[i] <= d2:
			currentmonth=date1[i].month
			currentyear=date1[i].year
			if monthold+yearold != currentmonth+currentyear and monthold != 0.:
				monthdd.append(monthold)
				yeardd.append(yearold)
				ndmonthd.append(nmonthsobs)
				ngmonthd.append(ngroupsobs)
				ndaysmonth=num_days_in_month(yearold, monthold)
				fracdmonthd.append(nmonthsobs/ndaysmonth)
				fracgrmonth.append(ngroupsobs/ndaysmonth)
				nmonthsobs=0. 
				ngroupsobs=0. 
			if gn1[i] >=limit: 
				nmonthsobs+=1.
				ngroupsobs+=gn1[i]
			if gn1[i] >=0.: 
				monthold=currentmonth
				yearold=currentyear
	monthdd.append(monthold)
	yeardd.append(yearold)
	ndmonthd.append(nmonthsobs)
	ngmonthd.append(ngroupsobs)
	ndaysmonth=num_days_in_month(yearold, monthold)
	fracdmonthd.append(nmonthsobs/ndaysmonth)
	fracgrmonth.append(ngroupsobs/ndaysmonth)

	return monthdd, yeardd, ndmonthd, fracdmonthd,ngmonthd,fracgrmonth
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
def my_condition(x,a,b):
	return x >=a and x <= b
#===============================================================================


#observers=['quimby','wolfer','winkler','tacchini','leppig','spoerer','weber','wolf','shea','schmidt','schwabe','pastorff']
#tcal1=[1900,1900,1889,1879,1867,1865,1859,1860,1847,1841,1832,1824]
#tcal2=[1921,1928,1910,1900,1880,1893,1883,1893,1866,1883,1867,1833]
#Instance0
observers=['quimby','wolfer','winkler','broger','tacchini','leppig','spoerer','weber','wolf','shea','schmidt','schwabe','pastorff']
tcal1=[1900,1900,1889,1900,1879,1867,1865,1859,1860,1847,1841,1832,1824]
tcal2=[1921,1928,1910,1935,1900,1880,1893,1883,1893,1866,1883,1866,1833]
#Instance1
#observers=['quimby','wolfer','winkler','broger','tacchini','leppig','spoerer','weber','wolf','shea','schmidt','schwabe','pastorff']
#tcal1=[1900,1902,1889,1900,1879,1867,1865,1859,1860,1847,1841,1832,1824]
#tcal2=[1921,1923,1910,1935,1900,1880,1893,1883,1893,1866,1883,1866,1833]
#Instance2
#observers=['quimby','wolfer','winkler','broger','tacchini','leppig','spoerer','weber','wolf','shea','schmidt','schwabe','pastorff']
#tcal1=[1902,1902,1900,1902,1879,1867,1880,1859,1860,1847,1841,1832,1824]
#tcal2=[1921,1923,1910,1923,1900,1880,1893,1883,1893,1866,1883,1866,1833]
nbobs=len(observers)
ff=[]

#os.chdir(dirIn)
#for file in glob.glob("*.dat"):
for i in range(0,nbobs) :
	#obs=file[7:-4]
	obs=observers[i]
#	file='corr_d_'+obs+'.dat' # corr files
	file=obs+'_corr.dat' # Daily files
	print 'FILE *', file, '*'
	fileObscsv=open(dirOut+'Observer_'+obs+'_parameters_'+strdate+'.csv','w')
	filecheckcsv=open(dirOut+'Observer_'+obs+'_CheckFracmonthFile.csv','w')

	#**************READ OBSERVER DATA*************************************
	#Declare all tables
	year1=[] ; month1=[] ; day1=[] ; gn1=[] ; date1=[]; gn1corrIU=[] ; dates1=[]
	fileTable=open(dirIn+file,'r') 
	strlines=fileTable.readlines()
	fileTable.close()
		
	nblines=len(strlines)
	for j in range(0,nblines) :
		column=strlines[j].split()
		readyear=float(column[0])
		readmonth=float(column[1])
		readday=float(column[2])
		readgn=float(column[3])
		_, ndays=calendar.monthrange(int(readyear),int(readmonth))
		#print readyear, readmonth, readday, ndays
		if readday <= ndays and readgn >= 0:
			date1.append(dt.date(int(readyear),int(readmonth),int(readday)))
			dates1.append(date_conv(readyear, readmonth, mt.floor(readday), 0., 0., 0.))
			year1.append(float(column[0]))
			month1.append(float(column[1]))
			day1.append(float(column[2]))
			gn1.append(float(column[3]))
			gn1corrIU.append(float(column[4]))
	#*********************************************************************
	#import pdb; pdb.set_trace()
	
	#determine actual number of days and months during the observations
	delta0=date1[-1]-date1[0]
	ndaystot0=delta0.days
	nmonthtot0=(date1[-1].year - date1[0].year)*12 + date1[-1].month - date1[0].month
	
	#get calibration period from vectors
	tc1=tcal1[i] ; tc2=tcal2[i]
	tc1s=str(tc1).strip() ; tc2s=str(tc2).strip()
	d1=dt.date(tc1,1,1) ; d2=dt.date(tc2,12,31)
	
	#determine actual number of days and months during the calibration period
	delta=d2-d1
	print 'DELTA', d1, d2
	ndaystot=delta.days
	nmonthtot=(d2.year - d1.year)*12 + (d2.month - d1.month)+1
	
	
	#determine number of days with observations within the calibration period
	#import pdb; pdb.set_trace()

	ndaysobs=find_ndaysobs(date1,gn1,d1,d2)
	
	#determine fraction of days observed within the calibration period
	
	fracdays=ndaysobs/ndaystot
	ff.append(fracdays)
	lineOutcsv=('{:5.2f};{:5.2f};{:5.2f}').format(float(fracdays),float(-1.),float(-1.))
	fileObscsv.write(lineOutcsv+'\n')

	#determine number of months with observation within the calibration period
	
	#date1s,gn1s=return_subset(date1,gn1,d1,d2)	
	nmonthsobs=find_nmonthsobs(date1,gn1,d1,d2)
	print 'NMONTHOBS,nmonthtot',nmonthsobs,nmonthtot
	fracmonths=nmonthsobs/nmonthtot
	
	print 'OBSERVER ', obs, fracdays, fracmonths
	
	#determine number of days with groups for each month within the calibration period
	#monthd0, yeard0, ndmonth0, fracdmonth0=find_ndaysinmonthsobs(date1,gn1,d1,d2,0.)
	#monthd, yeard, ndmonth, fracdmonth1=find_ndaysinmonthsobs(date1,gn1,d1,d2,1.)
	#determine here the group number for the month corresponding to the fracmonth in question
	#*******************
	monthd0, yeard0, ndmonth0, fracdmonth0,ngmonthd0,fracgrmonth0=find_ngroupsinmonthsobs(date1,gn1,d1,d2,0.)
	monthd, yeard, ndmonth, fracdmonth1,ngmonthd,fracgrmonth1=find_ngroupsinmonthsobs(date1,gn1,d1,d2,1.)
	
	
	#limit for counting in stats 
	limitnd=3.

	print 'COMP', len(ndmonth0), len(ndmonth),nmonthsobs,nmonthtot
	datemonth=[] ; fracdmonth=[] ; datemonth0=[]; fracdmonth0=[]
	fracgmonth=[] ; fracgmonth0=[]
	for j in range(0,len(monthd)):
		if ndmonth0[j] >= limitnd:
			print yeard[j],monthd[j],ndmonth[j],fracdmonth1[j],ndmonth[j]/ndmonth0[j],ngmonthd[j],ngmonthd0[j],ndmonth0[j],ngmonthd[j]/ndmonth0[j]
			datemonth.append(dt.date(int(yeard[j]),int(monthd[j]),15))
			fracdmonth.append(ndmonth[j]/ndmonth0[j])
			fracgmonth.append(ngmonthd[j]/ndmonth0[j])
		datemonth0.append(dt.date(int(yeard[j]),int(monthd[j]),15))
		fracdmonth0.append(ndmonth[j]/ndmonth0[j])
		fracgmonth0.append(ngmonthd0[j]/ndmonth0[j])
	#import pdb; pdb.set_trace()
	for j in range(0,len(fracdmonth0)):
		print j, ndmonth0[j], limitnd
		if ndmonth0[j] < limitnd :
			#print float(datemonth0[j].month),float(fracdmonth0[j]),float(ndmonth[j]),float(ndmonth0[j]),float(fracgmonth0[j]),float(ngmonthd[j])
			lineOutcsv=('{:8.3f};{:8.3f};{:8.3f};{:8.3f};{:8.3f};{:8.3f};{:8.3f} *').format(float(datemonth0[j].year),float(datemonth0[j].month),float(fracdmonth0[j]),float(ndmonth[j]),float(ndmonth0[j]),float(fracgmonth0[j]),float(ngmonthd[j]))
		else:
			#print float(datemonth0[j].year),float(datemonth0[j].month),float(fracdmonth0[j]),float(ndmonth[j]),float(ndmonth0[j]),float(fracgmonth0[j]),float(ngmonthd[j])
			lineOutcsv=('{:8.3f};{:8.3f};{:8.3f};{:8.3f};{:8.3f};{:8.3f};{:8.3f}').format(float(datemonth0[j].year),float(datemonth0[j].month),float(fracdmonth0[j]),float(ndmonth[j]),float(ndmonth0[j]),float(fracgmonth0[j]),float(ngmonthd[j]))
		filecheckcsv.write(lineOutcsv+'\n')
		print lineOutcsv
	filecheckcsv.close()
	#import pdb; pdb.set_trace()

	#*********************************************************************
	#fig1=figure(1,figsize=(15.0,4.0),dpi=90)
	add=100
	figname1='fig1'+str(i).strip()
	figname1=figure(i+add,figsize=(15.0,4.0),dpi=90)

	matplotlib.rcParams.update({'font.size': 20})
	matplotlib.rc('xtick', labelsize=15) 
	matplotlib.rc('ytick', labelsize=15) 
	plt.axes([0.1, 0.2, 0.8, 0.7]) 
	#plt.plot(date1,gn1,'.', color='r', linewidth=1, alpha=0.25, label='Wolf') 
	#plt.plot(date2,gn2,'.', color='b', linewidth=1, alpha=0.25, label='Wolfer') 
	plt.plot(datemonth0,fracdmonth0,'o', color='r', linewidth=2, alpha=0.25, label=obs) 
	#plt.plot(dateintersect,gn2_inters,'.', color='b', linewidth=1, alpha=0.25, label=obs2) 
	#plt.axis([1800., 2016.,0., 20.])
	#legend(loc='upper right',prop={'size':11},labelspacing=0.2)
	
	#plt.title('Binned by '+str(binwin)+' days + bin Ri = '+ str(binsize))
	plt.xlabel('Time')
	plt.ylabel('Fraction of active days in months')
	savefig(dirOut+obs+'Fracmonth_'+strdate+'.png',dpi=72)
	#*********************************************************************
	#*********************************************************************
	#fig4=figure(4,figsize=(15.0,4.0),dpi=90)
	add=400
	figname4='fig4'+str(i).strip()
	figname4=figure(i+add,figsize=(15.0,15.0),dpi=90)

	matplotlib.rcParams.update({'font.size': 20})
	matplotlib.rc('xtick', labelsize=15) 
	matplotlib.rc('ytick', labelsize=15) 
	plt.axes([0.1, 0.2, 0.8, 0.7]) 
	#plt.plot(date1,gn1,'.', color='r', linewidth=1, alpha=0.25, label='Wolf') 
	#plt.plot(date2,gn2,'.', color='b', linewidth=1, alpha=0.25, label='Wolfer') 
	plt.plot(fracgmonth0,fracdmonth0,'o', color='r', linewidth=2, alpha=0.25, label=obs) 
	#plt.plot(dateintersect,gn2_inters,'.', color='b', linewidth=1, alpha=0.25, label=obs2) 
	#plt.axis([1800., 2016.,0., 20.])
	#legend(loc='upper right',prop={'size':11},labelspacing=0.2)
	
	#plt.title('Binned by '+str(binwin)+' days + bin Ri = '+ str(binsize))
	plt.xlabel('GN / month ('+obs+')')
	plt.ylabel('ADF / month')
	savefig(dirOut+obs+'_ADF_GN_'+strdate+'.png',dpi=72)
	#*********************************************************************
	WOLF = [x * 20.13 for x in fracgmonth0]
	#*********************************************************************
	#fig6=figure(6,figsize=(15.0,15.0),dpi=90)
	add=600
	figname6='fig6'+str(i).strip()
	figname6=figure(i+add,figsize=(15.0,15.0),dpi=90)

	matplotlib.rcParams.update({'font.size': 20})
	matplotlib.rc('xtick', labelsize=15) 
	matplotlib.rc('ytick', labelsize=15) 
	plt.axes([0.1, 0.2, 0.8, 0.7]) 
	#plt.plot(date1,gn1,'.', color='r', linewidth=1, alpha=0.25, label='Wolf') 
	#plt.plot(date2,gn2,'.', color='b', linewidth=1, alpha=0.25, label='Wolfer') 
	plt.plot(WOLF,fracdmonth0,'o', color='r', linewidth=2, alpha=0.25, label=obs) 
	#plt.plot(dateintersect,gn2_inters,'.', color='b', linewidth=1, alpha=0.25, label=obs2) 
	#plt.axis([1800., 2016.,0., 20.])
	#legend(loc='upper right',prop={'size':11},labelspacing=0.2)
	
	#plt.title('Binned by '+str(binwin)+' days + bin Ri = '+ str(binsize))
	plt.xlabel('SN(GN) / month ('+obs+')')
	plt.ylabel('ADF / month')
	savefig(dirOut+obs+'_ADF_SNGN_'+strdate+'.png',dpi=72)
	#*********************************************************************
	#*********************************************************************

	#histogram of GN values with A (fracdmonth) between 2 values:
	ADF=[0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7, 0.8,0.9,1.]
	for j in range(0,len(ADF)-1):
		ADFS=str(ADF[j]).strip()
		ADFS1=str(ADF[j+1]).strip()

		histoplot=[] ; histoplot_wolf=[]
		ADFA=np.asarray(fracdmonth0) ; GN=np.asarray(fracgmonth0) 
		titi=np.where((ADFA >= ADF[j]) & (ADFA <= ADF[j+1]))
		NG=len(titi[0])
		if NG > 0.:
			print NG
			#import pdb; pdb.set_trace()
			for jj in range(0,NG):
				histoplot.append(fracgmonth0[titi[0][jj]])
				histoplot_wolf.append(fracgmonth0[titi[0][jj]]*12.08)
		#for each ADF plot histogram of NG
		print 'RANGE ', ADF[j], ADF[j+1], len(histoplot)
		#print histoplot
		#import pdb; pdb.set_trace()

		#*********************************************************************
		if len(histoplot) > 1.:
			#fig5=figure(5,figsize=(15.0,4.0),dpi=90)
			add=5000
			figname5='fig50'+str(i).strip()+str(j).strip()
			#print figname5
			figname5=figure(i+add+20*j,figsize=(10.0,8.0),dpi=90)
			#print figname5
			
			matplotlib.rcParams.update({'font.size': 20})
			matplotlib.rc('xtick', labelsize=15) 
			matplotlib.rc('ytick', labelsize=15) 
			plt.axes([0.1, 0.2, 0.8, 0.7]) 
			plt.hist(histoplot, bins=20, histtype='stepfilled', color='red', alpha=0.25, label='With all obs.'+str(i).strip()+'_'+str(j).strip())
			#plt.hist(fracgmonth, bins=11, histtype='stepfilled', color='green', alpha=0.25,label='No months with nd< 3')
			legend(loc='upper left',prop={'size':11},labelspacing=0.2)
		
			#plt.title('Binned by '+str(binwin)+' days + bin Ri = '+ str(binsize))
			plt.xlabel('NG for observer '+obs+' ADF >='+ADFS+' AND ADF <= '+ADFS1)
			plt.ylabel('Number')
			savefig(dirOut+obs+'_HISTO_GN_ADF_'+ADFS+'_'+strdate+'.png',dpi=72)
			#*********************************************************************
			#*********************************************************************
			#fig6=figure(6,figsize=(15.0,4.0),dpi=90)
			add=6000
			figname6='fig60'+str(i).strip()+str(j).strip()
			figname6=figure(i+add+20*j,figsize=(10.0,8.0),dpi=90)
			
			matplotlib.rcParams.update({'font.size': 20})
			matplotlib.rc('xtick', labelsize=15) 
			matplotlib.rc('ytick', labelsize=15) 
			plt.axes([0.1, 0.2, 0.8, 0.7]) 
			plt.hist(histoplot_wolf, bins=20, histtype='stepfilled', color='red', alpha=0.25, label='With all obs.'+str(i).strip()+'_'+str(j).strip())
			#plt.hist(fracgmonth, bins=11, histtype='stepfilled', color='green', alpha=0.25,label='No months with nd< 3')
			legend(loc='upper left',prop={'size':11},labelspacing=0.2)
		
			#plt.title('Binned by '+str(binwin)+' days + bin Ri = '+ str(binsize))
			plt.xlabel('Wolf* for observer '+obs+' ADF >='+ADFS+' AND ADF <= '+ADFS1)
			plt.ylabel('Number')
			savefig(dirOut+obs+'_HISTO_SN_ADF_'+ADFS+'_'+strdate+'.png',dpi=72)
			#*********************************************************************
	#import pdb; pdb.set_trace()
	
	#*********************************************************************
	#*********************************************************************
	add=1000
	figname11='fig11'+str(i).strip()
	figname11=figure(i+add,figsize=(15.0,8.0),dpi=90)
	#fig11=figure(11,figsize=(15.0,8.0),dpi=90)
	
	plt.subplot(311)
	matplotlib.rcParams.update({'font.size': 20})
	matplotlib.rc('xtick', labelsize=15) 
	matplotlib.rc('ytick', labelsize=15) 
	plt.grid(True)
	plt.plot(datemonth0,ndmonth0,'o', color='r', linewidth=2, alpha=0.25, label=obs) 
	plt.xlabel('Time')
	plt.ylabel('OBS Days')
	
	
	plt.subplot(312)
	matplotlib.rcParams.update({'font.size': 20})
	matplotlib.rc('xtick', labelsize=15) 
	matplotlib.rc('ytick', labelsize=15) 
	plt.grid(True)
	plt.plot(datemonth0,ndmonth,'o', color='r', linewidth=2, alpha=0.25, label=obs) 
	plt.xlabel('Time')
	plt.ylabel('ACT Days')
	
	plt.subplot(313)
	matplotlib.rcParams.update({'font.size': 20})
	matplotlib.rc('xtick', labelsize=15) 
	matplotlib.rc('ytick', labelsize=15) 
	plt.grid(True)
	plt.plot(datemonth,fracdmonth,'o', color='r', linewidth=2, alpha=0.25, label=obs) 
	plt.xlabel('Time')
	plt.ylabel('ADF/month')
	
	
	savefig(dirOut+obs+'Fracmonth_all_'+strdate+'.png',dpi=72)
	#*********************************************************************

	#count the number of months with A (fracdmonth) between 2 values:
	ADF=[0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7, 0.8,0.9,1.]
	ncdf=[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
	cdf=[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
	errcdf=[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
	for j in range(0,len(ADF)):
		ncdf[j]=sum(1 for k in fracdmonth if my_condition(k,ADF[0],ADF[j]))
		nmonthsan=sum(1 for k in fracdmonth) #number of months analysed (removed months with less than 3 days observed)
		norm= nmonthsobs # nmonthsan,nmonthtot, nmonthsobs
		#it seems usoskin et al. normalize with the number of days observed in the month and not with the number of months with nobs>=3
		#I would have used the nmonthsan, because the cumulative DF for 1 should be 1
		#it is not the case in Usoskin method I think, but as he does not plot above 0.9, impossible to be sure
	#	for k in range(0,len(fracdmonth)):
	#		if fracdmonth[k] >=ADF[0] and fracdmonth[k] <= ADF[j]:
	#			ncdf[j]+=1.
		cdf[j]=float(ncdf[j])/float(norm) 
		errcdf[j]=numpy.sqrt(ncdf[j])/float(norm) 
		print '*********RANGE ', ADF[0], ADF[j],' CDF', ncdf[j], norm, cdf[j], errcdf[j],nmonthsan,nmonthtot, nmonthsobs
	#if obs == 'quimby' :
	#	import pdb; pdb.set_trace()

	#*********************************************************************
	#fig2=figure(2,figsize=(15.0,4.0),dpi=90)
	add=200
	figname2='fig2'+str(i).strip()
	figname2=figure(i+add,figsize=(10.0,8.0),dpi=90)
	matplotlib.rcParams.update({'font.size': 20})
	matplotlib.rc('xtick', labelsize=15) 
	matplotlib.rc('ytick', labelsize=15) 
	plt.axes([0.1, 0.2, 0.8, 0.7]) 
	#plt.plot(date1,gn1,'.', color='r', linewidth=1, alpha=0.25, label='Wolf') 
	#plt.plot(date2,gn2,'.', color='b', linewidth=1, alpha=0.25, label='Wolfer') 
	#plt.plot(ADF,cdf,'o', color='g', linewidth=1, alpha=0.5, label=obs) 
	plt.plot(ADF[0:-1],cdf[0:-1], color='b', linewidth=1, label=obs) 
	plt.errorbar(ADF[0:-1],cdf[0:-1], yerr=errcdf[0:-1],color='black', fmt='o')
	plt.plot(ADF[0:-1],cdf[0:-1],'o', color='b', linewidth=1, alpha=0.5, label=obs,markersize=2) 
	#plt.plot(dateintersect,gn2_inters,'.', color='b', linewidth=1, alpha=0.25, label=obs2) 
	plt.axis([0.,0.95, 0.,1.2*cdf[-2]])
	#legend(loc='upper right',prop={'size':11},labelspacing=0.2)
	
	#plt.title('Binned by '+str(binwin)+' days + bin Ri = '+ str(binsize))
	plt.xlabel('ADF for observer '+obs)
	plt.ylabel('CDF of ADF ')
	savefig(dirOut+obs+'_CDF_'+strdate+'.png',dpi=72)
	#*********************************************************************
	#*********************************************************************
	#fig3=figure(3,figsize=(15.0,4.0),dpi=90)
	add=300
	figname3='fig3'+str(i).strip()
	figname3=figure(i+add,figsize=(10.0,8.0),dpi=90)
	matplotlib.rcParams.update({'font.size': 20})
	matplotlib.rc('xtick', labelsize=15) 
	matplotlib.rc('ytick', labelsize=15) 
	plt.axes([0.1, 0.2, 0.8, 0.7]) 
	plt.hist(fracdmonth0, bins=11, histtype='stepfilled', color='red', alpha=0.25, label='With all obs.')
	plt.hist(fracdmonth, bins=11, histtype='stepfilled', color='green', alpha=0.25,label='No months with nd< 3')
	legend(loc='upper left',prop={'size':11},labelspacing=0.2)

	#plt.title('Binned by '+str(binwin)+' days + bin Ri = '+ str(binsize))
	plt.xlabel('ADF for observer '+obs)
	plt.ylabel('N months with ADF')
	savefig(dirOut+obs+'_HISTO_ADF_'+strdate+'.png',dpi=72)
	#*********************************************************************
	#create an output file for each observer that I can read in Part II of the algorithm
	#first line is fraction of observed days
	#after that 3 columns with ADF, CDF, errCDF
	#import pdb; pdb.set_trace()

	for ii in range(0,len(ADF)):
		lineOutcsv=('{:5.3f};{:5.3f};{:5.3f}').format(float(ADF[ii]),float(cdf[ii]),float(errcdf[ii]))
		fileObscsv.write(lineOutcsv+'\n')

fileObscsv.close()
	#import pdb; pdb.set_trace()

