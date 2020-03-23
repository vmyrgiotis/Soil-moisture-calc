# data            monthly data of temperature and precipitation. 
# latitude        latitude of the station in degrees.
# clim_norm       climatic normals.
# first_yr        first year of the period over which water balance is calculated. Default is \code{NULL} (calculations start with the first year of the series).
# last_yr         last year of the period over which water balance is calculated. Default is \code{NULL} (calculations stop with the last year of the series).
# quant           vector of quantiles for which water balance has to be assessed. Default is: min, 10th, 25th 50th, 75th, 90th, max.
# snow_init       initial water equivalent for snowpack (mm). Default is 20.
# Tsnow           maximum temperature (monthly mean) for precipitation to be treated as snowfall. Default is -1 degree C.
# TAW             maximum (field capacity) for soil water retention, and initial soil water content (mm). Default is 100.
# fr_sn_acc       fraction of snow that contributes to snowpack (0-1). fr_sn_acc is treated as liquid monthly precipitation Default is 0.95.
# snow_melt_coeff monthly coefficient(s) for snowmelt. Default is 1.

import pandas as pd 
import numpy as np
import daylength_calc 
import math
import datetime as dt
import matplotlib.pyplot as plt 

import numpy as np

def daylength(dayOfYear, lat):
    """Computes the length of the day (the time between sunrise and
    sunset) given the day of the year and latitude of the location.
    Function uses the Brock model for the computations.
    For more information see, for example,
    Forsythe et al., "A model comparison for daylength as a
    function of latitude and day of year", Ecological Modelling,
    1995.
    Parameters
    ----------
    dayOfYear : int
        The day of the year. 1 corresponds to 1st of January
        and 365 to 31st December (on a non-leap year).
    lat : float
        Latitude of the location in degrees. Positive values
        for north and negative for south.
    Returns
    -------
    d : float
        Daylength in hours.
    """
    latInRad = np.deg2rad(lat)
    declinationOfEarth = 23.45*np.sin(np.deg2rad(360.0*(283.0+dayOfYear)/365.0))
    if -np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth)) <= -1.0:
        return 24.0
    elif -np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth)) >= 1.0:
        return 0.0
    else:
        hourAngle = np.rad2deg(np.arccos(-np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth))))
        return 2.0*hourAngle/15.0

def SWB(data,latitude,Tsnow,TAW,first_yr,last_yr) :

	snow_init = 20
	fr_sn_acc = 0.95
	snow_melt_coeff = 1 

	# month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
	# month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

	def ExAtRa(DOY,latitude):
		# Calculates Extra-Atmospheric Radiation
		# DOY & latitude in degrees (negative for S emishpere).
		# unit for solar radiation is "MJ"
		Gsc    = 0.0820 # solar constant in MJ m-2 min-1 (default:0.0820) 
		pi     = 3.14159
		phi    = latitude*pi/180
		delta  = 0.4093 * math.sin(2*pi*(284+DOY)/365) # declination
		dr     = 1 + 0.033*math.cos(2*pi*DOY/365)      # relative distance Earth - Sun
		omegaS = math.acos(-math.tan(phi)*math.tan(delta))     # Sun angle at sunset  
		R_MJ   = (24*(60)/pi)*Gsc*dr*((omegaS)*math.sin(phi)*math.sin(delta) + math.cos(phi)*math.cos(delta)*math.sin(omegaS))
		return R_MJ 

	yDay = []
	YYYY = range(first_yr,last_yr+1)
	snow_melt_coeff = [snow_melt_coeff]*len(YYYY)*12

	for i in range(last_yr-first_yr+1): 
		for MM in range(1,13) :
			yDay.append((dt.date(YYYY[i], MM, 15) - dt.date(YYYY[i],1,1)).days + 1)

	rel_daylength = []
	for i in range(12) : rel_daylength.append(daylength_calc.daylength(yDay[i],latitude)/float(12))

	chg = []
	for i in range(12) : chg.append(ExAtRa(DOY = yDay[i], latitude = latitude))

	fTAW = (-1.05e-07 * TAW**2 + 5.4225e-05 * TAW - 0.00878)
	ultStorPrA = TAW

	years = range(first_yr,last_yr)
	df = pd.DataFrame(columns=['year','month','prec','PET','Storage','P_minus_ET','Def','Sur'])
	snowpack_jan = snow_init

	for ii in range(len(YYYY)) :

		Prec = np.array(data['P'][data.year == YYYY[ii]])
		Tmean = np.array((data['Tn'][data.year == YYYY[ii]]+data['Tx'][data.year == YYYY[ii]])*0.5)
		Storage = np.repeat(TAW, len(Tmean), axis=0)
		Percola  = np.repeat(0, len(Tmean), axis=0)
		TmeanCor = Tmean
		TmeanCor[TmeanCor < 0] = 0
		ITM = ((TmeanCor/5)**1.514)
		ITA = sum(ITM)
		exp = 6.75e-07 * ITA**3 - 7.71e-05 * ITA**2 + 0.01792 * ITA + 0.492
		PET = (16 * (10 * TmeanCor/float(ITA))**exp) * np.array(rel_daylength)
		SnowAcc  = np.repeat(0, [12], axis=0)
		SnowPrec = np.copy(Prec)
		SnowPrec[Tmean >= Tsnow] = 0
		month_max = max(min(np.where(Tmean >= Tsnow)[0] - 1),1)
		SnowAcc_wint = []

		for i in range(0,month_max) :
			if (i == 0) :
				SnowAcc_wint.append(snowpack_jan + SnowPrec[i] * fr_sn_acc)
			else : 
				SnowAcc_wint.append(SnowAcc_wint[i - 1] + SnowPrec[i] * fr_sn_acc)

		snowpack_ref = SnowAcc_wint[month_max-1]
		snow_depl    = np.zeros([12]) 
		SnowAcc[0]   = SnowAcc_wint[0]
		snow_depl[0] = (SnowPrec[0] * (1 - fr_sn_acc))
		if (Tmean[0] >= Tsnow) :
			snow_depl[0] = (snow_melt_coeff[0] * SnowAcc[0])
			SnowAcc[0]   = SnowAcc[0] - snow_depl[0]
		
		count_thaw = 0
		
		for i in range(1,12):
			snow_depl[i] = SnowPrec[i] * (1 - fr_sn_acc)
			SnowAcc[i] = SnowAcc[i - 1] + SnowPrec[i] * fr_sn_acc
			if (Tmean[i] >= Tsnow) :
				count_thaw = count_thaw + 1
				if (count_thaw > len(snow_melt_coeff)) :
					snow_depl[i] = SnowAcc[i]
				else :
					snow_depl[i] = snow_depl[i] + SnowAcc[i] * snow_melt_coeff[count_thaw]
			SnowAcc[i] = SnowAcc[i] - snow_depl[i]

		snowpack_jan = SnowAcc[11]
		Liq_Prec = np.copy(Prec)
		Liq_Prec[Tmean < Tsnow] = Prec[Tmean < Tsnow] * (1 - fr_sn_acc)
		P_minus_ET = Liq_Prec + snow_depl - PET
		if (ii == 0) :
			last_Storage = TAW

		if (P_minus_ET[0] > 0) :
			Storage[0] = last_Storage + P_minus_ET[1]
			if (Storage[0] > TAW) :
				Percola[0] = Storage[0] - TAW
				Storage[0] = TAW
		else :
			if (last_Storage == 0) : last_Storage = 1
			PETvir = (math.log10(TAW) - math.log10(last_Storage))/float(fTAW)
			Storage[0] = TAW * 10**(-(PETvir + P_minus_ET[1]) * fTAW)

		for i in range(1,len(Storage)): 
			if (P_minus_ET[i] > 0) :
				Storage[i] = Storage[i - 1] + P_minus_ET[i]
				if (Storage[i] > TAW) :
					Percola[i] = Storage[i] - TAW
					Storage[i] = TAW
			else :
				if (Storage[i - 1] == 0) : Storage[i - 1] = 1
				PETvir = (math.log10(TAW) - math.log10(Storage[i - 1]))/float(fTAW)
				Storage[i] = TAW * 10**(-(PETvir + P_minus_ET[i]) * fTAW)

		last_Storage = Storage[11]
		delta_sto = np.insert(np.diff(Storage),0,0)
		ETr = (Liq_Prec + snow_depl - delta_sto)

		for i in range(len(ETr)) :
			if (P_minus_ET[i] > 0) :
				ETr[i] = PET[i]

		Def = PET - ETr
		Def[Def < 0] = 0
		Sur = Liq_Prec + snow_depl - ETr - (TAW - Storage)
		Sur[Sur < 0] = 0
		
		for x in range(12) :
			df = df.append({'year':YYYY[ii],
							 'month':x, 
							 'prec':Prec[x],
							 'PET':PET[x],
							 'Storage':Storage[x],
							 'P_minus_ET':P_minus_ET[x],
							 'Def':Def[x],
							 'Sur':Sur[x]},ignore_index=True) 
	
	return df



	
