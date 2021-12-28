#Load needed python routines
import os
import numpy as np
import pandas as pd
import peakutils
from matplotlib import rcParams
from matplotlib import patches
import sys
from matplotlib import pyplot as plt
import xrayutilities as xu
from scipy import misc

#Load 4-id-c functions 
from s4idc_funcs_v2 import *

# bokeh packages
from bokeh.io import output_file,show,output_notebook,push_notebook, curdoc
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource,HoverTool,CategoricalColorMapper, Select, Slider
from bokeh.palettes import Category10
from bokeh.layouts import row,column,gridplot,widgetbox
from bokeh.models.widgets import Tabs,Panel
output_notebook()


#Set Fonts/Plot Style
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams.update({'font.size': 18})
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.top'] = True
rcParams['ytick.right'] = True
rcParams['xtick.major.size'] = 10
rcParams['ytick.major.size'] = 10
rcParams['xtick.minor.size'] = 5
rcParams['ytick.minor.size'] = 5
rcParams['xtick.minor.visible'] = True
rcParams['ytick.minor.visible'] = True

#Choose data file
specpath = "./data/"
specfile = "SSFeb21_mca.mca"

# Get the spec file
sf = specFile(specpath+specfile)
#df = get_specScan(sf, 176)  #use file number

#function for field dependent for Pr and Er
def field_dep_XMCD(field, Erscan,Prscan):
    Erdata = []
    Prdata = []
    for i in range(len(field)):
        Erdata.append(XMCD(sf,Erscan[i]))
        Prdata.append(XMCD(sf,Prscan[i]))
    
    Erdata = np.array(Erdata)
    Prdata = np.array(Prdata)

    #selecting energy, TEYsum, TEYxmcd
    Er_energy = Erdata[:,0,:]
    Er_TEYsum = Erdata[:,3,:]
    Er_TEYxmcd = Erdata[:,4,:]
    
    Pr_energy = Prdata[:,0,:]
    Pr_TEYsum = Prdata[:,3,:]
    Pr_TEYxmcd = Prdata[:,4,:]
    
    #norm_xmcd
    ScEr_TEYsum = np.zeros(Er_TEYsum.shape);
    ScEr_TEYxmcd = np.zeros(Er_TEYsum.shape);
    for i in range(len(field)):
        ScEr_TEYsum[i,:], ScEr_TEYxmcd[i,:] = norm_xmcd(Er_TEYsum[i,:],Er_TEYxmcd[i,:]);
    
    ScPr_TEYsum = np.zeros(Pr_TEYsum.shape);
    ScPr_TEYxmcd = np.zeros(Pr_TEYsum.shape);
    for i in range(len(field)):
        ScPr_TEYsum[i,:], ScPr_TEYxmcd[i,:] = norm_xmcd(Pr_TEYsum[i,:],Pr_TEYxmcd[i,:]);
    
    #Correct XMCD zero
    for i in range(len(field)):
        ScEr_TEYxmcd[i,:] -=  np.average(ScEr_TEYxmcd[i,-5:-1]);
        ScPr_TEYxmcd[i,:] -=  np.average(ScPr_TEYxmcd[i,-5:-1]);

    #ploting
    col = ["red", "blue" , "green", 'yellow', 'orange', 'purple', 'cyan', 'white', 'gray', 'navy', 
           'pink', 'olive', 'orchid', 'sienna',"red", "blue" , "green", 'yellow', 'orange', 'purple', 
           'cyan', 'white', 'gray', 'navy', 'pink', 'olive', 'orchid', 'sienna']
    colors=[]
    for i in range(len(field)):
        colors.append(col[i])
        
    #Er_XAS 
    plt.style.use('dark_background')
    fig,ax = plt.subplots(2,2,figsize=(20,20))
    ax[0,0].set_ylabel('Norm XAS')
    ax[0,0].set_xlabel('Energy (eV)')
    ax[0,0].set_title('Er XAS', color='w')
    for i in range(len(field)):
        ax[0,0].plot(Er_energy[i,:],ScEr_TEYsum[i,:],linewidth=3,color=colors[i],label=str(field[i])+' T')
    ax[0,0].legend(loc=2)
    
    #Er XMCD
    ax[0,1].set_ylabel('Norm XMCD')
    ax[0,1].set_xlabel('Energy (eV)')
    for i in range(len(field)):
        ax[0,1].plot(Er_energy[i,:],ScEr_TEYxmcd[i,:],linewidth=3,color=colors[i],label=str(field[i])+' T')
    ax[0,1].legend(loc=0)
    ax[0,1].set_title('Er XMCD', color='w')
    

    #Pr_XAS
    ax[1,0].set_title('Pr_XAS',color='w')
    ax[1,0].set_ylabel('Norm XAS')
    ax[1,0].set_xlabel('Energy')
    for i in range(len(field)):
        ax[1,0].plot(Pr_energy[i,:],ScPr_TEYsum[i,:],linewidth=3, linestyle='-',color=colors[i],label=str(field[i])+' T')
    ax[1,0].legend()
    #Pr_XMCD

    ax[1,1].set_title('Pr_XMCD',color='w')
    ax[1,1].set_ylabel('Norm XMCD')
    ax[1,1].set_xlabel('Energy')
    for i in range(len(field)):
        ax[1,1].plot(Pr_energy[i,:],ScPr_TEYxmcd[i,:],linewidth=3, linestyle='-',color=colors[i],label=str(field[i])+' T')
    ax[1,1].legend()
    
   #XMCD peak selection
    Pk_XMCD_Er = np.zeros(field.shape)
    Pk_XMCD_Pr = np.zeros(field.shape)

    for i in range(len(field)):
        if abs(np.nanmax(ScEr_TEYxmcd[i,:])) > abs(np.nanmin(ScEr_TEYxmcd[i,:])):
            Pk_XMCD_Er[i] = np.nanmax(ScEr_TEYxmcd[i,:])
        else:
            Pk_XMCD_Er[i] = np.nanmin(ScEr_TEYxmcd[i,:])               
    for i in range(len(field)):
        if abs(np.nanmin(ScPr_TEYxmcd[i,:]))>abs(np.nanmax(ScPr_TEYxmcd[i,:])):
            Pk_XMCD_Pr[i] = np.nanmin(ScPr_TEYxmcd[i,:]) 
        else:
            Pk_XMCD_Pr[i] = np.nanmax(ScPr_TEYxmcd[i,:]) 
    #XMCD_Peak Analysis      
    fig,ax = plt.subplots(figsize=(10,10))
    ax.set_title('Peak_XMCD',color='w')
    ax.set_ylabel('XMCD_Peak')
    ax.set_xlabel('Field (T)')
    ax.plot(field,Pk_XMCD_Er,'bo', label='Er',markersize=12)
    ax.plot(field,Pk_XMCD_Pr,'mo',label='Pr',markersize=12)
    ax.axhline(0,color='w')
    ax.legend(frameon=False)
    #name=data10[1:4]
    #return plt.savefig(name)
    plt.show()
    
    
#function for temperature dependent for Pr and Er
def temp_dep_XMCD(temp, Erscan,Prscan):
    Erdata = []
    Prdata = []
    for i in range(len(temp)):
        Erdata.append(XMCD(sf,Erscan[i]))
        Prdata.append(XMCD(sf,Prscan[i]))
    
    Erdata = np.array(Erdata)
    Prdata = np.array(Prdata)
    
    #selecting energy, TEYsum, TEYxmcd
    Er_energy = Erdata[:,0,:]
    Er_TEYsum = Erdata[:,3,:]
    Er_TEYxmcd = Erdata[:,4,:]
    
    Pr_energy = Prdata[:,0,:]
    Pr_TEYsum = Prdata[:,3,:]
    Pr_TEYxmcd = Prdata[:,4,:]
    
    #normalizing
    ScEr_TEYsum = np.zeros(Er_TEYsum.shape);
    ScEr_TEYxmcd = np.zeros(Er_TEYsum.shape);
    for i in range(len(temp)):
        ScEr_TEYsum[i,:], ScEr_TEYxmcd[i,:] = norm_xmcd(Er_TEYsum[i,:],Er_TEYxmcd[i,:]);

    ScPr_TEYsum = np.zeros(Pr_TEYsum.shape);
    ScPr_TEYxmcd = np.zeros(Pr_TEYsum.shape);
    for i in range(len(temp)):
        ScPr_TEYsum[i,:], ScPr_TEYxmcd[i,:] = norm_xmcd(Pr_TEYsum[i,:],Pr_TEYxmcd[i,:]);
    
    #Correct XMCD zero
    for i in range(len(temp)):
        ScEr_TEYxmcd[i,:] -=  np.average(ScEr_TEYxmcd[i,-5:-1]);
        ScPr_TEYxmcd[i,:] -=  np.average(ScPr_TEYxmcd[i,-5:-1]);

    #ploting
    col = ["red", "blue" , "green", 'yellow', 'orange', 'purple', 'cyan', 'white', 'gray', 'navy', 
           'pink', 'olive', 'orchid', 'sienna',"red", "blue" , "green", 'yellow', 'orange', 'purple', 
           'cyan', 'white', 'gray', 'navy', 'pink', 'olive', 'orchid', 'sienna']
    colors=[]
    for i in range(len(temp)):
        colors.append(col[i])
        
       #Er_XAS 
    plt.style.use('dark_background')
    fig,ax = plt.subplots(2,2,figsize=(20,20))
    ax[0,0].set_ylabel('Norm XAS')
    ax[0,0].set_xlabel('Energy (eV)')
    ax[0,0].set_title('Er XAS', color='w')
    for i in range(len(temp)):
        ax[0,0].plot(Er_energy[i,:],ScEr_TEYsum[i,:],linewidth=3,color=colors[i],label=str(temp[i])+' K')
    ax[0,0].legend(loc=2)
    
    #Er XMCD
    ax[0,1].set_ylabel('Norm XMCD')
    ax[0,1].set_xlabel('Energy (eV)')
    for i in range(len(temp)):
        ax[0,1].plot(Er_energy[i,:],ScEr_TEYxmcd[i,:],linewidth=3,color=colors[i],label=str(temp[i])+' K')
    ax[0,1].legend(loc=0)
    ax[0,1].set_title('Er XMCD', color='w')
    
    #Pr_XAS
    ax[1,0].set_title('Pr_XAS',color='w')
    ax[1,0].set_ylabel('Norm XAS')
    ax[1,0].set_xlabel('Energy')
    for i in range(len(temp)):
        ax[1,0].plot(Pr_energy[i,:],ScPr_TEYsum[i,:],linewidth=3, linestyle='-',color=colors[i],label=str(temp[i])+' K')
    ax[1,0].legend()
    #Pr_XMCD

    ax[1,1].set_title('Pr_XMCD',color='w')
    ax[1,1].set_ylabel('Norm XMCD')
    ax[1,1].set_xlabel('Energy')
    for i in range(len(temp)):
        ax[1,1].plot(Pr_energy[i,:],ScPr_TEYxmcd[i,:],linewidth=3, linestyle='-',color=colors[i],label=str(temp[i])+' K')
    ax[1,1].legend()
    
   # XMCD height analysis
    Pk_XMCD_Er = np.zeros(temp.shape)
    Pk_XMCD_Pr = np.zeros(temp.shape)

    for i in range(len(temp)):
        if abs(np.nanmax(ScEr_TEYxmcd[i,:])) > abs(np.nanmin(ScEr_TEYxmcd[i,:])):
            Pk_XMCD_Er[i] = np.nanmax(ScEr_TEYxmcd[i,:])
        else:
            Pk_XMCD_Er[i] = np.nanmin(ScEr_TEYxmcd[i,:])               
    for i in range(len(temp)):
        if abs(np.nanmin(ScPr_TEYxmcd[i,:]))>abs(np.nanmax(ScPr_TEYxmcd[i,:])):
            Pk_XMCD_Pr[i] = np.nanmin(ScPr_TEYxmcd[i,:]) 
        else:
            Pk_XMCD_Pr[i] = np.nanmax(ScPr_TEYxmcd[i,:]) 
            
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_title('XMCD Peak',color='w')
    ax.set_ylabel('XMCD_Peak')
    ax.set_xlabel('Temp (K)')
    ax.plot(temp,Pk_XMCD_Er,'bo', label='Er',markersize=12)
    ax.plot(temp,Pk_XMCD_Pr,'mo',label='Pr',markersize=12)
    ax.axhline(0,color='w')
    ax.legend(frameon=False)
    #name=data10[1:4]
    return plt.savefig('name')
    plt.show()
    
    
#function for + - field dependent for Pr and Er
def field_dep_XMCD_diff(field, Erscan_pT, Erscan_nT, Prscan_pT, Prscan_nT ):
    Erdata_pT = []
    Erdata_nT = []

    Prdata_pT = []
    Prdata_nT = []

    for i in range(len(field)):
        Erdata_pT.append(XMCD(sf,Erscan_pT[i]))
        Erdata_nT.append(XMCD(sf,Erscan_nT[i]))
    
        Prdata_pT.append(XMCD(sf,Prscan_pT[i]))
        Prdata_nT.append(XMCD(sf,Prscan_nT[i]))
    
    Erdata_pT = np.array(Erdata_pT)
    Erdata_nT= np.array(Erdata_nT)

    Prdata_pT = np.array(Prdata_pT)
    Prdata_nT= np.array(Prdata_nT)
    
    #selecting energy, TEYsum, TEYxmcd
    energy_pT_Er = Erdata_pT[:,0,:]
    tey_sum_pT_Er = Erdata_pT[:,3,:]
    xmcd_tey_pT_Er = Erdata_pT[:,4,:]

    energy_pT_Pr = Prdata_pT[:,0,:]
    tey_sum_pT_Pr = Prdata_pT[:,3,:]
    xmcd_tey_pT_Pr = Prdata_pT[:,4,:]

    energy_nT_Er= Erdata_nT[:,0,:]
    tey_sum_nT_Er= Erdata_nT[:,3,:]
    xmcd_tey_nT_Er= Erdata_nT[:,4,:]

    energy_nT_Pr= Prdata_nT[:,0,:]
    tey_sum_nT_Pr= Prdata_nT[:,3,:]
    xmcd_tey_nT_Pr= Prdata_nT[:,4,:]
    
    #norm_xmcd
    sctey_pT_Er= np.zeros(tey_sum_pT_Er.shape)
    scxmcd_tey_pT_Er= np.zeros(tey_sum_pT_Er.shape)

    sctey_pT_Pr= np.zeros(tey_sum_pT_Pr.shape)
    scxmcd_tey_pT_Pr= np.zeros(tey_sum_pT_Pr.shape)

    for i in range(len(field)):
        sctey_pT_Er[i,:], scxmcd_tey_pT_Er[i,:] = norm_xmcd(tey_sum_pT_Er[i,:], xmcd_tey_pT_Er[i,:])
        sctey_pT_Pr[i,:], scxmcd_tey_pT_Pr[i,:] = norm_xmcd(tey_sum_pT_Pr[i,:], xmcd_tey_pT_Pr[i,:])
    
    sctey_nT_Er = np.zeros(tey_sum_nT_Er.shape)
    scxmcd_tey_nT_Er = np.zeros(tey_sum_nT_Er.shape)

    sctey_nT_Pr = np.zeros(tey_sum_nT_Pr.shape)
    scxmcd_tey_nT_Pr = np.zeros(tey_sum_nT_Pr.shape)

    for i in range(len(field)):
        sctey_nT_Er[i,:], scxmcd_tey_nT_Er[i,:] = norm_xmcd(tey_sum_nT_Er[i,:], xmcd_tey_nT_Er[i,:])
        sctey_nT_Pr[i,:], scxmcd_tey_nT_Pr[i,:] = norm_xmcd(tey_sum_nT_Pr[i,:], xmcd_tey_nT_Pr[i,:])
    
    #Correct XMCD zero
    for i in range(len(field)):
        scxmcd_tey_pT_Er[i,:] -=  np.average(scxmcd_tey_pT_Er[i,-5:-1])
        scxmcd_tey_nT_Er[i,:] -=  np.average(scxmcd_tey_nT_Er[i,-5:-1])
    
        scxmcd_tey_pT_Pr[i,:] -=  np.average(scxmcd_tey_pT_Pr[i,-5:-1])
        scxmcd_tey_nT_Pr[i,:] -=  np.average(scxmcd_tey_nT_Pr[i,-5:-1])
    
    #Plotting
    col = ["red", "blue" , "green", 'yellow', 'orange', 'purple', 'cyan', 'white', 
           'gray', 'navy', 'pink', 'olive', 'orchid', 'sienna']
    colors=[]
    for i in range(len(field)):
        colors.append(col[i])
        
    #Look at backgrounds and subtracted data Er
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
    ax1.set_ylabel('Norm XAS')
    ax1.set_xlabel('Energy (eV)')
    for i in range(len(field)):
        ax1.plot(energy_pT_Er[i,:],sctey_pT_Er[i,:],linewidth=3, color=colors[i],label=str(field[i])+' T')
        ax1.plot(energy_nT_Er[i,:],sctey_nT_Er[i,:],linewidth=3, linestyle='--', color=colors[i],label='-'+str(field[i])+' T')
    ax1.legend()
    ax1.set_title('XAS_Er')

    ax2.set_ylabel('Norm XMCD')
    ax2.set_xlabel('Energy (eV)')
    for i in range(len(field)):
        ax2.plot(energy_pT_Er[i,:],scxmcd_tey_pT_Er[i,:],linewidth=3,color=colors[i],label=str(field[i])+' T')
        ax2.plot(energy_nT_Er[i,:],scxmcd_tey_nT_Er[i,:],linewidth=3,color=colors[i],linestyle='--',label='-'+str(field[i])+' T')
    ax2.legend()
    ax2.set_title('XMCD_Er')
     
    # subtracted data Pr
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
    ax1.set_ylabel('Norm XAS')
    ax1.set_xlabel('Energy (eV)')
    for i in range(len(field)):
        ax1.plot(energy_pT_Pr[i,:],sctey_pT_Pr[i,:],linewidth=3, color=colors[i],label=str(field[i])+' T')
        ax1.plot(energy_nT_Pr[i,:],sctey_nT_Pr[i,:],linewidth=3, linestyle='--', color=colors[i],label='-'+str(field[i])+' T')
    ax1.axhline(0,color='w')
    ax1.legend()
    ax1.set_title('XAS_Pr')


    ax2.set_ylabel('Norm XMCD')
    ax2.set_xlabel('Energy (eV)')
    for i in range(len(field)):
        ax2.plot(energy_pT_Pr[i,:],scxmcd_tey_pT_Pr[i,:],color=colors[i],linewidth=3,label=str(field[i])+' T')
        ax2.plot(energy_nT_Pr[i,:],scxmcd_tey_nT_Pr[i,:],color=colors[i],linewidth=3,linestyle='--',label='-'+str(field[i])+' T')
    ax2.legend(loc=2)
    ax2.set_title('XMCD_Pr')
    
    #XAS, XMCD +ve and _ve diff calculation
    sctey_sum_Er= np.zeros(sctey_pT_Er.shape)
    scxmcd_tey_dif_Er= np.zeros(scxmcd_tey_pT_Er.shape)

    sctey_sum_Pr= np.zeros(sctey_pT_Pr.shape)
    scxmcd_tey_dif_Pr= np.zeros(scxmcd_tey_pT_Pr.shape)

    for i in range(len(field)):
        sctey_sum_Er[i,:]=(sctey_pT_Er[i,:]+np.interp(energy_pT_Er[i,:],energy_nT_Er[i,:],sctey_nT_Er[i,:]))/2
        scxmcd_tey_dif_Er[i,:] = (scxmcd_tey_pT_Er[i,:] - np.interp(energy_pT_Er[i,:],energy_nT_Er[i,:],scxmcd_tey_nT_Er[i,:]))/2
        sctey_sum_Pr[i,:]=(sctey_pT_Pr[i,:]+np.interp(energy_pT_Pr[i,:],energy_nT_Pr[i,:],sctey_nT_Pr[i,:]))/2
        scxmcd_tey_dif_Pr[i,:] = (scxmcd_tey_pT_Pr[i,:] - np.interp(energy_pT_Pr[i,:],energy_nT_Pr[i,:],scxmcd_tey_nT_Pr[i,:]))/2
    #scxmcd_tey_dif +=.0003
    
    #XAS +ve and _ve diff
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
    ax1.set_ylabel('XMCD_diff')
    ax1.set_xlabel('Energy (eV)')
    ax1.axhline(0,color='k')
    for i in range(len(field)):
        ax1.plot(energy_pT_Er[i,:],scxmcd_tey_dif_Er[i,:],color=colors[i],linewidth=3,label=str(field[i])+' T')
    ax1.legend()
    ax1.set_title('XMCD_diff_Er')

    #XMCD +ve and _ve diff
    ax2.set_ylabel('XMCD_diff')
    ax2.set_xlabel('Energy (eV)')
    ax2.axhline(0,color='k')
    for i in range(len(field)):
        ax2.plot(energy_pT_Pr[i,:],scxmcd_tey_dif_Pr[i,:],color=colors[i],linewidth=3,label=str(field[i])+' T')
    ax2.legend()
    ax2.set_title('XMCD_diff_Pr')
    
    #XMCD peak analysis
    Pk_scxmcd_dif_Er= np.zeros(field.shape)
    Pk_scxmcd_dif_Pr= np.zeros(field.shape)

    for i in range(len(field)):
        if abs(np.nanmax(scxmcd_tey_dif_Er[i,:])) >  abs(np.nanmin(scxmcd_tey_dif_Er[i,:])):
            Pk_scxmcd_dif_Er[i] = np.nanmax(scxmcd_tey_dif_Er[i,:])
        else:
            Pk_scxmcd_dif_Er[i] = np.nanmin(scxmcd_tey_dif_Er[i,:])
                 
    for i in range(len(field)):
        if abs(np.nanmax(scxmcd_tey_dif_Pr[i,:]))>abs(np.nanmin(scxmcd_tey_dif_Pr[i,:])):             
            Pk_scxmcd_dif_Pr[i] = np.nanmax(scxmcd_tey_dif_Pr[i,:])
        else:
            Pk_scxmcd_dif_Pr[i] = np.nanmin(scxmcd_tey_dif_Pr[i,:])
    #ploating peak       
    fig, ax1 = plt.subplots( figsize=(10,10))
    ax1.set_ylabel('diff_XMCD Peak')
    ax1.set_xlabel('Field (T)')
    ax1.plot(field,Pk_scxmcd_dif_Er, 'bo', label='Diff_Er_M5', markersize=15);
    ax1.plot(field,Pk_scxmcd_dif_Pr, 'mo', label='Diff_Pr_M5', markersize=15);
    ax1.axhline(0,color='w')
    ax1.legend(frameon=False)
    ax1.set_title('XMCD_diff_peak')
    plt.show()
    

#function for + - temp dependent for Pr and Er
def temp_dep_XMCD_diff(temp, Erscan_pT, Erscan_nT, Prscan_pT, Prscan_nT ):
    Erdata_pT = []
    Erdata_nT = []

    Prdata_pT = []
    Prdata_nT = []

    for i in range(len(temp)):
        Erdata_pT.append(XMCD(sf,Erscan_pT[i]))
        Erdata_nT.append(XMCD(sf,Erscan_nT[i]))
    
        Prdata_pT.append(XMCD(sf,Prscan_pT[i]))
        Prdata_nT.append(XMCD(sf,Prscan_nT[i]))
    
    Erdata_pT = np.array(Erdata_pT)
    Erdata_nT= np.array(Erdata_nT)

    Prdata_pT = np.array(Prdata_pT)
    Prdata_nT= np.array(Prdata_nT)

    energy_pT_Er = Erdata_pT[:,0,:]
    tey_sum_pT_Er = Erdata_pT[:,3,:]
    xmcd_tey_pT_Er = Erdata_pT[:,4,:]

    energy_pT_Pr = Prdata_pT[:,0,:]
    tey_sum_pT_Pr = Prdata_pT[:,3,:]
    xmcd_tey_pT_Pr = Prdata_pT[:,4,:]

    energy_nT_Er= Erdata_nT[:,0,:]
    tey_sum_nT_Er= Erdata_nT[:,3,:]
    xmcd_tey_nT_Er= Erdata_nT[:,4,:]

    energy_nT_Pr= Prdata_nT[:,0,:]
    tey_sum_nT_Pr= Prdata_nT[:,3,:]
    xmcd_tey_nT_Pr= Prdata_nT[:,4,:]
    
    sctey_pT_Er= np.zeros(tey_sum_pT_Er.shape)
    scxmcd_tey_pT_Er= np.zeros(tey_sum_pT_Er.shape)

    sctey_pT_Pr= np.zeros(tey_sum_pT_Pr.shape)
    scxmcd_tey_pT_Pr= np.zeros(tey_sum_pT_Pr.shape)

    for i in range(len(temp)):
        sctey_pT_Er[i,:], scxmcd_tey_pT_Er[i,:] = norm_xmcd(tey_sum_pT_Er[i,:], xmcd_tey_pT_Er[i,:])
        sctey_pT_Pr[i,:], scxmcd_tey_pT_Pr[i,:] = norm_xmcd(tey_sum_pT_Pr[i,:], xmcd_tey_pT_Pr[i,:])
    
    sctey_nT_Er = np.zeros(tey_sum_nT_Er.shape)
    scxmcd_tey_nT_Er = np.zeros(tey_sum_nT_Er.shape)

    sctey_nT_Pr = np.zeros(tey_sum_nT_Pr.shape)
    scxmcd_tey_nT_Pr = np.zeros(tey_sum_nT_Pr.shape)

    for i in range(len(temp)):
        sctey_nT_Er[i,:], scxmcd_tey_nT_Er[i,:] = norm_xmcd(tey_sum_nT_Er[i,:], xmcd_tey_nT_Er[i,:])
        sctey_nT_Pr[i,:], scxmcd_tey_nT_Pr[i,:] = norm_xmcd(tey_sum_nT_Pr[i,:], xmcd_tey_nT_Pr[i,:])
    
    #Correct XMCD zero
    for i in range(len(temp)):
        scxmcd_tey_pT_Er[i,:] -=  np.average(scxmcd_tey_pT_Er[i,-5:-1])
        scxmcd_tey_nT_Er[i,:] -=  np.average(scxmcd_tey_nT_Er[i,-5:-1])
    
        scxmcd_tey_pT_Pr[i,:] -=  np.average(scxmcd_tey_pT_Pr[i,-5:-1])
        scxmcd_tey_nT_Pr[i,:] -=  np.average(scxmcd_tey_nT_Pr[i,-5:-1])
    
    col = ["red", "blue" , "green", 'yellow', 'orange', 'purple', 'cyan', 'white', 
           'gray', 'navy', 'pink', 'olive', 'orchid', 'sienna']
    colors=[]
    for i in range(len(temp)):
        colors.append(col[i])
    
    #ErXAS
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))

    ax1.set_ylabel('Norm XAS')
    ax1.set_xlabel('Energy (eV)')
    for i in range(len(temp)):
        ax1.plot(energy_pT_Er[i,:],sctey_pT_Er[i,:],linewidth=3, color=colors[i],label=str(temp[i])+' K')
        ax1.plot(energy_nT_Er[i,:],sctey_nT_Er[i,:],linewidth=3, linestyle='--', color=colors[i],label='-'+str(temp[i])+' K')
    ax1.legend()
    ax1.set_title('XAS_Er')

    #Er XMCD
    ax2.set_ylabel('Norm XMCD')
    ax2.set_xlabel('Energy (eV)')
    for i in range(len(temp)):
        ax2.plot(energy_pT_Er[i,:],scxmcd_tey_pT_Er[i,:],linewidth=3,color=colors[i],label=str(temp[i])+' K')
        ax2.plot(energy_nT_Er[i,:],scxmcd_tey_nT_Er[i,:],linewidth=3,color=colors[i],linestyle='--',label='-'+str(temp[i])+' K')
    ax2.legend()
    ax2.set_title('XMCD_Er')
     
    #  Pr XAS
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
    ax1.set_ylabel('Norm XAS')
    ax1.set_xlabel('Energy (eV)')
    for i in range(len(temp)):
        ax1.plot(energy_pT_Pr[i,:],sctey_pT_Pr[i,:],linewidth=3, color=colors[i],label=str(temp[i])+' K')
        ax1.plot(energy_nT_Pr[i,:],sctey_nT_Pr[i,:],linewidth=3, linestyle='--', color=colors[i],label='-'+str(temp[i])+' K')
    ax1.axhline(0,color='w')
    ax1.legend()
    ax1.set_title('XAS_Pr')

    #PrXMCD
    ax2.set_ylabel('Norm XMCD')
    ax2.set_xlabel('Energy (eV)')
    for i in range(len(temp)):
        ax2.plot(energy_pT_Pr[i,:],scxmcd_tey_pT_Pr[i,:],linewidth=3,color=colors[i],label=str(temp[i])+' K')
        ax2.plot(energy_nT_Pr[i,:],scxmcd_tey_nT_Pr[i,:],linewidth=3,color=colors[i],linestyle='--',label='-'+str(temp[i])+' K')
    ax2.legend(loc=2)
    ax2.set_title('XMCD_Pr')
    
    #XAS, XMCD +ve and _ve diff
    sctey_sum_Er= np.zeros(sctey_pT_Er.shape)
    scxmcd_tey_dif_Er= np.zeros(scxmcd_tey_pT_Er.shape)

    sctey_sum_Pr= np.zeros(sctey_pT_Pr.shape)
    scxmcd_tey_dif_Pr= np.zeros(scxmcd_tey_pT_Pr.shape)

    for i in range(len(temp)):
        sctey_sum_Er[i,:]=(sctey_pT_Er[i,:]+np.interp(energy_pT_Er[i,:],energy_nT_Er[i,:],sctey_nT_Er[i,:]))/2
        scxmcd_tey_dif_Er[i,:] = (scxmcd_tey_pT_Er[i,:] - np.interp(energy_pT_Er[i,:],energy_nT_Er[i,:],scxmcd_tey_nT_Er[i,:]))/2
        sctey_sum_Pr[i,:]=(sctey_pT_Pr[i,:]+np.interp(energy_pT_Pr[i,:],energy_nT_Pr[i,:],sctey_nT_Pr[i,:]))/2
        scxmcd_tey_dif_Pr[i,:] = (scxmcd_tey_pT_Pr[i,:] - np.interp(energy_pT_Pr[i,:],energy_nT_Pr[i,:],scxmcd_tey_nT_Pr[i,:]))/2
    #scxmcd_tey_dif +=.0003
    
    #XAS +ve and -ve diff
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
    ax1.set_ylabel('XMCD_diff')
    ax1.set_xlabel('Energy (eV)')
    ax1.axhline(0,color='k')
    for i in range(len(temp)):
        ax1.plot(energy_pT_Er[i,:],scxmcd_tey_dif_Er[i,:],linewidth=3,color=colors[i],label=str(temp[i])+' K')
    ax1.legend()
    ax1.set_title('XMCD_diff_Er')

    #XMCD +ve and -ve diff
    ax2.set_ylabel('XMCD_diff')
    ax2.set_xlabel('Energy (eV)')
    ax2.axhline(0,color='k')
    for i in range(len(temp)):
        ax2.plot(energy_pT_Pr[i,:],scxmcd_tey_dif_Pr[i,:],linewidth=3,color=colors[i],label=str(temp[i])+' K')
    ax2.legend()
    ax2.set_title('XMCD_diff_Pr')
    
    #Simple XMCD height analysis
    Pk_scxmcd_dif_Er= np.zeros(temp.shape)
    Pk_scxmcd_dif_Pr= np.zeros(temp.shape)

    for i in range(len(temp)):
        if abs(np.nanmax(scxmcd_tey_dif_Er[i,:])) >  abs(np.nanmin(scxmcd_tey_dif_Er[i,:])):
            Pk_scxmcd_dif_Er[i] = np.nanmax(scxmcd_tey_dif_Er[i,:])
        else:
            Pk_scxmcd_dif_Er[i] = np.nanmin(scxmcd_tey_dif_Er[i,:])
                 
    for i in range(len(temp)):
        if abs(np.nanmax(scxmcd_tey_dif_Pr[i,:]))>abs(np.nanmin(scxmcd_tey_dif_Pr[i,:])):             
            Pk_scxmcd_dif_Pr[i] = np.nanmax(scxmcd_tey_dif_Pr[i,:])
        else:
            Pk_scxmcd_dif_Pr[i] = np.nanmin(scxmcd_tey_dif_Pr[i,:])
            
    fig, ax1 = plt.subplots(figsize=(10,10))
    ax1.set_ylabel('diff_XMCD Peak')
    ax1.set_xlabel('Field (T)')
    ax1.plot(temp,Pk_scxmcd_dif_Er, 'bo', label='Diff_Er_M5', markersize=15);
    ax1.plot(temp,Pk_scxmcd_dif_Pr, 'mo', label='Diff_Pr_M5', markersize=15);
    ax1.axhline(0,color='w')
    ax1.legend(frameon=False)
    ax1.set_title('XMCD_diff_peak')
    plt.show()
    
    

def ErPr_XMCD_diff( Erscan_pT, Erscan_nT, Prscan_pT, Prscan_nT ):
    energy_p5T_Er, _, _, tey_sum_p5T_Er, xmcd_tey_p5T_Er, tfy_sum_p5T_Er, xmcd_tfy_p5T_Er,_ = XMCD(sf,Erscan_pT)  
    energy_n5T_Er, _, _, tey_sum_n5T_Er, xmcd_tey_n5T_Er, tfy_sum_n5T_Er, xmcd_tfy_n5T_Er,_ = XMCD(sf,Erscan_nT)
    #energy_n5T_Er-=.15

    #reverse order of points
    energy_p5T_Er = energy_p5T_Er[::-1]
    energy_n5T_Er = energy_n5T_Er[::-1]
    tey_sum_p5T_Er = tey_sum_p5T_Er[::-1]
    tey_sum_n5T_Er = tey_sum_n5T_Er[::-1]
    xmcd_tey_p5T_Er = xmcd_tey_p5T_Er[::-1]
    xmcd_tey_n5T_Er = xmcd_tey_n5T_Er[::-1]


    energy_p5T_Pr, _, _, tey_sum_p5T_Pr, xmcd_tey_p5T_Pr, tfy_sum_p5T_Pr, xmcd_tfy_p5T_Pr,_ = XMCD(sf,Prscan_pT)
    energy_n5T_Pr, _, _, tey_sum_n5T_Pr, xmcd_tey_n5T_Pr, tfy_sum_n5T_Pr, xmcd_tfy_n5T_Pr,_ = XMCD(sf,Prscan_nT)
    #energy_n5T_Pr+=.1

    #reverse order of points
    energy_p5T_Pr = energy_p5T_Pr[::-1]
    energy_n5T_Pr = energy_n5T_Pr[::-1]
    tey_sum_p5T_Pr = tey_sum_p5T_Pr[::-1]
    tey_sum_n5T_Pr = tey_sum_n5T_Pr[::-1]
    xmcd_tey_p5T_Pr = xmcd_tey_p5T_Pr[::-1]
    xmcd_tey_n5T_Pr = xmcd_tey_n5T_Pr[::-1]
    
    #Fix energy shifts
    peak_pos_Er_p5T = (np.where(tey_sum_p5T_Er==np.nanmax(tey_sum_p5T_Er)))
    peak_pos_Er_n5T = (np.where(tey_sum_n5T_Er==np.nanmax(tey_sum_n5T_Er)))
    dif_Eng_Er_5T = energy_p5T_Er[peak_pos_Er_p5T] - energy_n5T_Er[peak_pos_Er_n5T]
    energy_n5T_Er += dif_Eng_Er_5T

    peak_pos_Pr_p5T = (np.where(tey_sum_p5T_Pr==np.nanmax(tey_sum_p5T_Pr)))
    peak_pos_Pr_n5T = (np.where(tey_sum_n5T_Pr==np.nanmax(tey_sum_n5T_Pr)))
    dif_Eng_Pr_5T = energy_p5T_Pr[peak_pos_Pr_p5T] - energy_n5T_Pr[peak_pos_Pr_n5T]
    energy_n5T_Pr += dif_Eng_Pr_5T
    
    #norm xmcd Er
    sctey_p5T_Er, scxmcd_tey_p5T_Er = norm_xmcd(tey_sum_p5T_Er, xmcd_tey_p5T_Er)
    sctey_n5T_Er, scxmcd_tey_n5T_Er = norm_xmcd(tey_sum_n5T_Er, xmcd_tey_n5T_Er)
    sctfy_p5T_Er, scxmcd_tfy_p5T_Er = norm_xmcd(tfy_sum_p5T_Er, xmcd_tfy_p5T_Er)
    sctfy_n5T_Er, scxmcd_tfy_n5T_Er = norm_xmcd(tfy_sum_n5T_Er, xmcd_tfy_n5T_Er)

    #norm xmcd Pr
    sctey_p5T_Pr, scxmcd_tey_p5T_Pr = norm_xmcd(tey_sum_p5T_Pr, xmcd_tey_p5T_Pr)
    sctey_n5T_Pr, scxmcd_tey_n5T_Pr = norm_xmcd(tey_sum_n5T_Pr, xmcd_tey_n5T_Pr)
    sctfy_p5T_Pr, scxmcd_tfy_p5T_Pr = norm_xmcd(tfy_sum_p5T_Pr, xmcd_tfy_p5T_Pr)
    sctfy_n5T_Pr, scxmcd_tfy_n5T_Pr = norm_xmcd(tfy_sum_n5T_Pr, xmcd_tfy_n5T_Pr)
    

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
 
    ax1.set_ylabel('Norm XAS')
    ax1.set_xlabel('Energy (eV)')
    ax1.plot(energy_p5T_Er,sctey_p5T_Er,'b',linewidth=3,label='p_xas_Er')
    ax1.plot(energy_n5T_Er,sctey_n5T_Er,'r',linewidth=3,label='n_xas_Er')
    ax1.legend()
    ax1.set_title('XAS_Er')

    
    ax2.set_ylabel('Norm XMCD')
    ax2.set_xlabel('Energy (eV)')
    ax2.plot(energy_p5T_Er,scxmcd_tey_p5T_Er,'m',linewidth=3,label='p_xmcd_Er')
    ax2.plot(energy_n5T_Er,scxmcd_tey_n5T_Er,'g',linewidth=3,label='n_xmcd_Er')
    ax2.legend()
    ax2.set_title('XMCD_Er')
    

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
    ax1.set_ylabel('Norm XAS')
    ax1.set_xlabel('Energy (eV)')
    ax1.plot(energy_p5T_Pr,sctey_p5T_Pr,'b',linewidth=3,label='p_xas_Pr')
    ax1.plot(energy_n5T_Pr,sctey_n5T_Pr,'r',linewidth=3,label='n_xas_Pr')
    ax1.legend()
    ax1.set_title('XAS_Pr')

    ax2.set_ylabel('Norm XMCD')
    ax2.set_xlabel('Energy (eV)')
    ax2.plot(energy_p5T_Pr,scxmcd_tey_p5T_Pr,'m',linewidth=3,label='p_xmcd_Pr')
    ax2.plot(energy_n5T_Pr,scxmcd_tey_n5T_Pr,'g',linewidth=3,label='n_xmcd_Pr')
    ax2.legend()
    ax2.set_title('XMCD_Pr')
    
    sctey_sum_Er=(sctey_p5T_Er+np.interp(energy_p5T_Er,energy_n5T_Er,sctey_n5T_Er))/2
    scxmcd_tey_dif_Er = (scxmcd_tey_p5T_Er - np.interp(energy_p5T_Er,energy_n5T_Er,scxmcd_tey_n5T_Er))/2
    sctfy_sum_Er=(sctfy_p5T_Er+np.interp(energy_p5T_Er,energy_n5T_Er,sctfy_n5T_Er))/2
    scxmcd_tfy_dif_Er = (scxmcd_tfy_p5T_Er - np.interp(energy_p5T_Er,energy_n5T_Er,scxmcd_tfy_n5T_Er))/2
    #scxmcd_tey_dif +=.0003

    sctey_sum_Pr=(sctey_p5T_Pr+np.interp(energy_p5T_Pr,energy_n5T_Pr,sctey_n5T_Pr))/2
    scxmcd_tey_dif_Pr = (scxmcd_tey_p5T_Pr - np.interp(energy_p5T_Pr,energy_n5T_Pr,scxmcd_tey_n5T_Pr))/2
    sctfy_sum_Pr=(sctfy_p5T_Pr+np.interp(energy_p5T_Pr,energy_n5T_Pr,sctfy_n5T_Pr))/2
    scxmcd_tfy_dif_Pr = (scxmcd_tfy_p5T_Pr - np.interp(energy_p5T_Pr,energy_n5T_Pr,scxmcd_tfy_n5T_Pr))/2
    scxmcd_tey_dif_Pr +=.00015
    

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
    ax1.set_ylabel('XMCD')
    ax1.set_xlabel('Energy (eV)')
    ax1.axhline(0,color='k')
    ax1.plot(energy_p5T_Er,scxmcd_tey_dif_Er,'g',label='dif_Er', linewidth=3, linestyle='-',color='r')
    ax1.legend()
    ax1.set_title('XMCD_diff_Er')
    
    
    ax2.set_ylabel('XMCD')
    ax2.set_xlabel('Energy (eV)')
    ax2.axhline(0,color='k')
    ax2.plot(energy_p5T_Pr,scxmcd_tey_dif_Pr,'g',linewidth=3, label='dif_Pr')
    ax2.legend()
    ax2.set_title('XMCD_diff_Pr')
    plt.show()
    
def ErPr_XMCD( Erscan, Prscan ):
    energy_Er, ref_sum_Er, xmcd_ref_Er, tey_sum_Er, xmcd_tey_Er, tfy_sum_Er, xmcd_tfy_Er,std_Er = XMCD(sf,Erscan)
    energy_Pr, ref_sum_Pr, xmcd_ref_Pr, tey_sum_Pr, xmcd_tey_Pr, tfy_sum_Pr, xmcd_tfy_Pr,std_Pr = XMCD(sf,Prscan)

    #reverse order of points
    energy_Er = energy_Er[::-1]
    tey_sum_Er = tey_sum_Er[::-1]
    xmcd_tey_Er = xmcd_tey_Er[::-1]
    ref_sum_Er = ref_sum_Er[::-1]
    xmcd_ref_Er = xmcd_ref_Er[::-1]
    std_Er = std_Er[::-1]

    energy_Pr = energy_Pr[::-1]
    tey_sum_Pr = tey_sum_Pr[::-1]
    xmcd_tey_Pr = xmcd_tey_Pr[::-1]
    ref_sum_Pr = ref_sum_Pr[::-1]
    xmcd_ref_Pr = xmcd_ref_Pr[::-1]
    std_Pr = std_Pr[::-1]
    
    sctey_Er, scxmcd_tey_Er = norm_xmcd(tey_sum_Er, xmcd_tey_Er);
    sctey_Pr, scxmcd_tey_Pr = norm_xmcd(tey_sum_Pr, xmcd_tey_Pr);
    
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))    

    ax1.set_ylabel('XAS')
    ax1.set_xlabel('Energy (eV)')
    ax1.plot(energy_Er,sctey_Er,'b',linewidth=3,label='Er')
    ax1.legend()
    ax1.set_title('XAS_Er')

    ax2.set_ylabel('XMCD')
    ax2.set_xlabel('Energy (eV)')
    ax2.plot(energy_Er,scxmcd_tey_Er,'m',linewidth=3,label='Er')
    ax2.legend()
    ax2.set_title('XMCD_Er')

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
    ax1.set_ylabel('XAS')
    ax1.set_xlabel('Energy (eV)')
    ax1.plot(energy_Pr,sctey_Pr,'b',linewidth=3,label='Pr')
    ax1.legend()
    ax1.set_title('XAS_Pr')

    ax2.set_ylabel('XMCD')
    ax1.set_xlabel('Energy (eV)')
    ax2.plot(energy_Pr,scxmcd_tey_Pr,'m',linewidth=3,label='Pr')
    ax2.legend()
    ax2.set_title('XMCD_Pr')
    plt.show()