#Functions for reading and analyzing 4-ID-C data
# J.W. Freeland(APS/ANL)  Updated 10/9/20
# Thanks to Yue Cao (ANL) for the functions for reading spec files

import os
import numpy as np
import pandas as pd
# for reading spec files
import PyMca5.PyMcaCore.SpecFileDataSource as SpecFileDataSource
# for reading mda files
from mda import readMDA

#Functions for reading spec files

# In PyMca, the scans are labeled as '15.1' instead of '15'.
def ID_to_str(ID):
    try:
        ID = int(ID)
        return str(ID)+'.1'
    except ValueError:
        print ("Error: Invalid scan ID format.")
        return 'nan'

# get spec file.
class specFile():
    def __init__(self, name, path=''):
        if path=='':
            path = os.getcwd()
        else:
            path = path
        filePath = os.path.abspath(os.path.join(path, name))
        
        specFile = SpecFileDataSource.SpecFileDataSource(filePath)
        
        list_scanID_txt = specFile.getSourceInfo()['KeyList']
        list_scanLen = specFile.getSourceInfo()['NumPts']
        list_scanID = []
        for txt in list_scanID_txt:
            num = int(txt[:-2])
            list_scanID.append(num)
            
        self.path = filePath
        self.spec = specFile
        self.scanList = list_scanID
        self.scanLen = list_scanLen
        return


def get_specScan(sf, ID):
    str_ID = ID_to_str(ID)
    scan = sf.spec.getDataObject(str_ID)
    scanKeys = scan.getInfo()['LabelNames']
    scanData = scan.getData()
    scanData = pd.DataFrame(scanData, columns=scanKeys)
    return scanData

#Reader for MCA files
def readmca(sf,scan,mcafilepath,scan_offset=0,kb=0):
    files = sorted(os.listdir(mcafilepath))
    #print(files)
    if (kb==0):
        i0 = np.array(get_specScan(sf, scan-scan_offset)['I0'])
    else:
        i0 = np.array(get_specScan(sf, scan-scan_offset)['4idc1:scaler2.S8'])
    mcadata = []
    for i,file in enumerate(files):
        data=readMDA(mcafilepath+file)
        mcadata.append(data[1].d[0].data/i0[i])

    return np.array(mcadata)

def readxmcdmca(sf,scan,mcafilepath):
    files = sorted(os.listdir(mcafilepath))
    files_A=files[::2]
    files_B=files[1::2]
    i0_A = np.array(get_specScan(sf, scan)['i0_A'])
    i0_B = np.array(get_specScan(sf, scan)['i0_B'])
    mcadata_A = []
    mcadata_B = []
    for i,file in enumerate(files_A):
        data=readMDA(mcafilepath+file)
        mcadata_A.append(data[1].d[0].data/i0_A[i])
    for i,file in enumerate(files_B):
        data=readMDA(mcafilepath+file)
        mcadata_B.append(data[1].d[0].data/i0_B[i])
    
    return np.array(mcadata_A), np.array(mcadata_B)

def readscanmda(filename):
    data=readMDA(filename)
    x = np.array(data[1].p[0].data)
    tey = np.array(data[1].d[8].data)
    tfy = np.array(data[1].d[9].data)
    ref = np.array(data[1].d[7].data)
    std = np.array(data[1].d[10].data)
    
    return x, tey, tfy, ref, std
    
def readscanmda29(filename):
    data=readMDA(filename)
    x = np.array(data[1].p[0].data)
    tey = np.array(data[1].d[13].data)
    i0 = np.array(data[1].d[14].data)

    return x, tey, i0
    
# Functions for 4-ID-C data analysis reading spec files

def XAS(sf,scan):
    
    df = get_specScan(sf, scan)
    energy = np.array(df['SGM1:Energy'])
    ref = np.array(df['reflectivity/I0'])
    tey = np.array(df['TEY/I0'])
    tfy = np.array(df['FY/I0'])
    std = np.array(df['reference/I0'])
    
    return energy, ref, tey, tfy, std

def XASkbi0(sf,scan):
    
    df = get_specScan(sf, scan)
    energy = np.array(df['SGM1:Energy'])
    i0 = np.array(df['4idc1:scaler2.S8'])
    refr = np.array(df['Laser'])
    teyr = np.array(df['4idc1:scaler1.S4'])
    tfyr = np.array(df['Vortex'])
    stdr = np.array(df['4idc1:scaler1.S6'])
    ref=refr/i0
    tey=teyr/i0
    tfy=tfyr/i0
    std=stdr/i0
    
    return energy, ref, tey, tfy, std

def XASi0(sf,scan):
    
    df = get_specScan(sf, scan)
    i0 = np.array(df['I0'])
    return i0

def XASsum(sf,st_scan, end_scan):
    df = get_specScan(sf, st_scan)
    energy = df['SGM1:Energy']
    ref = df['reflectivity/I0']
    tey = df['TEY/I0']
    tfy = df['FY/I0']
    std = df['reference/I0']

    num_avg=1
    for i in range(st_scan+1,end_scan+1):
        df = get_specScan(sf, i)
        energy = np.add(energy,df['SGM1:Energy'])
        ref = np.add(ref,df['reflectivity/I0'])
        tey = np.add(tey,df['TEY/I0'])
        tfy = np.add(tfy,df['FY/I0'])
        std = np.add(std,df['reference/I0'])
        num_avg+=1
    
    energy=np.array(energy/num_avg)
    ref=np.array(ref/num_avg)
    tey=np.array(tey/num_avg)
    tfy=np.array(tfy/num_avg)
    std=np.array(std/num_avg)
    
    return energy, ref, tey, tfy, std

def XASkbi0sum(sf,st_scan, end_scan):
    df = get_specScan(sf, st_scan)
    energy = df['SGM1:Energy']
    ref = np.array(df['Laser'])/np.array(df['4idc1:scaler2.S8'])
    tey = np.array(df['4idc1:scaler1.S4'])/np.array(df['4idc1:scaler2.S8'])
    tfy = np.array(df['Vortex'])/np.array(df['4idc1:scaler2.S8'])
    std = np.array(df['4idc1:scaler1.S6'])/np.array(df['4idc1:scaler2.S8'])

    num_avg=1
    for i in range(st_scan+1,end_scan+1):
        df = get_specScan(sf, i)
        energy = np.add(energy,df['SGM1:Energy'])
        ref = np.add(ref,np.array(df['Laser'])/np.array(df['4idc1:scaler2.S8']))
        tey = np.add(tey,np.array(df['4idc1:scaler1.S4'])/np.array(df['4idc1:scaler2.S8']))
        tfy = np.add(tfy,np.array(df['Vortex'])/np.array(df['4idc1:scaler2.S8']))
        std = np.add(std,np.array(df['4idc1:scaler1.S6'])/np.array(df['4idc1:scaler2.S8']))
        num_avg+=1
    
    energy=np.array(energy/num_avg)
    ref=np.array(ref/num_avg)
    tey=np.array(tey/num_avg)
    tfy=np.array(tfy/num_avg)
    std=np.array(std/num_avg)
    
    return energy, ref, tey, tfy, std

def XASi0sum(sf,st_scan, end_scan):
    df = get_specScan(sf, st_scan)
    i0 = df['I0']

    for i in range(st_scan+1,end_scan+1):
        df = get_specScan(sf, i)
        i0 = np.average([i0,df['I0']])
    
    return np.array(i0)

def XASmerge(sf,st_scan, end_scan):
    df = get_specScan(sf, st_scan)
    energy = df['SGM1:Energy']
    ref = df['reflectivity/I0']
    tey = df['TEY/I0']
    tfy = df['FY/I0']
    std = df['reference/I0']

    num_avg=1
    for i in range(st_scan+1,end_scan+1):
        df = get_specScan(sf, i)
        energy = np.append(energy,df['SGM1:Energy'])
        ref = np.append(ref,df['reflectivity/I0'])
        tey = np.append(tey,df['TEY/I0'])
        tfy = np.append(tfy,df['FY/I0'])
        std = np.append(std,df['reference/I0'])
    
    return energy, ref, tey, tfy, std

def XASlaser(sf,scan):
    
    df = get_specScan(sf, scan)
    energy = np.array(df['SGM1:Energy'])
    ref = np.array(df['reflectivity/I0'])
    tey = np.array(df['TEY/I0'])
    tfy = np.array(df['FY/I0'])
    std = np.array(df['reference/I0'])
    lon1 = np.array(df['Data_anal'])
    loff1 = np.array(df['Data_anal1'])
    lon2 = np.array(df['Data_anal2'])
    loff2 = np.array(df['Data_anal3'])
    return energy, ref, tey, tfy, std, lon1, loff1,lon2, loff2

def XASlasersum(sf,st_scan,end_scan):
    
    df = get_specScan(sf, st_scan)
    energy = np.array(df['SGM1:Energy'])
    ref = np.array(df['reflectivity/I0'])
    tey = np.array(df['TEY/I0'])
    tfy = np.array(df['FY/I0'])
    std = np.array(df['reference/I0'])
    lon1 = np.array(df['Data_anal'])
    loff1 = np.array(df['Data_anal1'])
    lon2 = np.array(df['Data_anal2'])
    loff2 = np.array(df['Data_anal3'])
    
    num_avg=1
    for i in range(st_scan+1,end_scan+1):
        df = get_specScan(sf, i)
        energy = np.add(energy,df['SGM1:Energy'])
        ref = np.add(ref,df['reflectivity/I0'])
        tey = np.add(tey,df['TEY/I0'])
        tfy = np.add(tfy,df['FY/I0'])
        std = np.add(std,df['reference/I0'])
        lon1 = np.add(lon1,df['Data_anal'])
        loff1 = np.add(loff1,df['Data_anal1'])
        lon2 = np.add(lon2,df['Data_anal2'])
        loff2 = np.add(loff2,df['Data_anal3'])
        num_avg+=1
    
    energy=np.array(energy/num_avg)
    ref=np.array(ref/num_avg)
    tey=np.array(tey/num_avg)
    tfy=np.array(tfy/num_avg)
    std=np.array(std/num_avg)
    lon1=np.array(lon1/num_avg)
    loff1=np.array(loff1/num_avg)
    lon2=np.array(lon2/num_avg)
    loff2=np.array(loff2/num_avg)
    
    return energy, ref, tey, tfy, std, lon1, loff1,lon2, loff2

def XASldel(sf,scan):
    
    df = get_specScan(sf, scan)
    delay = np.array(df['set_time_delay'])
    ref = np.array(df['reflectivity/I0'])
    tey = np.array(df['TEY/I0'])
    tfy = np.array(df['FY/I0'])
    lon1 = np.array(df['Data_anal'])
    loff1 = np.array(df['Data_anal1'])
    lon2 = np.array(df['Data_anal2'])
    loff2 = np.array(df['Data_anal3'])
    
    return delay, ref, tey, tfy, lon1, loff1,lon2, loff2

def XASldelsum(sf,st_scan,end_scan):
    
    df = get_specScan(sf, st_scan)
    delay = np.array(df['set_time_delay'])
    ref = np.array(df['reflectivity/I0'])
    tey = np.array(df['TEY/I0'])
    tfy = np.array(df['FY/I0'])
    lon1 = np.array(df['Data_anal'])
    loff1 = np.array(df['Data_anal1'])
    lon2 = np.array(df['Data_anal2'])
    loff2 = np.array(df['Data_anal3'])
    
    num_avg=1
    for i in range(st_scan+1,end_scan+1):
        df = get_specScan(sf, i)
        delay = np.add(delay,df['set_time_delay'])
        ref = np.add(ref,df['reflectivity/I0'])
        tey = np.add(tey,df['TEY/I0'])
        tfy = np.add(tfy,df['FY/I0'])
        lon1 = np.add(lon1,df['Data_anal'])
        loff1 = np.add(loff1,df['Data_anal1'])
        lon2 = np.add(lon2,df['Data_anal2'])
        loff2 = np.add(loff2,df['Data_anal3'])
        num_avg+=1
    
    delay=np.array(delay/num_avg)
    ref=np.array(ref/num_avg)
    tey=np.array(tey/num_avg)
    tfy=np.array(tfy/num_avg)
    lon1=np.array(lon1/num_avg)
    loff1=np.array(loff1/num_avg)
    lon2=np.array(lon2/num_avg)
    loff2=np.array(loff2/num_avg)
    
    return delay, ref, tey, tfy, lon1, loff1,lon2, loff2

def XASwp(sf,scan):
    
    df = get_specScan(sf, scan)
    wp = np.array(df['Waveplate'])
    ref = np.array(df['reflectivity/I0'])
    tey = np.array(df['TEY/I0'])
    tfy = np.array(df['FY/I0'])
    lon1 = np.array(df['Data_anal'])
    loff1 = np.array(df['Data_anal1'])
    lon2 = np.array(df['Data_anal2'])
    loff2 = np.array(df['Data_anal3'])
    
    return wp, ref, tey, tfy, lon1, loff1,lon2, loff2

def XASwpsum(sf,st_scan,end_scan):
    
    df = get_specScan(sf, st_scan)
    wp = np.array(df['Waveplate'])
    ref = np.array(df['reflectivity/I0'])
    tey = np.array(df['TEY/I0'])
    tfy = np.array(df['FY/I0'])
    lon1 = np.array(df['Data_anal'])
    loff1 = np.array(df['Data_anal1'])
    lon2 = np.array(df['Data_anal2'])
    loff2 = np.array(df['Data_anal3'])
    
    num_avg=1
    for i in range(st_scan+1,end_scan+1):
        df = get_specScan(sf, i)
        wp = np.add(wp,df['Waveplate'])
        ref = np.add(ref,df['reflectivity/I0'])
        tey = np.add(tey,df['TEY/I0'])
        tfy = np.add(tfy,df['FY/I0'])
        lon1 = np.add(lon1,df['Data_anal'])
        loff1 = np.add(loff1,df['Data_anal1'])
        lon2 = np.add(lon2,df['Data_anal2'])
        loff2 = np.add(loff2,df['Data_anal3'])
        num_avg+=1
    
    wp=np.array(wp/num_avg)
    ref=np.array(ref/num_avg)
    tey=np.array(tey/num_avg)
    tfy=np.array(tfy/num_avg)
    lon1=np.array(lon1/num_avg)
    loff1=np.array(loff1/num_avg)
    lon2=np.array(lon2/num_avg)
    loff2=np.array(loff2/num_avg)
    
    return wp, ref, tey, tfy, lon1, loff1,lon2, loff2

def XMCD(sf,scan):
    
    df = get_specScan(sf, scan)
    energy = np.array(df['SGM1:Energy'])
    ref_sum = np.array(df['Sum_reflectivity'])
    xmcd_ref = np.array(df['XMCD_reflectivity'])
    tey_sum = np.array(df['Sum_TEY'])
    xmcd_tey = np.array(df['XMCD_TEY'])
    tfy_sum = np.array(df['Sum_FY'])
    xmcd_tfy = np.array(df['XMCD_FY'])
    std = np.array(df['reference'])
    return energy, ref_sum, xmcd_ref, tey_sum, xmcd_tey, tfy_sum, xmcd_tfy, std


def XMCDsum(sf,st_scan, end_scan):
    df = get_specScan(sf, st_scan)
    energy = df['SGM1:Energy']
    ref_sum = df['Sum_reflectivity']
    xmcd_ref = df['XMCD_reflectivity']
    tey_sum = df['Sum_TEY']
    xmcd_tey = df['XMCD_TEY']
    tfy_sum = df['Sum_FY']
    xmcd_tfy = df['XMCD_FY']
    std = df['reference']

    num_avg=1
    for i in range(st_scan+1,end_scan+1):
        df = get_specScan(sf, i)
        energy = np.add(energy,df['SGM1:Energy'])
        ref_sum = np.add(ref_sum,df['Sum_reflectivity'])
        xmcd_ref = np.add(xmcd_ref,df['XMCD_reflectivity'])
        tey_sum = np.add(tey_sum,df['Sum_TEY'])
        xmcd_tey = np.add(xmcd_tey,df['XMCD_TEY'])
        tfy_sum = np.add(tfy_sum,df['Sum_FY'])
        xmcd_tfy = np.add(xmcd_tfy,df['XMCD_FY'])
        std = np.add(std,df['reference'])
        num_avg+=1
    
    energy=np.array(energy/num_avg)
    ref_sum=np.array(ref_sum/num_avg)
    xmcd_ref=np.array(xmcd_ref/num_avg)
    tey_sum=np.array(tey_sum/num_avg)
    xmcd_tey=np.array(xmcd_tey/num_avg)
    tfy_sum=np.array(tfy_sum/num_avg)
    xmcd_tfy=np.array(xmcd_tfy/num_avg)
    std=np.array(std/num_avg)
    
    return energy, ref_sum, xmcd_ref, tey_sum, xmcd_tey, tfy_sum, xmcd_tfy, std

def XMCDlaser(sf,scan):
    
    df = get_specScan(sf, scan)
    energy = np.array(df['SGM1:Energy'])
    loff_sum = np.array(df['Sum_reflectivity'])
    xmcd_loff = np.array(df['XMCD_reflectivity'])
    tey_sum = np.array(df['Sum_TEY'])
    xmcd_tey = np.array(df['XMCD_TEY'])
    lon_sum = np.array(df['Sum_FY'])
    xmcd_lon = np.array(df['XMCD_FY'])
    std = np.array(df['reference'])
    return energy, loff_sum, xmcd_loff, tey_sum, xmcd_tey, lon_sum, xmcd_lon, std

def XMCDlasersum(sf,st_scan, end_scan):
    df = get_specScan(sf, st_scan)
    energy = df['SGM1:Energy']
    loff_sum = df['Sum_reflectivity']
    xmcd_loff = df['XMCD_reflectivity']
    tey_sum = df['Sum_TEY']
    xmcd_tey = df['XMCD_TEY']
    lon_sum = df['Sum_FY']
    xmcd_lon = df['XMCD_FY']
    std = df['reference']

    num_avg=1
    for i in range(st_scan+1,end_scan+1):
        df = get_specScan(sf, i)
        energy = np.add(energy,df['SGM1:Energy'])
        loff_sum = np.add(loff_sum,df['Sum_reflectivity'])
        xmcd_loff = np.add(xmcd_loff,df['XMCD_reflectivity'])
        tey_sum = np.add(tey_sum,df['Sum_TEY'])
        xmcd_tey = np.add(xmcd_tey,df['XMCD_TEY'])
        lon_sum = np.add(lon_sum,df['Sum_FY'])
        xmcd_lon = np.add(xmcd_lon,df['XMCD_FY'])
        std = np.add(std,df['reference'])
        num_avg+=1
    
    energy=np.array(energy/num_avg)
    loff_sum=np.array(loff_sum/num_avg)
    xmcd_loff=np.array(xmcd_loff/num_avg)
    tey_sum=np.array(tey_sum/num_avg)
    xmcd_tey=np.array(xmcd_tey/num_avg)
    lon_sum=np.array(lon_sum/num_avg)
    xmcd_lon=np.array(xmcd_lon/num_avg)
    std=np.array(std/num_avg)
    
    return energy, loff_sum, xmcd_loff, tey_sum, xmcd_tey, lon_sum, xmcd_lon, std

def XMCDhys(sf,scan):
    
    df = get_specScan(sf, scan)
    if df.columns[0] == 'Hys_Control_Hor':
        field = np.array(df['Hys_Control_Hor'])
    elif df.columns[0] == 'Hys_Control_Hor_Angle':
        field = np.array(df['Hys_Control_Hor_Angle'])
    ref_sum = np.array(df['XMCD_Sum2'])
    xmcd_ref = np.array(df['XMCD_REF'])
    tey_sum = np.array(df['XMCD_Sum'])
    xmcd_tey = np.array(df['XMCD_TEY'])
    tfy_sum = np.array(df['XMCD_Sum1'])
    xmcd_tfy = np.array(df['XMCD_TFY'])
 
    return field, ref_sum, xmcd_ref, tey_sum, xmcd_tey, tfy_sum, xmcd_tfy

def XMCDhyssum(sf,st_scan, end_scan):
    df = get_specScan(sf, st_scan)
    if df.columns[0] == 'Hys_Control_Hor':
        field = np.array(df['Hys_Control_Hor'])
    elif df.columns[0] == 'Hys_Control_Hor_Angle':
        field = np.array(df['Hys_Control_Hor_Angle'])
    ref_sum = df['XMCD_Sum2']
    xmcd_ref = df['XMCD_REF']
    tey_sum = df['XMCD_Sum']
    xmcd_tey = df['XMCD_TEY']
    tfy_sum = df['XMCD_Sum1']
    xmcd_tfy = df['XMCD_TFY']

    num_avg=1
    for i in range(st_scan+1,end_scan+1):
        df = get_specScan(sf, i)
        if df.columns[0] == 'Hys_Control_Hor':
            field = np.add(field,df['Hys_Control_Hor'])
        elif df.columns[0] == 'Hys_Control_Hor_Angle':
            field = np.add(field,df['Hys_Control_Hor_Angle'])
        ref_sum = np.add(ref_sum,df['XMCD_Sum2'])
        xmcd_ref = np.add(xmcd_ref,df['XMCD_REF'])
        tey_sum = np.add(tey_sum,df['XMCD_Sum'])
        xmcd_tey = np.add(xmcd_tey,df['XMCD_TEY'])
        tfy_sum = np.add(tfy_sum,df['XMCD_Sum1'])
        xmcd_tfy = np.add(xmcd_tfy,df['XMCD_TFY'])
        num_avg+=1
    
    field=np.array(field/num_avg)
    ref_sum=np.array(ref_sum/num_avg)
    xmcd_ref=np.array(xmcd_ref/num_avg)
    tey_sum=np.array(tey_sum/num_avg)
    xmcd_tey=np.array(xmcd_tey/num_avg)
    tfy_sum=np.array(tfy_sum/num_avg)
    xmcd_tfy=np.array(xmcd_tfy/num_avg)
    
    return field, ref_sum, xmcd_ref, tey_sum, xmcd_tey, tfy_sum, xmcd_tfy

def XMCDldel(sf,scan):
    
    df = get_specScan(sf, scan)
    delay = np.array(df['set_time_delay'])
    loff_sum = np.array(df['XMCD_Sum2'])
    xmcd_loff = np.array(df['XMCD_REF'])
    tey_sum = np.array(df['XMCD_Sum'])
    xmcd_tey = np.array(df['XMCD_TEY'])
    lon_sum = np.array(df['XMCD_Sum1'])
    xmcd_lon = np.array(df['XMCD_TFY'])
 
    return delay, loff_sum, xmcd_loff, tey_sum, xmcd_tey, lon_sum, xmcd_lon

def XMCDldelsum(sf,st_scan, end_scan):
    df = get_specScan(sf, st_scan)
    delay = np.array(df['set_time_delay'])
    loff_sum = df['XMCD_Sum2']
    xmcd_loff = df['XMCD_REF']
    tey_sum = df['XMCD_Sum']
    xmcd_tey = df['XMCD_TEY']
    lon_sum = df['XMCD_Sum1']
    xmcd_lon = df['XMCD_TFY']

    num_avg=1
    for i in range(st_scan+1,end_scan+1):
        df = get_specScan(sf, i)
        delay = np.add(delay,df['set_time_delay'])
        loff_sum = np.add(loff_sum,df['XMCD_Sum2'])
        xmcd_loff = np.add(xmcd_loff,df['XMCD_REF'])
        tey_sum = np.add(tey_sum,df['XMCD_Sum'])
        xmcd_tey = np.add(xmcd_tey,df['XMCD_TEY'])
        lon_sum = np.add(lon_sum,df['XMCD_Sum1'])
        xmcd_lon = np.add(xmcd_lon,df['XMCD_TFY'])
        num_avg+=1
    
    delay=np.array(delay/num_avg)
    loff_sum=np.array(loff_sum/num_avg)
    xmcd_loff=np.array(xmcd_loff/num_avg)
    tey_sum=np.array(tey_sum/num_avg)
    xmcd_tey=np.array(xmcd_tey/num_avg)
    lon_sum=np.array(lon_sum/num_avg)
    xmcd_lon=np.array(xmcd_lon/num_avg)
    
    return delay, loff_sum, xmcd_loff, tey_sum, xmcd_tey, lon_sum, xmcd_lon

def XMCDlpow(sf,scan):
    
    df = get_specScan(sf, scan)
    wp = np.array(df['Waveplate'])
    loff_sum = np.array(df['XMCD_Sum2'])
    xmcd_loff = np.array(df['XMCD_REF'])
    tey_sum = np.array(df['XMCD_Sum'])
    xmcd_tey = np.array(df['XMCD_TEY'])
    lon_sum = np.array(df['XMCD_Sum1'])
    xmcd_lon = np.array(df['XMCD_TFY'])
 
    return wp, loff_sum, xmcd_loff, tey_sum, xmcd_tey, lon_sum, xmcd_lon

def XMCDlpowsum(sf,st_scan, end_scan):
    df = get_specScan(sf, st_scan)
    wp = np.array(df['Waveplate'])
    loff_sum = df['XMCD_Sum2']
    xmcd_loff = df['XMCD_REF']
    tey_sum = df['XMCD_Sum']
    xmcd_tey = df['XMCD_TEY']
    lon_sum = df['XMCD_Sum1']
    xmcd_lon = df['XMCD_TFY']

    num_avg=1
    for i in range(st_scan+1,end_scan+1):
        df = get_specScan(sf, i)
        wp = np.add(wp,df['Waveplate'])
        loff_sum = np.add(loff_sum,df['XMCD_Sum2'])
        xmcd_loff = np.add(xmcd_loff,df['XMCD_REF'])
        tey_sum = np.add(tey_sum,df['XMCD_Sum'])
        xmcd_tey = np.add(xmcd_tey,df['XMCD_TEY'])
        lon_sum = np.add(lon_sum,df['XMCD_Sum1'])
        xmcd_lon = np.add(xmcd_lon,df['XMCD_TFY'])
        num_avg+=1
    
    wp=np.array(wp/num_avg)
    loff_sum=np.array(loff_sum/num_avg)
    xmcd_loff=np.array(xmcd_loff/num_avg)
    tey_sum=np.array(tey_sum/num_avg)
    xmcd_tey=np.array(xmcd_tey/num_avg)
    lon_sum=np.array(lon_sum/num_avg)
    xmcd_lon=np.array(xmcd_lon/num_avg)
    
    return wp, loff_sum, xmcd_loff, tey_sum, xmcd_tey, lon_sum, xmcd_lon


def norm_xas(xas, ntype = 1, npnts = 5):
    
    if (ntype==1):
        scxas = xas - np.nanmin(xas)
        pkhght = np.nanmax(scxas)
        scxas = scxas/pkhght
        print('Peak height: ', pkhght)
    else:
        if (xas[0] < xas[-1]):
            scxas = xas - np.average(xas[0:npnts])
            edge = np.average(scxas[-npnts:])
            scxas = scxas/edge
            print('Edge jump: ', edge)
        else: 
            scxas = xas - np.average(xas[-npnts:])
            edge = np.average(scxas[0:npnts])
            scxas = scxas/edge
            print('Edge jump: ', edge)
    
    return scxas
            
def norm_xasl(xas):
    
    scxas = xas - np.nanmin(xas)
    pkhght = np.nanmax(scxas)
    scxas = scxas/pkhght
    print('Peak height: ', pkhght)
    
    return scxas, pkhght
            

def norm_xmcd(xas, xmcd):
    
    scxas = xas - np.nanmin(xas)
    norm = np.nanmax(scxas)
    scxas = scxas/norm
    scxmcd = xmcd/norm
    print('Normalization: ', norm)
    return scxas, scxmcd

def norm_hys(hysdata):
    
    schys = hysdata
    hyslen = round(len(hysdata)/2)
    mask = np.zeros(len(hysdata), dtype= bool)
    mask[0:hyslen+1] = True
    schys[mask] -= np.average(schys[mask][0:3])
    schys[mask] /= np.average(schys[mask][-3:])

    mask = np.zeros(len(hysdata), dtype= bool)
    mask[hyslen+1:] = True

    schys[mask] -= np.average(schys[mask][-3:])
    schys[mask] /= np.average(schys[mask][0:3])

    schys -=0.5
    schys *=2
    
    return schys

def norm_xmcdhys(hysdata,numpnt=10):
    
    schys = hysdata
    hyslen = round(len(hysdata)/2)
    mask = np.zeros(len(hysdata), dtype= bool)
    mask[0:hyslen+1] = True
    avg1 = np.average(schys[mask][:numpnt])
    avg2 = np.average(schys[mask][-numpnt:])
    avg=(avg2-avg1)
    schys[mask] -= avg1
    schys[mask] /= avg

    mask = np.zeros(len(hysdata), dtype= bool)
    mask[hyslen+1:] = True
    avg1 = np.average(schys[mask][:numpnt])
    avg2 = np.average(schys[mask][-numpnt:])
    
    schys[mask] -= avg2
    schys[mask] /= (avg1-avg2)

    avg+=(avg1-avg2)
    avg/=2
    schys -=0.5
    schys *=avg
    
    return schys

def spikerem(data,threshold):
    #Assumes first point is good

    loc_spike = []
    for i in range(len(data)-1):
        if abs(data[i+1] - data[i]) > threshold:
            loc_spike.append(i+1)
            
    
        
        
    
