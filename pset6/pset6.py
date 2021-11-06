import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
from scipy import signal as sig
from scipy import interpolate as intp
from scipy import integrate 


'''
I ran the code in the same folder as the folder of the data, so Im not sure what variable is should have that sets the directory
'''

def read_template(filename):
    dataFile=h5py.File(filename,'r')
    template=dataFile['template']
    th=template[0]
    tl=template[1]
    return th,tl
def read_file(filename):
    dataFile=h5py.File(filename,'r')
    dqInfo = dataFile['quality']['simple']
    qmask=dqInfo['DQmask'][...]

    meta=dataFile['meta']
    #gpsStart=meta['GPSstart'].value
    gpsStart=meta['GPSstart'][()]
    #print meta.keys()
    #utc=meta['UTCstart'].value
    utc=meta['UTCstart'][()]
    #duration=meta['Duration'].value
    duration=meta['Duration'][()]
    #strain=dataFile['strain']['Strain'].value
    strain=dataFile['strain']['Strain'][()]
    dt=(1.0*duration)/len(strain)

    dataFile.close()
    return strain,dt,utc

hanford = ['H-H1_LOSC_4_V2-1126259446-32.hdf5','H-H1_LOSC_4_V1-1167559920-32.hdf5','H-H1_LOSC_4_V2-1128678884-32.hdf5','H-H1_LOSC_4_V2-1135136334-32.hdf5']
livingston = ['L-L1_LOSC_4_V2-1126259446-32.hdf5','L-L1_LOSC_4_V1-1167559920-32.hdf5','L-L1_LOSC_4_V2-1128678884-32.hdf5','L-L1_LOSC_4_V2-1135136334-32.hdf5']
template = ['GW150914_4_template.hdf5','GW170104_4_template.hdf5','LVT151012_4_template.hdf5','GW151226_4_template.hdf5']

#take average of data to use for noise model since for a the noise for a
#detector should be similar for each event

sum_h, sum_l = 0,0 #create var for each detector which sum the data of each event
strain_hanford = [] # store array of data for each event at Hanford
strain_livingston = [] # store array of data for each event at Livingston
th = [] #template Hanford
tl = [] #template Livingston

for i in range(4):
    
    fname_h = hanford[i]
    strain_h,dt_h,utc_h=read_file(fname_h)#read file Hanford
    
    strain_hanford.append(strain_h)#append strain of each event of Hanford to list 'strain_hanford'
    sum_h += strain_h
    
    
    fname_l = livingston[i]#read file Livingston
    strain_l,dt_l,utc_l=read_file(fname_l)
    
    strain_livingston.append(strain_l)#append strain of each event of Livingston to list "strain_livingston"
    sum_l += strain_l
    
    temp_h, temp_l = read_template(template[i]) #read template
    th.append(temp_h) #append Hanford template 
    tl.append(temp_l) #append Livingston template

strain_h = sum_h/4 #average over all data of Hanford detector
strain_l = sum_l/4 #average over all data of Livingston detector

n = len(strain_h) #number of data points
dt = dt_h #time interval (evenly spaced)

strain_hanford = np.asarray(strain_hanford)
strain_livingston = np.asarray(strain_livingston)
th = np.asarray(th)
tl = np.asarray(tl)

#a) noise model
fs = 1/dt #sampling frequency

def noise_model(strain,nperseg,n_smooth):
    '''
    nperseg : number of data chunked
    n_smooth : width of window
    '''
    
    #use welch method for the noise 
    #the data are windowed using the tukey window to get noise of approx the same height
    freqs, noise_ps = sig.welch(strain,fs=fs,window='tukey',nperseg = nperseg)
    
    #smooth noise model by convolving with square window which does a moving average
    smooth_noise_ps = np.convolve(noise_ps,np.ones(n_smooth)/n_smooth,mode='same')
    
    return smooth_noise_ps, freqs

n_smooth = 5
nperseg = 2000
noise_h,freqs_h = noise_model(strain_h,nperseg,n_smooth) #noise model for Hanford
noise_l, freqs_l = noise_model(strain_l,nperseg,n_smooth) #noise model for Livingston

#interpolate noise so that the length is the same as the strain
noise_ps_h = intp.interp1d(freqs_h,noise_h,kind='linear') 
noise_ps_l = intp.interp1d(freqs_l,noise_l,kind='linear')

#b) search for events using mf
def mf(strain,template,noise_ps):

    win = sig.windows.tukey(n) #tukey window
    freq = np.fft.fftfreq(n,dt)
    noise_pwr_spectrum = noise_ps(np.abs(freq)) #noise with same length as strain
    
    sft = np.fft.fft(strain*win) #add window to data (same as for noise model)
    tft_white=np.fft.fft(template*win)/np.sqrt(noise_pwr_spectrum) #pre whiten template
    sft_white = sft/np.sqrt(noise_pwr_spectrum) #pre whiten data
    
    xcorr=np.fft.ifft(sft_white*np.conj(tft_white)) #cross correlation
    
    norm = ((template)**2/noise_pwr_spectrum).sum() #normalize by computing the Hessian
    
    return xcorr/np.sqrt(norm)

mf_h = []
mf_l = []

for i in range(4):
    h = mf(strain_hanford[i],th[i],noise_ps_h)
    l = mf(strain_livingston[i],tl[i],noise_ps_l)
    mf_h.append(h)
    mf_l.append(l)
mf_h = np.asarray(mf_h)
mf_l = np.asarray(mf_l)

'''
xcorr = mf_h[0]
plt.plot(np.abs(np.fft.fftshift(xcorr)),label='Hanford')
xcorr = mf_l[0]
plt.plot(np.abs(np.fft.fftshift(xcorr)),label='Livingston')
plt.xlabel('time')
plt.ylabel('strain')
plt.legend()
plt.show()

xcorr = mf_h[1]
plt.plot(np.abs(np.fft.fftshift(xcorr)),label='Hanford')
xcorr = mf_l[1]
plt.plot(np.abs(np.fft.fftshift(xcorr)),label='Livingston')
plt.xlabel('time')
plt.ylabel('strain')
plt.legend()
plt.show()

xcorr = mf_h[2]
plt.plot(np.abs(np.fft.fftshift(xcorr)),label='Hanford')
xcorr = mf_l[2]
plt.plot(np.abs(np.fft.fftshift(xcorr)),label='Livingston')
plt.xlabel('time')
plt.ylabel('strain')
plt.legend()
plt.show()

xcorr = mf_h[3]
plt.plot(np.abs(np.fft.fftshift(xcorr)),label='Hanford')
xcorr = mf_l[3]
plt.plot(np.abs(np.fft.fftshift(xcorr)),label='Livingston')
plt.xlabel('time')
plt.ylabel('strain')
plt.legend()
plt.show()'''

#c) estimation noise+snr
def snr(matched_filter,a=0,b=-1):
    noise_estimate = np.std(matched_filter[a:b]) #take std of matched filter
    snr = np.max(np.abs(matched_filter))/noise_estimate #signal to noise ratio
    
    return noise_estimate, snr

#estimate noise, snr and combined snr

for i in range(4):
    noise_h, snr_h = snr(mf_h[i])
    noise_l, snr_l = snr(mf_l[i])
    
    print('data : ', hanford[i],'and ', livingston[i])
    print('estimate of noise for Hanford', noise_h,'and Livingston', noise_l)
    print('snr Hanford', snr_h, 'snr Livingston', snr_l)
    print('combined snr for event : ',(snr_h+snr_l)/2,'\n')
    
#d) np.sqrt(norm) vs std(xcorr) wher xcorr is not normalized
def noise(strain,th,noise_ps):

    win = sig.windows.tukey(n) #window
    freq = np.fft.fftfreq(n,dt)
    noise_pwr_spectrum = noise_ps(np.abs(freq)) #noise with same length as strain
    
    sft = np.fft.fft(strain*win) #add window to data (same as for noise model)
    tft_white=np.fft.fft(th*win)/np.sqrt(noise_pwr_spectrum) #pre whiten template
    sft_white = sft/np.sqrt(noise_pwr_spectrum) #pre whiten data
    
    xcorr=np.fft.ifft(sft_white*np.conj(tft_white)) #cross correlation
    
    norm = ((th)**2/noise_pwr_spectrum).sum()
    
    return np.sqrt(norm), np.std(xcorr)

for i in range(4):
    noise_analytic_h, noise_estimate_h = noise(strain_hanford[i],th[i],noise_ps_h)
    noise_analytic_l, noise_estimate_l = noise(strain_livingston[i],tl[i],noise_ps_l)
    
    print(noise_analytic_h,noise_estimate_h,noise_estimate_h/noise_analytic_h)
    
#the analytic noise is not the same as the noise estimate
    
#e) loop integral over all value of freq using 

weight_freq = []

freq = np.fft.fftfreq(n,dt)
for i in range(4):
    mf = np.fft.fft(mf_h[i])
    total_cum = np.trapz(mf) #compute cumulative area
    for i,f in enumerate(freq):

        cum = np.trapz(mf[0:i]) #compute cum area for each point and find find which freq gives half of the total cum area
        if 0.5*np.abs(total_cum)-0.2 <= np.abs(cum) <= 0.5*np.abs(total_cum)+0.2 :
            half_freq = f
    weight_freq.append(half_freq)
    
print('half freq where the weight is separeted in 2 ',weight_freq)
