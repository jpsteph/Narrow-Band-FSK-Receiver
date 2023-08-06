import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.signal import wiener
import cProfile
import scipy.io.wavfile

'''
from rtlsdr import RtlSdr
def init_radio(Fs, Fc, gain = None):
    sdr = RtlSdr()
    sdr.sample_rate = Fs
    sdr.center_freq = Fc 
    if gain != None:     
        sdr.gain = gain
    else:
        gain = 'auto'
    return sdr
'''
    
def get_radio_samples(sdr, samplenum, simple = bool):

    if simple:
        return sdr.read_samples(samplenum)

    samples_per_call = 100000

    sm = sdr.read_samples(samples_per_call)
    while sm.shape[0] < samplenum:
        buffer = sdr.read_samples(samples_per_call)
        sm = np.concatenate((sm, buffer))

    return sm

def close_radio(sdr):
    sdr.close

#numpy type can either be "np.complex128" or "np.complex64"
def get_samples_from_file(file_name, bin = True, numpytype = str):
    if bin == True:
        if numpytype == "complex128":
            return np.fromfile(file_name +".bin", dtype=np.complex128)
        elif numpytype == "complex64":
            return np.fromfile(file_name +".bin", dtype=np.complex64)
    else:
        if numpytype == "complex128":
            return np.fromfile(file_name, dtype=np.complex128)
        elif numpytype == "complex64":
            return np.fromfile(file_name, dtype=np.complex64)
        

def get_parameters_from_txt(file_name):
    proptxt = open(file_name + "_properties.txt", mode = "r")
    Fs_str = proptxt.readline() #getting sample rate
    Fc_str = proptxt.readline() #getting center frequency
    type_str = proptxt.readline() #getting center frequency
    proptxt.close()
    return int(Fs_str.split(',')[1].replace('\n','')), float(Fc_str.split(',')[1].replace('\n','')), type_str.split(',')[1].replace('\n','')

def save_IQ_to_bin_file(file_name, IQ_data):
    IQ_data.tofile(file_name + ".bin")

def spectogram(IQ_sig, bins, Fs, graph=None, n_point_fft=None):
    """
    Compute the spectrogram of a signal.

    Parameters:
        sig (array-like): Input IQ signal data.
        bins (int): Number of FFT bins for each segment.
        Fs (float): Sampling frequency of the signal.
        graph (optional): If not None, display the spectrogram plot.
        specto_resolution (optional): Number of FFT points for spectral resolution.

    Returns:
        array-like: Spectrogram matrix.

    """

    num_rows = len(IQ_sig) // bins

    if n_point_fft is not None:   #creating spectrogram matrix to store resulting ffts (column size is dictated by fft size)
        spectrogram = np.zeros((num_rows, n_point_fft))
    else:
        spectrogram = np.zeros((num_rows, bins))

    for i in range(num_rows): 
        #getting section of samples to do fft
        start = i * bins
        end = (i + 1) * bins
        segment = IQ_sig[start:end]

        if n_point_fft is not None:
            spectrum = np.fft.fftshift(np.fft.fft(segment, n_point_fft))    #doing n-point fft dicated by n_point_fft
        else:
            spectrum = np.fft.fftshift(np.fft.fft(segment)) #doing fft of resolution based on size of sample segment (larger segments will result in a higher resolution fft)

        spectrogram[i, :] = 10 * np.log10(np.square(np.abs(spectrum)))    #getting magnitude of fft and converting to dbm and storing fft in row in spectrogram

    if graph is not None:
                #x labels                           #y labels
        extent = [(-Fs / 2) / 1e6, (Fs / 2) / 1e6, len(IQ_sig) / Fs, 0] #spectrogram goes top to bottom
        plt.imshow(spectrogram, aspect='auto', extent=extent)
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Time [s]")
        plt.show()

    return spectrogram

def simple_psd(buff, NFFT, Fs, frf):
    plt.psd(buff, #data buffer
    NFFT, #FFT Size
    Fs, #Sampling Rate
    frf, #Center Frequency (Hz)
    scale_by_freq = True
    ) 
    plt.show()

def plot_fir_filter(num_taps, cutoff, window, Fs):
    h = scipy.signal.firwin(num_taps, cutoff, window = window, fs = Fs)
    freq_response = np.abs(np.fft.fft(h))
    # Determine the sampling frequency and the length of the FIR filter coefficients
    sampling_freq = Fs  # Replace with your desired sampling frequency
    filter_length = len(h)
    # Create the frequency axis
    freq_axis = np.fft.fftfreq(filter_length, d=1/sampling_freq)
    plt.plot(freq_axis, freq_response)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('FIR Filter Frequency Response')
    plt.grid(True)
    plt.show()

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def complex_moving_average(input_signal, window_size):
    # Pad the input signal with zeros at the beginning
    #padded_signal = np.pad(input_signal, (window_size - 1, 0), mode='constant')
    
    # Generate the window coefficients
    window = np.ones(window_size) / window_size
    
    # Apply the moving average filter
    filtered_signal = np.convolve(input_signal, window, mode='valid')
    
    return filtered_signal

def IQ_conj(IQarr):
    return np.conjugate(IQarr)

def IQ_delay(IQarr, delay, real = None):
    arrreal = IQarr.real
    arrimag = IQarr.imag
    #rolling by delay indexes
    arrrealdelay= np.roll(arrreal, delay)
    arrimagdelay = np.roll(arrimag, delay)

    if real:
        return arrrealdelay
    else:
        return arrrealdelay + 1j * arrimagdelay

#this function uses a quadrature demodulation algorithm to create an instantaneous frequency signal
def frequency_detector(samples, samplerate, averaging = None, graph = None):
    #delaying samples by 1
    samples_delay = IQ_delay(samples, 1)

    #getting complex conjugate signal
    samples_delay_conj = IQ_conj(samples_delay)

    #np multiply implements complex multiplication
    samples_mult = np.multiply(samples, samples_delay_conj)

    #taking the angle of the previous signal gives us instantaneous frequency (then multiplying by a constant to get the correct frequency) 
    samples_angle = np.angle(samples_mult)
    freq = np.multiply((samplerate / 6.3), samples_angle)

    if averaging != None:
        freq = moving_average(freq, averaging)

    if graph != None:
        plt.plot(freq)
        plt.show()

    return freq

def plot_IQ_time_domain(x):
    I = np.real(x)
    Q = np.imag(x)

    # Assuming a sampling rate of 1 sample per unit time
    x_axis = np.arange(len(I))

    plt.figure()
    plt.plot(x_axis, I, label='I')
    plt.plot(x_axis, Q, label='Q')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()


def fix_iq_imbalance(x):
    # remove DC and save input power
    z = x - np.mean(x)
    p_in = np.var(z)

    # scale Q to have unit amplitude (remember we're assuming a single input tone)
    Q_amp = np.sqrt(2*np.mean(x.imag**2))
    z /= Q_amp

    I, Q = z.real, z.imag

    alpha_est = np.sqrt(2*np.mean(I**2))
    sin_phi_est = (2/alpha_est)*np.mean(I*Q)
    cos_phi_est = np.sqrt(1 - sin_phi_est**2)

    I_new_p = (1/alpha_est)*I
    Q_new_p = (-sin_phi_est/alpha_est)*I + Q

    y = (I_new_p + 1j*Q_new_p)/cos_phi_est

    #print 'phase error:', arccos(cos_phi_est)*360/2/pi, 'degrees'
    #print 'amplitude error:', 20*log10(alpha_est), 'dB'

    return y*np.sqrt(p_in/np.var(y))

def get_samples(result_queue, sdr):
    result_queue.put(sdr.read_samples(500000))

def process_samples(x, h, bz, az):
    x = np.diff(np.unwrap(np.angle(x)))

    x = scipy.signal.lfilter(h,1.0,x)
    
    x = moving_average(x, 30)
    #wiener is only good when signal to noise ratio is low
    x = wiener(x, 51)
    
    x = scipy.signal.lfilter(bz, az, x)

    # decimate by 6 to get mono audio
    x = x[::6]
    x = moving_average(x, 3)

    # normalize volume so its between -1 and +1
    x /= np.max(np.abs(x))

    # some machines want int16s
    x *= 32767
    x = x.astype(np.int16).tobytes()

    return x

#control loop to lock FSK signal to IF by trying to maximum DC content of signal 
def freq_offset_control(ki, kp, error, error_sum):
    p_term = kp * error

    # Compute the integral term
    error_sum += error
    i_term = ki * error_sum

    # Compute the control output
    control_output = p_term + i_term

    return control_output, error_sum

def complex_moving_average_filter(signal, window_size):
    # Create the complex moving average kernel
    kernel = np.ones(window_size, dtype=np.complex128) / window_size

    # Apply the complex moving average filter
    filtered_signal = np.convolve(signal, kernel, mode='same')

    return filtered_signal

def break_fft_into_parts(signal, section_size):
    num_sections = len(signal) // section_size
    fft_results = []

    for i in range(num_sections):
        start_index = i * section_size
        end_index = (i + 1) * section_size
        section = signal[start_index:end_index]
        fft_results.append(np.sum(np.abs(np.fft.fft(section[0:10]))))

    return np.mean(np.array(fft_results))

def frequency_domain_smoothing(signal, smoothing_factor):
    fft_result = np.fft.fft(signal)
    num_samples = len(signal)
    frequencies = np.fft.fftfreq(num_samples)

    # Apply smoothing filter
    smoothing_filter = np.exp(-smoothing_factor * frequencies**2)
    smoothed_fft = fft_result * smoothing_filter

    # Inverse Fourier transform to obtain the smoothed signal
    smoothed_signal = np.fft.ifft(smoothed_fft)

    return smoothed_signal

def average_ffts(signal, num_ffts):
    segment_length = len(signal) // num_ffts

    # Split the signal into segments
    segments = np.split(signal[:segment_length*num_ffts], num_ffts)

    # Apply FFT to each segment
    ffts = np.fft.fft(segments)

    # Compute average of FFT results
    averaged_fft = np.mean(ffts, axis=0)

    # Take inverse FFT of averaged signal
    averaged_signal = np.fft.ifft(averaged_fft)

    return averaged_signal

#simple_psd(x, 2**7, Fs = Fs, frf = Fc)

#plt.plot(np.abs(np.fft.fftshift(np.fft.fft(x))))
#plt.show()

#w, h = scipy.signal.freqz(bz, az)
#f = w * Fs / (2 * np.pi)
#plt.plot(f, np.abs(h))
#plt.sh

def freq_offset(sdr, Fs, Fc): 
    h1 = scipy.signal.firwin(5, Fs/20, window = "hamming", fs = Fs)

    kp = .02 # Proportional gain
    ki = .0001 # Integral gain
    new_shift = 5e3
    error = 0
    error_sum = 0
    plt.ion()

    fig, ax = plt.subplots()
    line, = ax.plot([], [])

    for n in range(15):
        x = sdr.read_samples(500000)

        t = np.arange(x.size)/Fs
        x_shift  = x * np.exp(2j*np.pi*new_shift*t)
        x_print = x_shift
        
        t = np.arange(x.size)/Fs
        x  = scipy.signal.lfilter(h1,1.0,x)
        x_shift  = scipy.signal.lfilter(h1,1.0,x_shift)
        x_sum = np.sum(np.abs(np.fft.fftshift(np.fft.fft(x, 2**19))))
        x_shift_sum = np.sum(np.abs(np.fft.fftshift(np.fft.fft(x_shift, 2**19))))
        print("Sum of Non-Shifted: " + str(x_sum))
        print("Sum of Shifted: " + str(x_shift_sum))
        
        error = (x_shift_sum - x_sum) * (new_shift/abs(new_shift)) 

        new_shift, error_sum = freq_offset_control(ki, kp, error, error_sum) 

        print('Error: ' + str(error))
        print("Frequency Shift: " + str(new_shift) + '\n')
        
        
        f, psd= plt.psd(x_print, 2**9, Fs = Fs, Fc = Fc, scale_by_freq = True) 
        line.set_data(f, psd)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)
        ax.clear()

    return new_shift

#gets average based on max and min
def get_avg(sig, avgnum):
    siglen = sig.size
    xinit = 3000
    sigmax = np.array([])
    sigmin = np.array([])
    x = xinit
    while x < siglen - avgnum - xinit:
        sigarr = sig[x:x+avgnum]
        sigmax = np.append(sigmax, np.max(sigarr))
        sigmin = np.append(sigmin, np.min(sigarr))

        x += avgnum
    return (np.mean(sigmax) + np.mean(sigmin)) / 2

#next three functions are methods to capture clock phase/frequency for coherent recieving.    
def midpoint(a):
    mean_a = np.mean(a)
    mean_a_greater = np.ma.masked_greater(a, mean_a)
    high = np.ma.median(mean_a_greater)
    mean_a_less_or_equal = np.ma.masked_array(a, ~mean_a_greater.mask)
    low = np.ma.median(mean_a_less_or_equal)
    a = (high + low) / 2
    return (high + low) / 2

def find_clock_frequency(spectrum):
    maxima = scipy.signal.argrelextrema(spectrum, np.greater_equal)[0]
    while maxima[0] < 2:
        maxima = maxima[1:]
    if maxima.any():
        threshold = max(spectrum[2:-1])*0.8
        indices_above_threshold = np.argwhere(spectrum[maxima] > threshold)
        return maxima[indices_above_threshold[0]]
    else:
        return 0

def wpcr(a):
    if len(a) < 4:
        return []
    #b = (a > midpoint(a)) * 1.0    #NOTE UNCOMMENT AND DELETE NEXT LINE FOR BETTER MIDPOINT MEASUREMENT ACCURACY 
    b = (a > np.mean(a)) * 1.0
    d = np.diff(b)**2
    if len(np.argwhere(d > 0)) < 2:
        return []
    f = scipy.fft.fft(d, len(a))
    p = find_clock_frequency(abs(f))
    if p == 0:
        return []
    cycles_per_sample = (p*1.0)/len(f)
    clock_phase = 0.5 + np.angle(f[p])/(np.pi * 2)
    if clock_phase <= 0.5:
        clock_phase += 1
    symbols = []
    for i in range(len(a)):
        if clock_phase >= 1:
            clock_phase -= 1
            symbols.append(a[i])
        clock_phase += cycles_per_sample
    debug = True
    if debug:
        print("peak frequency index: %d / %d" % (p, len(f)))
        print("samples per symbol: %f" % (1.0/cycles_per_sample))
        print("clock cycles per sample: %f" % (cycles_per_sample))
        print("clock phase in cycles between 1st and 2nd samples: %f" % (clock_phase))
        print("clock phase in cycles at 1st sample: %f" % (clock_phase - cycles_per_sample/2))
        print("symbol count: %d" % (len(symbols)))
    return symbols

def find_bit_sequence(binary_list, sequence):
    sequence_length = len(sequence)
    for i in range(len(binary_list) - sequence_length + 1):
        if binary_list[i:i + sequence_length] == sequence:
            return i
    return -1

def bits_list_to_number(bits_list):
    # Convert the list of bits to a string representation
    bits_string = ''.join(str(bit) for bit in bits_list)
    
    # Convert the binary string to an integer using the int() function
    number = int(bits_string, 2)
    return number

def bits_list_to_ascii(bits_list):
    # Convert the list of bits to a string representation
    bits_string = ''.join(str(bit) for bit in bits_list)
    
    # Convert the binary string to an integer using the int() function
    number = int(bits_string, 2)

    # Convert the integer to an ASCII character using chr()
    ascii_char = chr(number)
    return ascii_char

def main():

    #sdr config
    Fc = 433e6
    Fs = 250000
    
    '''
    sdr = init_radio(Fs = Fs, Fc = Fc, gain = None)
    x = sdr.read_samples(500000)
    x = sdr.read_samples(500000)
    '''

    # De-emphasis filter, H(s) = 1/(RC*s + 1), implemented as IIR via bilinear transform
    h = scipy.signal.firwin(6, Fs/4, window = "hamming", fs = Fs)
    #bz, az = scipy.signal.bilinear(1, [90e-5, 1], fs=Fs)

    squelch_theshold_constant = 0.6

    BER_total = 0
    BER_fail = 0

    squelch_indices_length_min = 50

    #preamble is unreliable, using info bits (0x0304)
    info_bits = [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0]
    byte_size = 8

    count = 0
    while True:

        '''
        spectogram(x, 2**8, Fs, graph = True, n_point_fft = 2**8)

        simple_psd(x, 2**8, Fs, frf = Fc)

        plt.plot(np.abs(x))
        plt.show()

        '''
        x = get_samples_from_file('RFM69HW_FSK_Signal', bin = True, numpytype = 'complex128')
        #comment these out to remove graphs
        spectogram(x, 2**8, Fs, graph = True, n_point_fft = 2**8)
        plt.plot(np.abs(x))
        plt.show()

        #anti-alias/narrow band filtering and non-integer decimation
        x_filtered = scipy.signal.convolve(x, h) 
        #non-integer decimation
        Fs = Fs//1.3333
        indices_to_remove = np.arange(1, x_filtered.size, 4) 
        x_filtered = np.delete(x_filtered, indices_to_remove) 
        
        #simple power squelch to detect signals 
        squelch_theshold = np.max(np.abs(x_filtered)) - squelch_theshold_constant 
        x_squelch_indices = np.nonzero(np.abs(x_filtered) > squelch_theshold)[0]
        differences = np.diff(x_squelch_indices)

        #transforming signal indices above squelch threshold into list of np arrays 
        x_squelch_indices_matrix = [ [x_squelch_indices[0]] ]
        i = int(0)
        for diff in differences:
            if diff > 100:
                x_squelch_indices_matrix.append([x_squelch_indices[i+1]])
            else:
                x_squelch_indices_matrix[-1].append(x_squelch_indices[i+1])
            i += 1

        #demod/clock synch of list of np arrays
        count_arr = 0
        for x_squelch_indices_subarray in x_squelch_indices_matrix:
            
            if len(x_squelch_indices_subarray) > squelch_indices_length_min:

                x_squelch = x_filtered[np.min(x_squelch_indices_subarray) : np.max(x_squelch_indices_subarray)]

                x_quad_demod = np.diff(np.unwrap(np.angle(x_squelch)))  #instantaneous frequency/quadrature demod
                
                x_moving_average = moving_average(x_quad_demod, 3)  #seems like moving average is computationally cheap
                x_moving_average = moving_average(x_moving_average, 3)

                x_one_zero = np.where(x_moving_average > np.mean(x_moving_average), 1, 0)   #thresholding to 1, 0 for clock recovery

                #plt.plot(x_one_zero)
                #plt.show()

                bits = wpcr(x_one_zero) #clock recovery
                print(bits)


                info_index = find_bit_sequence(bits, info_bits)
            
                if info_index != -1 and info_index < 500:

                    payload_byte_size = bits_list_to_number(bits[info_index-byte_size:info_index]) - 3
                    #to_node_id = bits_list_to_number(bits[info_index:info_index+byte_size])
                    #node_id = bits_list_to_number(bits[info_index+byte_size:info_index+byte_size*2])

                    payload_string = ''
                    for i in range(0, payload_byte_size):
                        payload_string += bits_list_to_ascii(bits[info_index+byte_size*(2 + i):info_index+byte_size*(3 + i)])

                    print('Subarray Num:')
                    print(count_arr)
                    count_arr += 1

                    print('Squelch Array Index in Matrix:')
                    print(info_index)

                    print('Payload Size:')
                    print(payload_byte_size)

                    print('Payload:')
                    print(payload_string)

                    #if payload_string != "TEST":
                    if "TEMP" not in payload_string:
                        BER_fail += 1

                        plt.plot(np.abs(x_filtered))
                        plt.show()

                        plt.plot(x_one_zero)
                        plt.show()
                    
                        
                    BER_total += 1
                    print('\nBER Total So Far: ' + str(BER_fail) + ' Fails out of ' + str(BER_total) + ' Total')
                        
                else:
                    #plt.plot(np.abs(x))
                    #plt.show()
                    print('Sync Sequence Not Found in Bits...')
            else:
                #plt.plot(np.abs(x))
                #plt.show()
                print('Recieved Partial Packet or Unknown Signal... Discarding') 

        #direct implmentation of sdr.readsamples
        '''
        num_bytes = 2*500000
        raw_data = sdr.read_bytes(num_bytes)
        x = sdr.packed_bytes_to_iq(raw_data)
        '''
        #count += 1

if __name__ == '__main__':
    #cProfile.run('main()') #used to profile speed of each individual module/function 
    main()

