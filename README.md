# Narrow-Band-FSK-Receiver
Narrow Band FSK Receiver Based on RFM69HW Radio Chip
(Note the Sampling Rate of the Bin File is 250000).  Should work plug and play with the bin files if you have the necessary libraries installed.  The script uses power squelch thresholds to identify the signal and takes the signal indices above the threshold.  Quadrature demod is performed on the signal array with the specified indices.  Bit rate calc and clock recovery is based on this video: https://www.youtube.com/watch?v=rQkBDMeODHc.
