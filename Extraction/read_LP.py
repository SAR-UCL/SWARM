import glob 
import os 
import numpy as np
import pandas as pd
import pandas.io.formats.excel

path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Data/20180521/'

pkt_format = '*.cdf'
pkt_size = 94

all_files_binary = []
for filename in glob.glob(os.path.join(path, pkt_format)):      
    with open(os.path.join(os.getcwd(), filename), 'rb') as f:
        open_bits = ["{:08b}".format(c) for c in f.read()] #Opens as hex
        all_files_binary.append(open_bits)
print('Number of files:', len(all_files_binary))
print('Size of file (B):', len(all_files_binary[0]))
#print(len(all_files_binary[0]))
#print(all_files_binary[0][54000])
#print(all_f


flatten_binary = [i for j in all_files_binary for i in j] #if multiple .cdf in folder, they flatten into a single list
num_of_pkts = (len(flatten_binary)//(pkt_size)) #determines the number of pkts. This should be 86400 * 2(hz) 
print('Number of packets in files:', num_of_pkts)

split_file = np.array_split(flatten_binary, num_of_pkts) #split into pkt sizes (90B)

#Split, then place back into a list, whilst looping through the file
#'11110000' -> 1111111 -> '1','1','1'...'0'
split_file_list = []
for i in split_file:
    pkt_index = list(i[0:pkt_size])
    join_index = (list(k for k in("".join(j for j in i)))) 
    split_file_list.append(join_index)


#print(split_file_list[0])
#print(len(split_file_list[87359]))

def signed32(binary_str):
            as_bytes = int(binary_str, 2).to_bytes(4, 'big') #4 = bytes / 32 = 8-bit
            return int.from_bytes(as_bytes, 'big', signed=False)

def signed64(binary_str):
            as_bytes = int(binary_str, 2).to_bytes(8, 'big') #4 = bytes / 32 = 8-bit
            return int.from_bytes(as_bytes, 'little', signed=False)

def getTimestampData(startIndex):
    counts = []
    for i in split_file_list:
        counts_index = i[(startIndex + 0):(startIndex + 64)]
        counts_join = str("".join(i for i in counts_index))
        counts_sixteen = [counts_join[index:index+64] for index in range(0, len(counts_join), 64)]
        counts.append(counts_sixteen)
    #print ('counts join', counts_join)

    #This feeds the non-time values into signed32 and then returns ints
    counts_to_int = [[signed64(x) for x in i] for i in counts]
    flatten_counts = [i for j in counts_to_int for i in j]
    return(flatten_counts)

def cdfDouble(binary_str):
    import struct
    as_bytes = int(binary_str, 2).to_bytes(8, 'big') #8 = bytes / 64 = 8-bit
    #return int.from_bytes(as_bytes, 'little', signed=False)
    return struct.unpack('>d', as_bytes)[0]

def getDoubleData(startIndex):
    counts = []
    for i in split_file_list:
        counts_index = i[(startIndex + 0):(startIndex + 64)]
        counts_join = str("".join(i for i in counts_index))
        counts_sixteen = [counts_join[index:index+64] for index in range(0, len(counts_join), 64)]
        counts.append(counts_sixteen)
    #print ('counts join', counts_join)

    #This feeds the non-time values into signed32 and then returns ints
    counts_to_int = [[cdfDouble(x) for x in i] for i in counts]
    flatten_counts = [i for j in counts_to_int for i in j]
    return(flatten_counts)

utc = getDoubleData(0)
lat = getDoubleData(80)
longi = getDoubleData(144)
radius = getDoubleData(208)
ne = getDoubleData(336)
te = getDoubleData(464)

utc_data = {'utc':utc,'lat':lat, 'long':longi, 'radius':radius,'density':ne,'temp':te}
utc_data = pd.DataFrame(utc_data)

def utc(utc):
    from astropy.time import TimeDelta, Time
    from astropy import units as u
    new_time = (Time(2000, format='jyear') + TimeDelta(utc*u.s)).iso
    return(new_time)

#utc_data['utc'] = utc_data['utc'].apply(utc)

print(utc_data)

'''
self.sci_only = []
for i in div_pkts_main:
    if i [0] == '00001000': #this is 08 in hex
        joined_pkts = "".join(j for j in i) #merge into single string
        relist_pkts = (list(j for j in joined_pkts)) #split into list
        self.sci_only.append(relist_pkts)


div_pkts_header = []
        for x in header_pkt_size:
            pkts_header_index = list(x[0:(pkt_header_size)+1]) #global header
            div_pkts_header.append(pkts_header_index)
        #print ('8-Bit Pkt Info, Header included', div_pkts_header) #split file into 90B lengths'''