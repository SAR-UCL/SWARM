
#-------------------------------------------------------------------------------------
# This file contains a number of useful routines for processing instrument data
#-------------------------------------------------------------------------------------

'''Merge multiple csvs and sort by yyyy-mm-dd and utc'''
def mergeCSV():
    import glob
    import pandas as pd

    #path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/Phoenix/Instrument Data/analysis/first_sixty/data/stm'
    #path = r'/Users/sr2/Documents/OneDrive - University College London/PhD/Research/Missions/Phoenix/Instrument Data/analysis/Mar-21/data/double-peak'
    path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Non-Flight Data/Analysis/Jan-22/data/systematic/train_set/'
    
    filenames = glob.glob(path + "/*.csv")

    #Loops through the .csv files in directory above
    dataframes = []
    for filename in filenames:
        dataframes.append(pd.read_csv(filename))

    #Concatinate into single dataframe
    merge_csv = pd.concat(dataframes, ignore_index=True)

    #merge_csv['Record'] = pd.to_datetime(merge_csv.date) #reformat date #date for Phoenix, #record for SWARM
    #sort_df = merge_csv.sort_values(['Record','Timestamp'], ascending =[True, True]) #sort by date, then by time
    
    def utc(utc):
        from astropy.time import TimeDelta, Time
        from astropy import units as u
        new_time = (Time(2000, format='jyear') + TimeDelta(utc*u.s)).iso
        return(new_time)
    

    def drop(df):

        df = df[['date','utc','mlt','lat','long','s_id','pass','Ne','Ti','pot','id','epb_gt']]
        #df['epb_gt'] = 0
        return df

    merge_csv = drop(merge_csv)

    #sort_df['utc'] = sort_df['utc'].apply(utc)
    #print(sort_df)

    csv_output_pathfile = path + "train_set.csv" # -4 removes '.pkts' or '.dat'
    merge_csv.to_csv(csv_output_pathfile, index = False, header = True)
    
    print(merge_csv)

mergeCSV()

'''Sort .csv by date and then time'''
def sortCSV():
    #path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/Phoenix/Instrument Data/merged_batches/science/presentation/science_eVcolumns.csv'
    path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/Phoenix/Instrument Data/analysis/day-night/5-day-night/data/all-5-day-night.csv'

    load_science = pd.read_csv(path)
    #load_science['date'] = pd.to_datetime(load_science['date'], format="%Y/%m/%d")
    #print(load_science)
    sorted_science = load_science.sort_values(['date', 'utc'], ascending =[True, True])
    #sorted_science = load_science.sort_values('pitch', ascending = False)
    print(sorted_science)
    csv_path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/Phoenix/Instrument Data/analysis/day-night/5-day-night/data/'
    csv_output_pathfile = csv_path + "all-5-day-night.csv" # -4 removes '.pkts' or '.dat'
    sorted_science.to_csv(csv_output_pathfile, index = False, header = True)

#sortCSV()

'''Converts .txt files from previous QB50 work into .csv'''
def readTxt():
    import pandas as pd

    path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/CIRCE/Documents/pkt-header_150421.txt' 
    to_csv =  r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/CIRCE/Documents' 


    #Read .txt and convert to .csv
    read_txt = pd.read_csv(path, header=None, delim_whitespace=True)
    #print(read_txt)
    csv_path = to_csv + '/pkt-header_150421.csv'
    #read_txt.columns = ['date','utc','lat','long','height','count'] #assign column headers
    print(read_txt)
    
    #Export as .csv
    read_txt.to_csv(csv_path, index = False, header = True)

#readTxt()
