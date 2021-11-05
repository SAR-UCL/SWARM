'''
Adapted by Sachin Alexander Reddy, UCL MSSL.

Main source: #https://github.com/pignalberi/TITIPy/blob/master/Downloading_Swarm_data.py

03/11/21
'''

from ftplib import FTP
import os
import zipfile
import patoolib

SATE = ['A','C']
YEAR = [2019]
MONTH = [3]
DAYS = 1

for SAT in SATE:
    for YR in YEAR:
        for MON in MONTH:
            for DOM in range(DAYS):
                DOM = DOM + 16
                
                path = '/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/in-flight data/LP'  #Density and potential
                #path = '/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/in-flight data/EFI/March-19' #Ion temp
                #path = '/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/in-flight data/IBI/April-16' #Bubbles and IPIR

                path_initial = os.path.join(path,str(YR).zfill(4)+str(MON).zfill(2)+str(DOM).zfill(2)+str(SAT))
                path_downloaded_data_LP=os.path.join(path_initial)
                os.makedirs(path_downloaded_data_LP, exist_ok=True)

                try:
                    for sat in SAT:
                        ftp = FTP('swarm-diss.eo.esa.int')   # connect to host, default port
                        ftp.login()   # user anonymous, passwd anonymous
                        
                        #Specific the location of the instrument or product you wish you download
                        ftp.cwd('/Level1b/Latest_baselines/EFIxLPI/Sat_'+str(sat)) #Langmuir Probe (plasma den & space pot)
                        #ftp.cwd('/Level2daily/Latest_baselines/EFI/TIE/Sat_'+str(sat)) #Electric field ins (ion temp)
                        #ftp.cwd('/Level2daily/Latest_baselines/IBI/TMS/Sat_'+str(sat)) #Bubbles and iregularities
                        
                        
                        listing=[]
                        ftp.retrlines("LIST", listing.append)
                        
                        filenames=[]
                        for index in listing:
                            words=index.split(None, 8)
                            if(words[-1].lstrip()[-4:]=='.ZIP'):
                                filenames.append(words[-1].lstrip())
                        
                        #Checks if folder exists
                        flag=False
                        for filename in filenames:
                            if(filename[19:23]==str(YR).zfill(4) and filename[23:25]==str(MON).zfill(2) and filename[25:27]==str(DOM).zfill(2)):
                                file_founded=filename
                                flag=True
                                break
                    
                        os.chdir(path_downloaded_data_LP)

                        print('Downloading file for Swarm '+str(sat)+' for '+str(YR).zfill(4)+'/'+str(MON).zfill(2)+'/'+str(DOM).zfill(2)+' ...')
                        
                        with open(file_founded, 'wb' ) as file :
                                ftp.retrbinary('RETR %s' % file_founded, file.write)
                                file.close()                

                        if(not flag):
                            print('File not found for Swarm '+str(sat)+' for '+str(YR).zfill(4)+'/'+str(MON).zfill(2)+'/'+str(DOM).zfill(2))
                            pass
                            #raise Exception('File not found for Swarm '+str(sat)+' for '+str(YR).zfill(4)+'/'+str(MON).zfill(2)+'/'+str(DOM).zfill(2))
                            
                        #os.chdir(main_folder) 

                    ftp.close()
                except RuntimeError:
                    raise Exception('Problems in connecting to or downloading from Swarm FTP')

                #Unzip and delete zip
                try:
                    swarm_files=os.listdir(path_downloaded_data_LP)
                    swarm_files.sort()
                    for file in swarm_files:
                        patoolib.extract_archive(file, outdir=path_downloaded_data_LP,verbosity=-1)
                        os.remove(file)

                    files=os.listdir()
                    for file in files:
                        if(file[-4:]=='.cdf'):
                            pass
                        else:
                            os.remove(file)
                    print('Extraction for Swarm '+str(sat)+' for '+str(YR).zfill(4)+'/'+str(MON).zfill(2)+'/'+str(DOM).zfill(2)+' complete')
                except RuntimeError:
                    raise Exception('Problems in connecting to or downloading from Swarm FTP')