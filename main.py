'''
Fan Coil Unit Simulation Framework: 
    Inputs: 
        1. FCU Data Path
            Folder containing the FCU Data 
        2. Fault Data Path
            File containing the relevant Fault Code mappings. 
    Outputs:
        1. GZIP Files: 
            - Located in models/data/ folder. 
                (This is a much faster way to load the data) 
            - Each file is a simulation of a single fault.
            

                                
'''
import pandas as pd 
import numpy as np
from glob import glob 
from tqdm import tqdm 
import json 
import time 


class FCU:
    def __init__(self, fcu_path, fault_path):
        self.fcu_path = fcu_path
        self.fault_path = fault_path
        self.path_df = self.load_data()
        self.lodf = self._import_data()
        self.lodf_stats = self._load_stats()
        self.lodf_sim = self._gen_sims()
        self.all_faults = list(set(list(self.lodf.keys())) - set(['FCU_FaultFree']))
        self.df_skeleton = self._df_setup()
        print("Ready\n\n")
        
    def _load_fcu_files(self):
        '''
        Load FCU Files
        '''
        files = glob(self.fcu_path + '*')
        fault_files = [x.split(self.fcu_path)[-1] for x in files]
        fault_file_path = [x for x in files]
        fname_df = pd.DataFrame(fault_files, columns=['fname'])
        fname_df['fpath'] = fault_file_path
        return fname_df
    
    def _load_faults(self):
        '''
        Load Faults
        '''
        faults = pd.read_csv(self.fault_path)
        different_faults = list(faults['Fault Type'].unique())
        fault_dict = dict(zip(different_faults, range(len(different_faults))))
        faults['FaultCode'] = faults['Fault Type'].map(fault_dict)
        self.faults = faults.copy()
        return faults
    
    def load_data(self):
        '''
        Load Data
        '''
        fname_df = self._load_fcu_files()
        faults = self._load_faults()
        a = fname_df.merge(faults[['Fault File Name', 'Fault Type', 'Fault Intensity ', 'FaultCode']], left_on='fname', right_on='Fault File Name')
        return a 
    
    def _import_data(self):
        '''
        Load in Raw data
        '''
        a = self.path_df
        # Store Each of the Tables in A Dictionary
        lodf = {}
        for i in tqdm(range(a.shape[0]), desc = "Importing Raw Data"):
            table_name = a['fname'][i].split('.csv')[0]
            df = pd.read_csv(a['fpath'][i])
            df['FaultCode'] = int(a['FaultCode'][i])
            #df['FaultDesc'] = a['Fault Intensity '][i]
            #df.to_sql(table_name, conn, index=False, if_exists='replace')
            lodf[table_name] = df
        return lodf
    
    def _load_stats(self):
        '''
        Get Stats
        '''
        lodf = self.lodf.copy()
        faults = self.faults.copy()
        # Get Stats From The Fault Tables: 
        lodf_stats = {}
        for i in tqdm(list(lodf.keys()), desc = 'Getting Stats From Faulty Tables'):
            fault = lodf[i].copy().drop(columns = ['Datetime', 'FaultCode'])
            fault_code = faults.loc[faults['Fault File Name'] == i + '.csv']['FaultCode'].values[0]
            mu = fault.mean(axis = 0)
            sd = fault.std(axis = 0)
            lodf_stats[i] = {'mu': mu, 'sd': sd, 'fault_code': fault_code}
            
  
        return lodf_stats

    def _gen_sims(self):
        '''
        Generate Simulations 
        '''
        lodf_stats = self.lodf_stats.copy()
        # Simulate All Data 
        lodf_sim = {}
        for i in tqdm(list(lodf_stats.keys()), desc = 'Pre-Simulating All Data'):
            mu = lodf_stats[i]['mu']
            sd = lodf_stats[i]['sd']
            fault_code = lodf_stats[i]['fault_code']
            n = self.lodf[i].shape[0]
            lodf_sim[i] = pd.DataFrame(np.random.normal(mu, sd, (n, len(mu))), columns = mu.index)
            lodf_sim[i]['FaultCode'] = fault_code
        
        return lodf_sim
    
    def _df_setup(self):
        '''
        Setup Data:
        1. Create a Skeleton DataFrame
        2. Create a Flag for Operating Hours, 6am - 6pm
        
        '''
        time_index = pd.to_datetime(self.lodf['FCU_FaultFree'].Datetime)
        cols = self.lodf['FCU_FaultFree'].copy().drop(columns = ['Datetime']).columns
        df = pd.DataFrame(index = time_index, columns = cols)
        operating_index = df[df.index.dayofweek < 5].between_time('06:00', '18:00').index
        op_col = np.where(df.index.isin(operating_index), 1, 0)
        df.insert(1, 'op', op_col)
        return df

    def sim_setup(self, percent_faulty = 0.2):
        """ Return the index of the faulty and non-faulty data """
        df = self.df_skeleton.copy()
        n = df.shape[0]
        n_faulty = int(n * percent_faulty)
        n_non_faulty = n - n_faulty
        idx_faulty = np.random.choice(range(n), n_faulty, replace = False)
        idx_non_faulty = np.setdiff1d(range(n), idx_faulty)
        return idx_faulty, idx_non_faulty
    
    def single_sim(self, fault, percent_faulty = 0.2):
        '''
        Simulate A single Fault with the given fault code
        '''
        df1 = self.df_skeleton.copy()
        cols = self.lodf['FCU_FaultFree'].copy().drop(columns = ['Datetime']).columns
        time_index = df1.index
        faulty_idx, non_faulty_idx = self.sim_setup(percent_faulty)
        df1.loc[time_index[faulty_idx], cols] = self.lodf_sim[fault].iloc[faulty_idx].values
        df1.loc[time_index[non_faulty_idx], cols] = self.lodf_sim['FCU_FaultFree'].iloc[non_faulty_idx].values
        return df1
    
    def multi_sim(self, faults, percent_faulty = 0.2):
        """ Simulate Multiple Faults """
        df1 = self.df_skeleton.copy()
        cols = self.lodf['FCU_FaultFree'].copy().drop(columns = ['Datetime']).columns
        time_index = df1.index
        faulty_idx, non_faulty_idx = self.sim_setup(percent_faulty)
        for fault in faults:
            df1.loc[time_index[faulty_idx], cols] = self.lodf_sim[fault].iloc[faulty_idx].values
        df1.loc[time_index[non_faulty_idx], cols] = self.lodf_sim['FCU_FaultFree'].iloc[non_faulty_idx].values
        return df1
        

if __name__ == '__main__':
    print("If not you then who? If not now then when?")
    
    # Note the / at the end of the path. 
    # For Windows, use Double Backslashes. 
    fcu_path = 'FCU/LBNL_FDD_Dataset_FCU/'
    fault_path = 'features/Faults.csv'
    fcu = FCU(fcu_path, fault_path)
    
    
    # Simulate a single fault
    f = fcu.all_faults[0]
    # df1 = fcu.single_sim(f)
    # print(df1.head())
    import gzip
    d = {}
    for i in tqdm(fcu.all_faults):
        df = fcu.single_sim(i)
        df.to_csv('models/data/' + i + '.gz', compression = 'gzip')
        
          
        


