import os, sys
sys.path.insert(1, os.path.join(os.getcwd(), '..'))
import shallowsim as sb
import pandas as pd
import math
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import time
from tqdm import tqdm, trange

# Initialize model arguments and configuration
args = sb.ModelArgs()
c = sb.Config()
gpu_all_decode = sb.get_gpu_info('./device/gpu_info.csv',
                                 decoding_mode=True, print_console=True)

# Generate data
dfs = []
for seq_len in trange(4096, 5120, 1024):
    c.seq_len = seq_len
    df = sb.decode_time_with_ep_list(args, gpu_all_decode, c, fp8_combine=True)
    df['index_value'] = seq_len
    # df_o = df.groupby(['GPU', 'BatchSize', 'EP'], as_index=False).apply(lambda t: t[t.Total == t.Total.max()]).sort_values(['Total'], ascending=False).reset_index(drop=True)
    # df_o.drop_duplicates(subset=['GPU', 'BatchSize', 'EP'], keep='first', inplace=True)
    dfs.append(df)
df = pd.concat(dfs)    
df.reset_index(inplace=True, drop=True)
df.to_csv('perf_vs_seq_len.csv')