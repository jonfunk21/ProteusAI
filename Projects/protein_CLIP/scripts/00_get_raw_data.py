import os

script_path = os.path.dirname(__file__)
data_dir = os.path.join(script_path, '../data')
dest = os.path.join(data_dir, 'raw')
enzyme_data = os.path.join(dest, 'enzyme.dat')

if not os.path.exists(dest):
    os.makedirs(dest)

expasy_url = 'https://ftp.expasy.org/databases/enzyme/enzyme.dat'

if not os.path.exists(enzyme_data):
    os.system(f'wget {expasy_url} -P {dest}')