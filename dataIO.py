
import os
import re
import numpy as np


def load_3ds(data_path):
    try:            
        with open(data_path, 'rb') as f:
                # Read metadata
                metadata = parse_metadata(f)

                # Define values based on metadata
                xdim, ydim = metadata['grid_dim']                     
                pixels = xdim * ydim                                
                num_params = int(metadata['num_parameters'])          # N_para
                spec_points = int(metadata['points'])                 # Ne
                num_channels = len(metadata['channels'])              # N_channel

                # Define # of binary blocks per pixel
                binary_blocks = num_params + spec_points * num_channels

                # Read binary floats (big-endian) to 1D array
                all_floats = np.fromfile(f,
                                         dtype = '>f4',
                                         count = pixels * binary_blocks)
                
                # Re-shape to 2D array (1 datablock per pixel)
                all_floats = all_floats.reshape((pixels, binary_blocks), order='C')

                # Split array between parameters and data
                par_flat = all_floats[:, :num_params]    #(pixels, N_para)
                data_flat = all_floats[:, num_params:]   #(pixels, Ne * N_channel)

                # Reshape parameter data into N_y x N_x x Ne
                parameters = par_flat.reshape((xdim, ydim, num_params), order='C')

                # Reshape spectral data into N_y x N_x x Ne x N_channel
                data_4D = (
                     data_flat
                        .reshape((pixels, spec_points, num_channels), order = 'F')
                        .reshape((xdim, ydim, spec_points, num_channels), order = 'C')
                )

    except FileNotFoundError:
        print('File does not exist.')
        return None, '', {}

    return data_4D, metadata, parameters

def parse_metadata(f):
    metadata = {}

    while True:
        line = f.readline().decode().strip()
        if line == ':HEADER_END:':
            break

        # Key-value pairs are separated by '='
        # This code splits them at the '=' 
        # removes quotes around values, replaces space with _
        if '=' not in line:
            continue
        key, val = line.split('=', 1)
        key = key.lower().replace(' ', '_')
        val = val.strip().strip('"')

        # Grid Size --> grid_dim
        if key == 'grid_dim':
            metadata['grid_dim'] = tuple(int(x.strip()) for x in val.split('x'))

        # Parses semi-colon separated floats to vector
        elif key == 'grid_settings':
            metadata['grid_settings'] = [float(x) for x in val.split(';')]

        # Puts experiment parameters into list format
        elif key in ('fixed_parameters', 'experiment_parameters', 'channels'):
            metadata[key] = [x.strip() for x in val.split(';')]

        elif key == '#_parameters_(4_byte)':
            metadata['num_parameters'] = int(val)

        elif key == 'experiment_size_(bytes)':
            metadata['experiment_size'] = int(val)

        elif key == 'points':
            metadata['points'] = int(val)

        elif key == 'delay_before_measuring_(s)':
            metadata['measurement_delay'] = float(val)

        # Normalizes any unexpected keys
        else:
            key = re.sub(r'[^a-z0-9_]', '_', key.lower())
            key = key.strip('_')
            metadata[key] = val.strip()

    return metadata

def find_channel_index(channels, prefix):
    """Find first channel that starts with the given prefix."""
    for i, channel in enumerate(channels):
        if channel.startswith(prefix):
            return i
    raise ValueError(f"No channel starting with '{prefix}' found")

def load_data(data_path=None, data_dir=None, DATA_NAME=None):
    target_path = data_path if data_path is not None else DATA_NAME
    if target_path is None:
        raise ValueError("A file path or file name must be provided to load_data.")
    if data_dir is not None:
        target_path = os.path.join(data_dir, target_path)

    data, metadata, parameters = load_3ds(target_path)
    if data is None:
        raise FileNotFoundError(f"Failed to load .3ds file at '{target_path}'.")

    map_index = find_channel_index(metadata['channels'], "LI Demod 1 X")
    topo_index = find_channel_index(metadata['channels'], "Z ")  
    bias_index = find_channel_index(metadata['channels'], "Bias ")
    x_index = find_channel_index(metadata['channels'], "X (m)")
    y_index = find_channel_index(metadata['channels'], "Y (m)")

    TOPO = data[:, :, 0, topo_index] * 1e9
    topo_std = np.std(TOPO, axis=1, keepdims=True)
    TOPO = (TOPO - np.mean(TOPO, axis=1, keepdims=True)) / np.where(topo_std == 0, 1.0, topo_std)

    DATA = data[:, :, :, map_index] * 1e9
    BIAS = data[0, 0, :, bias_index]
    X = data[0, :, 0, x_index]
    Y = data[:, 0, 0, y_index]
    print("X Span:", (X.max() - X.min()) * 1e9)
    print("Y Span:", (Y.max() - Y.min()) * 1e9)
    print("Voltage:", BIAS.min(), " to ", BIAS.max())

    return TOPO, DATA, BIAS, X, Y