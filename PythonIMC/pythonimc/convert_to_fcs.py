import os
import pandas as pd
import numpy as np
import re
from flowio import create_fcs

def parse_channel_and_reagent(col):
    """
    From a column like '113In_CD235ab_CD61', extract:
    - Channel name: 'In113Di'
    - Reagent name: 'CD235ab_CD61' (i.e., everything after the first underscore)
    """
    match = re.match(r"(\d+)([A-Za-z]+)_(.+)", col)
    if match:
        mass, metal, target = match.groups()
        channel_name = f"{metal}{mass}Di"
        reagent = target  # CD235ab_CD61
        return channel_name, reagent, col  # channel, reagent, full label
    else:
        return col, col, col  # fallback

# def csv_to_fcs(csv_path, output_path):
#     df = pd.read_csv(csv_path)
#     data = df.to_numpy(dtype=np.float32)

#     channel_labels = df.columns.astype(str).tolist()
#     channel_info = [parse_channel_and_reagent(label) for label in channel_labels]

#     channel_names = [name for name, _, _ in channel_info]
#     channel_labels = [label for _, _, label in channel_info]

#     metadata = {
#         '$PAR': str(data.shape[1]),      # Number of parameters/channels
#         '$TOT': str(data.shape[0]),      # Number of events
#         '$CYT': 'CyTOF Helios',          # Instrument name
#         '$DATATYPE': 'F',                # Data type: F = float
#         '$MODE': 'L',                    # List mode
#         '$BYTEORD': '1,2,3,4',           # Byte order
#         '$FIL': os.path.basename(output_path),  # File name
#     }

#     for i, col in enumerate(df.columns.astype(str), start=1):
#         channel, reagent, label = parse_channel_and_reagent(col)

#         metadata[f'$P{i}N'] = channel        # e.g., In113Di
#         metadata[f'$P{i}S'] = reagent        # ✅ e.g., CD235ab_CD61
#         metadata[f'$P{i}D'] = label          # Optional full label
#         metadata[f'$P{i}B'] = '32'
#         metadata[f'$P{i}E'] = '0,0'
#         metadata[f'$P{i}R'] = '262144'
#         metadata[f'$P{i}G'] = '1'

#     event_data = data.flatten().tolist()

#     with open(output_path, 'wb') as f:
#         create_fcs(
#             file_handle=f,
#             event_data=event_data,
#             channel_names=channel_names,
#             opt_channel_names=channel_labels,
#             metadata_dict=metadata
#         )

#     print(f"✅ Saved FCS: {output_path}")



def csv_to_fcs(csv_path, output_path, randomize=True, noise_seed=42):
    # Load CSV
    df = pd.read_csv(csv_path)
    data = df.to_numpy(dtype=np.float32)

    # Optionally dequantize signals by adding uniform noise
    if randomize:
        np.random.seed(noise_seed)
        nonzero_mask = data > 0
        random_noise = np.random.uniform(0, 1, size=data.shape)
        data[nonzero_mask] = data[nonzero_mask] - 1 + random_noise[nonzero_mask]
        data = np.clip(data, a_min=0, a_max=None)

    # Prepare channel metadata
    channel_labels = df.columns.astype(str).tolist()
    channel_info = [parse_channel_and_reagent(label) for label in channel_labels]

    channel_names = [name for name, _, _ in channel_info]
    channel_labels = [label for _, _, label in channel_info]

    metadata = {
        '$PAR': str(data.shape[1]),
        '$TOT': str(data.shape[0]),
        '$CYT': 'CyTOF Helios',
        '$DATATYPE': 'F',
        '$MODE': 'L',
        '$BYTEORD': '1,2,3,4',
        '$FIL': os.path.basename(output_path),
    }

    for i, col in enumerate(df.columns.astype(str), start=1):
        channel, reagent, label = parse_channel_and_reagent(col)

        metadata[f'$P{i}N'] = channel
        metadata[f'$P{i}S'] = reagent
        metadata[f'$P{i}D'] = label
        metadata[f'$P{i}B'] = '32'
        metadata[f'$P{i}E'] = '0,0'
        metadata[f'$P{i}R'] = '262144'
        metadata[f'$P{i}G'] = '1'

    # Flatten event data row-wise
    event_data = data.flatten().tolist()

    # Write FCS file
    with open(output_path, 'wb') as f:
        create_fcs(
            file_handle=f,
            event_data=event_data,
            channel_names=channel_names,
            opt_channel_names=channel_labels,
            metadata_dict=metadata
        )

    print(f"✅ Saved FCS: {output_path}")



def batch_convert_to_fcs(base_dir):
    input_dir = os.path.join(base_dir, 'intensities')
    output_dir = os.path.join(base_dir, 'fcs_files')
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if fname.endswith(".csv"):
            csv_file = os.path.join(input_dir, fname)
            out_file = os.path.join(output_dir, fname.replace(".csv", ".fcs"))
            csv_to_fcs(csv_file, out_file)
