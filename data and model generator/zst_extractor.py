import os
import zstandard as zstd

def decompress_zst_to_csv(zst_file_path, output_folder):
    """
    Decompress a .zst file and save it as a .csv file in the output folder.

    :param zst_file_path: Path to the .zst file
    :param output_folder: Directory where the decompressed .csv file will be saved
    """
    file_name = os.path.basename(zst_file_path)
    csv_file_name = file_name.replace('.zst', '.csv')
    csv_file_path = os.path.join(output_folder, csv_file_name)

    dctx = zstd.ZstdDecompressor()
    with open(zst_file_path, 'rb') as zst_file, open(csv_file_path, 'wb') as csv_file:
        dctx.copy_stream(zst_file, csv_file)
    print(f'Decompressed {zst_file_path} to {csv_file_path}')

def decompress_all_zst_files(input_folder, output_folder):
    """
    Decompress all .zst files in the input folder and save them as .csv files in the output folder.

    :param input_folder: Directory containing the .zst files
    :param output_folder: Directory where the decompressed .csv files will be saved
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.zst'):
            zst_file_path = os.path.join(input_folder, file_name)
            decompress_zst_to_csv(zst_file_path, output_folder)

if __name__ == '__main__':
    input_folder = './futures_data/testing/zst'  # Replace with the path to your folder containing .zst files
    output_folder = './futures_data/testing/csv'  # Replace with the path where you want to save .csv files
    decompress_all_zst_files(input_folder, output_folder)
