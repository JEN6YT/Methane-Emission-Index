import pandas as pd
import os


def split_into_control_and_noncontrol(file_path):
    data = pd.read_csv(file_path)
    control_data = data[data['Controllable'] == 'C']
    control_indicator = control_data['Abbreviation'].tolist()
    uncontrol_data = data[data['Controllable'] == 'NC']
    uncontrol_indicator = uncontrol_data['Abbreviation'].tolist()
    return control_indicator, uncontrol_indicator

def split_country_into_control_and_noncontrol(
        country_data_folder_path, 
        control_indicator_list, 
        uncontrol_indicator_list, 
        country_control_path, 
        country_uncontrol_path):
    
    # make directory if not exist
    os.makedirs(country_control_path, exist_ok=True)
    os.makedirs(country_uncontrol_path, exist_ok=True)

    file_names = os.listdir(country_data_folder_path)
    for file in file_names:
        file_path = os.path.join(country_data_folder_path, file)
        df = pd.read_excel(file_path)

        control_columns = []
        uncontrol_columns = []
        # Separate columns into control and non-control DataFrames
        for c in control_indicator_list:
            for i in df.columns:
                if c in i:
                    control_columns.append(i)

        for uc in uncontrol_indicator_list:
            for i in df.columns:
                if uc in i:
                    uncontrol_columns.append(i)
        # control_columns = [c for c in df.columns if c in control_indicator_list]
        # uncontrol_columns = [c for c in df.columns if c in uncontrol_indicator_list]
        
        # print(len(control_columns) + len(uncontrol_columns))
        
        control_df = df[control_columns]
        uncontrol_df = df[uncontrol_columns]
        
        # Save the DataFrames to separate files
        try:
            control_output_path = os.path.join(country_control_path, f"control_{file}")
            uncontrol_output_path = os.path.join(country_uncontrol_path, f"uncontrol_{file}")
            
            control_df.to_excel(control_output_path, index=False)
            uncontrol_df.to_excel(uncontrol_output_path, index=False)
        except Exception as e:
            print(f"Error saving files for {file}: {e}")

        print(f"Done for {file}")
