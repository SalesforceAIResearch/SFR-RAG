import sys
import subprocess
import os

def execute_dataset_script(dataset_name):

    script_name = dataset_name.lower()
    script_filename = f"./Datasets/{script_name}/{script_name}.py"
    

    if not os.path.exists(script_filename):
        print(f"Error: Could not find a script named {script_filename} for the dataset '{dataset_name}'.")
        return
    
    try:
        print(f"Executing script for dataset: {dataset_name}")
        process = subprocess.Popen(['python', script_filename], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True, 
                                   bufsize=1, 
                                   universal_newlines=True)
        
        for line in process.stdout:
            print(line, end='')

        return_code = process.wait()
        
        if return_code != 0:
            print("Script encountered an error. Error output:")
            for line in process.stderr:
                print(line, end='')
    
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run.py <dataset_name>")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    execute_dataset_script(dataset_name)