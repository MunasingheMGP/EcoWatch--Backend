from pathlib import Path
import os
import laspy
import subprocess

BASE_DIR = Path(__file__).resolve().parent


def laz2las(input_laz):
    las2las_path = BASE_DIR/'LAStools/las2las.exe'
    laz_file = BASE_DIR/'USGS_LPC_SC_SavannahPeeDee_2019_B19_10491821.laz'

    if os.path.exists(las2las_path) == True:
        # Create command for las2las conversion
        # create the command string for las2las.exe
        command = ['"'+str(las2las_path)+'"']
        command.append("-cpu64")
        command.append("-i")
        command.append('"'+str(laz_file)+'"')
        command.append("-odir")
        lasOut = BASE_DIR  # output dir
        command.append('"'+str(lasOut)+'"')
        command.append("-olas")  # to convert to las

        command_length = len(command)
        command_string = str(command[0])
        command[0] = command[0].strip('"')

        for i in range(1, command_length):
            command_string = command_string + " " + str(command[i])
            command[i] = command[i].strip('"')

        # LAZ to LAS conversion
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, universal_newlines=True)
        process.wait()

        las_file_path = Path(
            BASE_DIR/'USGS_LPC_SC_SavannahPeeDee_2019_B19_10491821.las')
        if las_file_path.exists():
            las_file = laspy.read(las_file_path)
            return las_file
        else:
            raise Exception(f"File {las_file_path} not found")
    else:
        raise Exception("LAStools not found")


def las2tif(input_las):
    las2dem_path = BASE_DIR/'LAStools/las2dem.exe'
    las_file = BASE_DIR/'USGS_LPC_SC_SavannahPeeDee_2019_B19_10491821.las'

    if os.path.exists(las2dem_path) == True:
        # Create command for las2dem conversion
        # create the command string for las2dem.exe
        # command = ['"'+str(las2dem_path)+'"']
        # command.append("-cpu64")
        # command.append("-i")
        # command.append('"'+str(las_file)+'"')
        # command.append("-oasc")
        # command.append("-odir")
        # demOut = BASE_DIR  # output dir
        # command.append('"'+str(demOut)+'"')
        # command.append("-otif")  # to convert to tif

        # command_length = len(command)
        # command_string = str(command[0])
        # command[0] = command[0].strip('"')

        # for i in range(1, command_length):
        #     command_string = command_string + " " + str(command[i])
        #     command[i] = command[i].strip('"')

        # # LAS to TIF conversion
        # process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
        #                            stderr=subprocess.STDOUT, universal_newlines=True)
        # process.wait()

        tif_file_path = Path(
            BASE_DIR/'USGS_LPC_SC_SavannahPeeDee_2019_B19_10491821.tif')
        if tif_file_path.exists():
            return tif_file_path
        else:
            raise Exception(f"File {tif_file_path} not found")
    else:
        raise Exception("LAStools not found")
