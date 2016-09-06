__author__ = 'Alex'
import os

def grab_files ():
    path = "data_corrected\classification task"
    for (path, dirs, files) in os.walk(path):
        if len(files):
            get_corpus(path, files)

def get_corpus (path, files):
    for file in files:
        file_path = path + '\\' + file
        open_file = open(file_path, 'r')
        file_string = open_file.read()
        header_ending_index = file_string.find('writes')
        #print(header_ending_index)
        if (header_ending_index != -1) :
            file_string = file_string[header_ending_index + 9:]
            print(file_string)
        elif (header_ending_index == -1) :
            header_ending_index = file_string.find('Subject')
            file_string = file_string[header_ending_index + 10:]
            print (file_string)

if __name__ == '__main__':
    grab_files()