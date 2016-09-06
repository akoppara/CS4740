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
        print (open_file.read())

if __name__ == '__main__':
    grab_files()