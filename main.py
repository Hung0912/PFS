import numpy as np
import csv
import os
from readImage import *
from picture_fuzzy_clustering import *
from picture_fuzzy_rules import *
from cropper import *

def main_menu():
    print("------------------MENU------------------")
    ans=True
    while ans:
        print ("""
        1. Picture fuzzy clustering
        2. Cropper
        3. PFC-PFR show contrast of input image
        4. PFC-PFR input image
        5. PFC-PFR test images and show accuratecy
        6. Exit/Quit
        """)
        ans= input("What would you like to do? ") 
        if ans=="1": 
            print("\n Picture fuzzy clustering")
            os.system('python picture_fuzzy_clustering.py')

        elif ans=="2":
            print("\n Cropper")
            cropper()  

        elif ans=="3":
            print("\n PFC-PFR show contrast of input image")
            input_path = input("Enter your image path: ") 
            main1(input_path)

        elif ans=="4":
            print("\n PFC-PFR input image") 
            input_path = input("Enter your image path: ") 
            main(input_path)

        elif ans=="5":
            print("\n PFC-PFR /images and show accuratecy") 
            main2()

        elif ans=="6":
            print("\n Goodbye.")
            ans = None
        elif ans !="":
            print("\n Not Valid Choice! Try again:") 
if __name__ == "__main__":
    
    main_menu()
