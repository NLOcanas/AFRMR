AFRMR-VideoStream.py
#Programmer: Nathaniel L Ocanas
#HEC-3 Advance Fire Rescue Mobile Robot
#Date: 2-21-2021 
## The purpose of this code is to send data and info back to the user/controller. This code 
## interfaces with a mounted video camera on the AFRMRs D3 Prototype. A bit stream from 
## sensors on an Arduino are written to a text file then used to display information on the
## AFRMR’s HUD. In this code we use cv2 libraries to create a numpy array image. From here #  we are able to overlay data and information from arduino onto the AFRMR’s HUD. 
##
 
#NOTE: This Code still needs to grab the bitstream data from the txt file to display our variables. 
# 2/21/2021 : Continuing progress for prototype code.
 
#  importing libraries
 
from gpiozero import CPUTemperature
import cv2
from utils import CFEVideoConf, image_resize
from time import time, ctime
import numpy as np
import serial
 
arduinoData = serial.Serial('/dev/ttyACM0',9600)
 
 
#Creating numpy array of video  capture to create variable "cap."
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)#frames per second read
#timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
 
# path to save video
save_path = 'images/RobotView.avi'
# video config and frame-rate
frames_per_seconds =fps
#print (fps)
#configure video on util.py file
config = CFEVideoConf(cap, filepath=save_path, res='720')
# save video
out = cv2.VideoWriter(save_path, config.video_type, frames_per_seconds,(640,480))# To save videos, size must be 640,480 on raspi
# Watermark Image-Tank
image_path='images/tank.jpg'
logo =cv2.imread(image_path, -1)
watermark = image_resize(logo, height = 50)
 
watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2BGRA)
#cv2.imshow('Advanced Mobile Robot',watermark) # show watermark on numpy plot
 
#Start Loop for capturing frames and laying images/data.
while(True):
    #Read data from arduino
     if(arduinoData.in_waiting>0):
         ardbit = arduinoData.readline()
         print(ardbit)
           
         file1= open("Ardtxt.txt","a")
         file1.write(str(ardbit))
         #file1.write("\n")
         file1.close()
     #Capture frame-by-frame
 
     ret, frame= cap.read()
     #taking frame shape to place info on screen with offsets
     frame_h, frame_w, frame_c = frame.shape
    # print(frame.shape)
    
     width_offset = frame_h-475 # Horizontal offset for display variables
     height_offset= frame_w-250 # Vertical offset for display variables
     #sensor variables
     cpu=CPUTemperature()
     cputemp= int(cpu.temperature)
     otempv=str(65) # Target temperature interim variable-- need to move into loop when reading
                              #variable
     rtempv=str(cputemp) # Internal CPU temperature interim variable-- need to move into loop
                                        #when reading variable
     itempv=str(75) # Internal Chassis temperature interim variable-- need to move into loop when 
                             #reading variable
 
     ULTS1=int(0) # Incoming ultrasonic  sensor 1 interim variable
     ULTS2=int(1) # Incoming ultrasonic sensor 2 interim variable
     vfound=str('VICTIM FOUND')
 
     if ret:
          
          # display variables TEXT
          
          otemp=str("Outside Temp:    F ") # Target Temp:   F --display
          rtemp=str("RasPi-CPU Temp:   F || Internal Temp:   F") # Internal Target Temp:   F --display
          
          t=ctime(time())# initial timestamp variable
          #print(t) # print time stamp
          ct=str(t) # change time stamp varible to string 
          
          ULTtxt=str("[ULT]:|Front|--|Back|")
          ULTma=str("DIST#") # PIR "ALERT" text. feed into ifloop to display text 
          clear=str("CLEAR")
          # font to display
          font = cv2.FONT_HERSHEY_PLAIN
          
         # display variables on stream putText(image, variable, displayplacement(y,x), Font, font   color(B,R,G), font size, ?linetype? not sure-can use 8)
          frame=cv2.putText(frame, otemp, (width_offset,height_offset), font, 1, (255,255,255), 1, cv2.LINE_4)# Target TEMP:     C--display
          frame=cv2.putText(frame, otempv, (width_offset+123,height_offset), font, 1, (255,255,255), 1, cv2.LINE_4)# Temp. sesnor variable
          frame=cv2.putText(frame, rtemp, (width_offset,height_offset+15), font, 1, (255,255,255), 1, cv2.LINE_4)# Internal TEMP:     C--display
          frame=cv2.putText(frame, rtempv, (width_offset+144,height_offset+15), font, 1, (255,255,255), 1, cv2.LINE_4)# Temp. sesnor variable
          frame=cv2.putText(frame, itempv, (width_offset+320,height_offset+15), font, 1, (255,255,255), 1, cv2.LINE_4)# Temp. sesnor variable
          frame=cv2.putText(frame, ct, (width_offset,height_offset+30), font, 1, (255,255,255), 1, cv2.LINE_4)# Date/Time stamp
          
          frame=cv2.putText(frame, ULTtxt, (width_offset,height_offset+45), font, 1, (60,255,10), 1, cv2.LINE_4)# PIR text LEFT FRONT RIGHT BACK
          #loop for PIR "ALERT" text display for each side
          if ULTS1!=0:
                frame=cv2.putText(frame, ULTma, (width_offset+50,height_offset+55), font, .75, (100,200,255), 1, cv2.LINE_4)# PIR text LEFT 
          else:
                frame=cv2.putText(frame, clear, (width_offset+50,height_offset+55), font, .75, (255,255,255), 1, cv2.LINE_4)# PIR text LEFT F
               
          if ULTS2!=0:
                frame=cv2.putText(frame, ULTma, (width_offset+127,height_offset+55), font, .75, (100,200,255), 1, cv2.LINE_4)# PIR text FRONT
          else:
                frame=cv2.putText(frame, clear, (width_offset+127,height_offset+55), font, .75, (255,255,255), 1, cv2.LINE_4)# PIR text FRONT 
       
          
          frame=cv2.putText(frame, vfound, (width_offset,height_offset+75), font, 1, (10,255,10), 2, cv2.LINE_4)# Victim Found
     #Convert frame to have 4 deminension channel Alpha
     frame=cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
     
 # Rectangular frame on video stream,*uncomment to use.    
##     #print(frame[50,150])
##     start_cord_x = 525
##     start_cord_y =375
##     color = (255, 0, 0) #BGR 0-255
##     stroke = 2
##     w=100
##     h= 100
##     end_cord_x = start_cord_x + w
##     end_cord_y = start_cord_y + h
##     cv2.rectangle(frame, (start_cord_x, start_cord_y ), (end_cord_x, end_cord_y), color, stroke)
##     print(frame[start_cord_x:end_cord_x, start_cord_y:end_cord_y])
     #print(frame.shape)
 
 
# Build overlay for water mark tank image   
     # overlay with 4 channels BGR and Alpha
     overlay = np.zeros((frame_h, frame_w, 4), dtype='uint8')
## Custommize overlay
     #overlay[100:250, 100:125]=(255,255,0, 1)#overly[y,x]=(B, G, R, A)
    # overlay[100:250, 150:255]=(0,255,0, 1)#overly[y,x]=(B, G, R, A)
     #overlay[start_y:end_y, start_x:end_x]=(B, G, R, A)
     #cv2.imshow("overlay",overlay)
    
# Getting watermark shape     
     watermark_h, watermark_w, watermark_c = watermark.shape
# for loop to equate overley pixels to watermark pixels
 
     for i in range(0, watermark_h):
          for j in range(0, watermark_w):
               #print(watermark[i,j])
               if watermark[i,j][3] !=0:
                    #watermark[i,j] # RBGA
                    h_offset = frame_h-watermark_h-10
                    w_offset= frame_w-watermark_w-10
                    overlay[h_offset+i, w_offset+j]= watermark[i,j]
                    
## Transparency to overlay                  
     cv2.addWeighted(overlay, .35 , frame, 1.0, 0, frame)
 
 
    #frame.addimage(watermark)
     frame=cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
     out.write(frame)
     # Display the resulting frame
     cv2.imshow('RescueRobotView',frame)
     if cv2.waitKey(20) & 0xFF == ord('q'):
          break
##END of while loop.##
     
#Release capture
cap.release()
out.release()
cv2.destroyAllWindows()

