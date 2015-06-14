#!/usr/bin/env python

import time
import sched
import urllib
import json
import sqlite3
import requests
import xml.etree.ElementTree as ET


def query_google(start, dest):
    output = requests.get("http://dev.virtualearth.net/REST/V1/Routes/Driving?o=xml&"
                      "wp.0=%s"
                      "&wp.1=%s"
                      "&key=AqzWtdm7sV9VxXRvoDPl-E2Y1McCAvUYtOzFqOWhpVY4Oku9lT86EuECAkmxO907"%(start, dest))

    root = ET.fromstring(output.content)

    try: 
    	return root[6][0][1][0][6].text

    except:
    	return 'error'




def get_commute_time(south_towns, north_towns, start_time): 
    
    #pairs of points on the map that I want to map between
    routes = [[0, 1],[1,2],[2,3],
              [3,4],[4,5],[5,6],[6,7],
              [6,8],[6,9],[9,10],[9,11],
              [9,12],[12,13],[12,14],[12,15],
              [12,16],[12,17],[16,18],
              [0,19],[19,20],[20,21],[21,22],
              [22,10],[22,23],[22,7],[22,11],[23,24],[23,15],[24,17],[24,18]]
    
    
    print 'start', time.strftime("%a %b %d %Y %H:%M:%S", time.localtime())
    
    day = time.strftime("%a", time.localtime())
    
    conn = sqlite3.connect('commute.db')
    cur = conn.cursor()
    for i in routes:
        duration =  query_google(south_towns[i[0]],south_towns[i[1]])
        cur.execute('INSERT INTO commutes (origin, destination, duration, day, time) \
        VALUES (?, ?, ?, ?, ?)', (i[0],i[1],duration,day,start_time) )

        #Now get the northbound commute
        duration =  query_google(north_towns[i[1]],north_towns[i[0]])
        cur.execute('INSERT INTO commutes (origin, destination, duration, day, time) \
        VALUES (?, ?, ?, ?, ?)', (i[1],i[0],duration,day,start_time) )
    conn.commit()
    conn.close()
  
 

def run_day(south_towns, north_towns):
    s = sched.scheduler(time.time, time.sleep)
    
    #14 hours in 10 minute intervals (measured in seconds)
    time_checks = range(0, 51000, 300)
    
    #create time stamps readable to humans for sql table
    loop_abs_times = range(1357020000, 1357071000, 300)
    loop_times=[]
    for i in loop_abs_times:
        loop_times.append(time.strftime("%H:%M", time.gmtime(i)))
    
    #checks the commute times every 10 minutes from 6am to 8pm
    for i in range(len(time_checks)):
        s.enter(time_checks[i], 1, get_commute_time, (south_towns, north_towns,loop_times[i]))
    s.run()




def get_commute_data():
	conn = sqlite3.connect('commute.db')
	cur = conn.cursor()
	# cur.execute('DROP TABLE IF EXISTS towns ')
	# cur.execute('DROP TABLE IF EXISTS commutes ')
	# cur.execute('CREATE TABLE IF NOT EXISTS towns (id INTEGER PRIMARY KEY, name TEXT UNIQUE)')
	# cur.execute('CREATE TABLE IF NOT EXISTS commutes \
	#             (origin INTEGER, destination INTEGER, \
	#             duration REAL, day TEXT, time TEXT)')

	south_towns = ['sanfrancisco', #0
	        '37.720889,-122.400318', #1    #vistacion valley 101
	        '37.656338,-122.406637', #2    #south san francisco 101
	        '37.584581,-122.329035', #burlingame 101
	        '37.540008,-122.283885', #san carlos 101
	        '37.488879,-122.212967', #redwood city 101
	        '37.484438,-122.184397', #marsh rd exit 101
	        'menlopark',
	        '37.482937,-122.150322', #1 hacker way, facebook
	        '37.460792,-122.141677', #university ave exit
	        'stanford',
	        'paloalto',
	        #'1500pagemillroad',\
	        '37.430953,-122.104366', #san antonio exit
	        '37.422031,-122.084320', #'1600AmphitheatreParkway',
	        'mountainview',
	        '37.388200,-122.060522', #'381EEvelynAveMountainView',
	        '37.399790,-122.032204', #'sunnyvale 101 
	        #'santa clara',\
	        '37.331847,-122.030750', #1 infinite loop, Apple
	        'sanjose',
	        '37.623086,-122.427762', #280 right after 380
	        '37.535302,-122.363144', # 280 before 92
	        '37.495348,-122.309264', #280 after 92
	        '37.436633,-122.244945', #280 by 84
	        '37.388519,-122.158825', #280 after page mill
	        '37.333559,-122.064036', #280 before cupertino]
	        ]
	north_towns = ['sanfrancisco', #0
        '37.720957,-122.400018', #1    #vistacion valley 101
        '37.656270,-122.406412', #2    #south san francisco 101

        '37.584700,-122.328874', #burlingame 101

        '37.539906,-122.283434', #san carlos 101
        '37.488794,-122.212077', #redwood city 101
        '37.481926,-122.177327', #marsh rd exit 101
        'menlopark',
        '37.482937,-122.150322', #1 hacker way, facebook
        '37.458314,-122.137064', #university ave exit
        'stanford',
        'paloalto',
        #'1500pagemillroad',\
        '37.425347,-122.098111', #san antonio exit
        '37.422031,-122.084320', #'1600AmphitheatreParkway',
        'mountainview',
        '37.388200,-122.060522', #'381EEvelynAveMountainView',
        '37.398205,-122.023954', #'sunnyvale 101 
        #'santa clara',\
        '37.331847,-122.030750', #1 infinite loop, Apple
        'sanjose',
        '37.621157,-122.426206', #280 right after 380
        '37.535438,-122.362661', # 280 before 92
        '37.495714,-122.309007', #280 after 92
        '37.431939,-122.238958', #280 by 84
        '37.388169,-122.155971', #280 after page mill
        '37.333388,-122.062749', #280 before cupertino]
        ]


	print "started"
	# for t in south_towns:
	#     cur.execute('INSERT INTO towns (name) VALUES (?)', (t,) )
	#     conn.commit()
	conn.close()
	#28 days (in seconds)
	days = range(0, 2419200, 86400)
	s = sched.scheduler(time.time, time.sleep)    
	for i in days:
	    s.enter(i, 1, run_day, (south_towns, north_towns))
	s.run() 



def main():
	check = 1
	while check:
		if time.strftime("%H", time.localtime()) == '06':
			print 'started'
			check == 0
			get_commute_data() 
		else:
			time.sleep(30)



if __name__ == "__main__":
    main()

     


