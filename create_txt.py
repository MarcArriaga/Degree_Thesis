import pandas as pd
import os

'''
This code gets the path of each frame and saves it in a txt file with the yaw, pitch and roll 
angle and its x and y position.
'''

root = '/mnt/gpid07/users/morros/VFOA/TRAYECTORIAS2020_03_13'


dirs = os.listdir(root)
dires = []
for dire in dirs:
    if len(dire) == 20:
        dires.append(dire)


for directory in dires:
    print(directory)

    if directory == 'M2020-03-13-12-37-23' or directory == 'M2020-03-13-15-00-01' or directory == 'M2020-03-13-10-49-16' or directory == 'M2020-03-13-10-48-16':
        continue

    path = '/mnt/gpid07/users/morros/VFOA/TRAYECTORIAS2020_03_13/'+directory+'/angcomp.txt'
    path_img = '/mnt/gpid07/users/morros/VFOA/TRAYECTORIAS2020_03_13/'+directory+'/img/image_'
    rec = '/mnt/gpid07/users/morros/VFOA/TRAYECTORIAS2020_03_13/'+directory+'/parametrosyaw.txt'

    '''
    depth_f19.png - depth_f111.png 
    frame 20 - frame 96

    delay= 5,offset= 122.565452, nini= 202, nfin= 583

    '''

    try: 
        f = open(path, "r")
    except OSError:
        continue

    ang1 = []
    ang2 = []
    num = []
    try:
        r = open(rec,'r')

        for li in r:
            dif = int(li[li.find('= ')+2:li.find(',')])
    except OSError:
        dif = 0
    
    for line in f:
        ang1 = []
        if line.find('frame:'):
            ang1.append(line[line.find('frame:')+7:line.find(' Punto')])
            num.append(float(ang1[0]))
        if line.find('Punto:'):
            punto = line[line.find('Punto: [ ')+8:line.find(']')]
            ang1.append([float(punto[:punto.find(',')]), float(punto[punto.find(',')+1:])])
        if line.find('angleimu'):
            ang1.append(line[line.find('angleimu')+10:line.find(' , dif')])
        if line.find('angle2:'):
            ang1.append(line[line.find('angle2:')+8:line.find(' , angle3')])
        if line.find('angle3'):
            ang1.append(line[line.find('angle3 :')+8:line.find(' , Z')])
        ang2.append(ang1)


    df = pd.DataFrame(ang2, columns = ['frame', 'pos', 'angle1', 'angle2', 'angle3'])

    for data in range(len(df['angle1'])):
        if data+dif < len(df['angle1']):
            df['angle1'][data] = df['angle1'][data+dif] 
            df['angle2'][data] = df['angle2'][data+dif]
            df['angle3'][data] = df['angle3'][data+dif]
        else:
            df['angle1'][data] = 0
            df['angle2'][data] = 0
            df['angle3'][data] = 0

    print(df)

    paths=[]

    for rootdir, dirs, files in os.walk(path_img):
        for name in files:
            if name[0:7] == 'depth_f':
                paths.append(name)

    paths.sort()

    join = []

    for dire in ang2:
        join.append(path_img + dire[0]+'.png')

    with open('paths/{}.txt'.format(directory),"a") as t:
        for item in range(len(join)):
            if df['angle1'][item] != 0 or df['angle2'][item] != 0 or df['angle3'][item] != 0:
                t.write(join[item] + ' ' +str(df['angle1'][item]) + ' ' + str(df['angle2'][item])+' '+str(df['angle3'][item])+' '+str(df['pos'][item])+'\n')
            
    t.close()

