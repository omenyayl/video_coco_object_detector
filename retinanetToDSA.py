#7/16/18 11:30
#Converts a bunch of retinanet classifications in a folder into one dataset annotator json file
import os
import json
import argparse

def getObjects(fileName):
    rectangles = []
    inFile = open(fileName, 'r')
    annotations = json.loads(inFile.read())
    for annotation in annotations:
        a = {}
        a.update({"topLeft": {"x": annotation['topLeft'][1], "y": annotation['topLeft'][0]}})
        a.update({"bottomRight": {"x": annotation['bottomRight'][1], "y": annotation['bottomRight'][0]}})
        a.update({"label": annotation['label'].split()[0]})
        rectangles += [a]
    return rectangles

def convertAll():
    frames = []
    for fileName in os.listdir(os.getcwd()):
        if(fileName[-5:] == ".json"):
            itemName = os.path.splitext(fileName)[0]+".jpg"
            frame = {"src": os.path.join(os.getcwd(), itemName)}
            frame.update({"lines": []})
            frame.update({"rectangles": getObjects(fileName)})
            frame.update({"polygons": []})
            frames += [frame]
    output = json.dumps({"frames": frames}, indent=4)
    return output

parser = argparse.ArgumentParser(description="This will convert a lot of retinanet annotation json files into one DSA json file")
parser.add_argument('--output', '-o', help="The name of the DSA json file to output to")
args = parser.parse_args()
data = convertAll()
outFileName = args.output
outFile = open(outFileName, 'w')
outFile.write(data)
outFile.close()
