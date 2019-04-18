# necessary packages
import os
from tqdm import tqdm
import argparse
import cv2
import pickle
import face_recognition

# arguement parser
parser = argparse.ArgumentParser()
parser.add_argument("-m","--model",
         help="To use a the mode of encoding: 'cnn' or 'hog' ",
          default="hog")
parser.add_argument("-r","--root",
        help="Root folder to search for the faces dataset and store the encodings",
         default='dataset')
args = vars(parser.parse_args())
print('[DEBUG] init variables...')
# root paths
root_path = args['root']
face_encodings_path = os.path.join(
                    root_path, 'encodings')
face_path = os.path.join(root_path, 'faces')
names = os.listdir(face_path)

# init lists of known encodings
knownEncodings = []
knownNames = []

# loop over images
for name in names:
    print('Encoding name:', name)
    # :name: path
    name_path = os.path.join(face_path, name)

    # images in the :name: folders
    name_images = os.listdir(name_path)

    for name_image in tqdm(name_images):
        # path of :name: images
        image_path = os.path.join(name_path,
                            name_image)
        
        # reading and converting image from bgr to rgb
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image,
                        cv2.COLOR_BGR2RGB)

        # detecting face and
        # extracting bound box's (x,y) points
        boxes = face_recognition.face_locations(
            image_rgb, model=args['model'])

        # computes the facial embedding of the detected face
        encoding = face_recognition.face_encodings(
            image_rgb, boxes)

        # saving the encodings
        knownEncodings.append(encoding)
        knownNames.append(name)

# saving encodings
file_name = "encodings.pickle"
encoded_data = {
        "encodings" : knownEncodings,
        "names" : knownNames
                }
print(face_encodings_path)
f = open(os.path.join(
    face_encodings_path, file_name), "wb")
f.write(pickle.dumps(encoded_data))
f.close()