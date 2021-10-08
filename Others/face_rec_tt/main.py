import torch
import os
import cv2
import pandas as pd
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import datasets
from facenet_pytorch.models.utils.detect_face import extract_face
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, jsonify


def extract_features(img):
    aligner = MTCNN(keep_all=True, thresholds=[0.6, 0.7, 0.9])
    bbs, _ = aligner.detect(img)
    if bbs is None:
        # if no face is detected
        return None, None
    facenet = InceptionResnetV1(pretrained='vggface2').eval()
    faces = torch.stack([extract_face(img, bb) for bb in bbs])
    embeddings = facenet(faces).detach().numpy()
    return bbs, embeddings


def load_data(path):
    dataset = datasets.ImageFolder(path)
    embeddings = []
    labels = []
    for img_path, label in dataset.samples:
        print(img_path)
        _, embedding = extract_features(dataset.loader(img_path))
        if embedding is None:
            print("Could not find face on {}".format(img_path))
            continue
        if embedding.shape[0] > 1:
            print("Multiple faces detected for {}, taking one with highest probability".format(img_path))
            embedding = embedding[0, :]
        embeddings.append(embedding.flatten())
        labels.append(label)
    return embeddings, labels, dataset.class_to_idx

def train(embeddings, labels):
    clf = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=1000, max_iter=10000)
    clf.fit(embeddings, labels)
    return clf

def recognise_faces(img):
    img = cv2.imread(img)
    embeddings, labels, class_to_idx = load_data('./imgs/')
    clf = train(embeddings, labels)
    bbs, embeddings = extract_features(img)
    if len(bbs) is None:
        return jsonify(error='No face on image')
    if len(bbs) > 1:
        return jsonify(error='More then 1 face')
    predictions = clf.predict_proba(embeddings)
    df = pd.Series(data=predictions[0], index=class_to_idx.keys())
    confidence = df.max()
    id = df.idxmax()
    if THRESHOLD > confidence:
        return jsonify(error='THRESHOLD more then confidence!', threshold=THRESHOLD)
    print(id, confidence)
    return jsonify(id=id, confidence=confidence, Count_of_faces=len(bbs), threshold=THRESHOLD)


app = Flask(__name__, template_folder='./')

UPLOAD_FOLDER = './upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
THRESHOLD= 0.5

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'there is no file1 in form!'
        file1 = request.files['file1']
        path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        file1.save(path)

        return recognise_faces(path)


    return '''
    <h1>Upload image for recognise</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file1">
      <input type="submit">
    </form>
    '''
app.run()

# Local WebCam option
#    cam = cv2.VideoCapture(0)
#    cv2.namedWindow("test")
#    img_counter = 0
#    while True:
#        ret, frame = cam.read()
#        if not ret:
#            print("failed to grab frame")
#            break
#        cv2.imshow("test", frame)

#        k = cv2.waitKey(1)
#        if k % 256 == 27:
#            # ESC pressed
#            print("Escape hit, closing...")
#            break
#        elif k % 256 == 32:
#            # SPACE pressed
#            img_name = "opencv_frame_{}.png".format(img_counter)
#            cv2.imwrite(img_name, frame)
#            print("{} written!".format(img_name))
#            img_counter += 1

#    cam.release()

#    cv2.destroyAllWindows()
#    print('./' + img_name)
#    pred_res = recognise_faces('./' + img_name)
#    print(pred_res)








