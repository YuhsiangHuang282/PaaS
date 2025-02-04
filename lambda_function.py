
import logging
import os
import boto3
import json
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch


os.environ['TORCH_HOME'] = '/tmp' 

# Initialize MTCNN and ResNet
mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) 
resnet = InceptionResnetV1(pretrained='vggface2').eval()


logger=logging.getLogger()
logger.setLevel(logging.INFO)
s3_client = boto3.client('s3')

data_pt_path='/tmp/data.pt'
data_bucket_name="datapt"
data_pt_key = 'data.pt'




def face_recognition_function(key_path):
    img = cv2.imread(key_path, cv2.IMREAD_COLOR)
    boxes, _ = mtcnn.detect(img)

    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    face, prob = mtcnn(img, return_prob=True, save_path=None)
    saved_data = torch.load(data_pt_path)  # loading data.pt file
    if face != None:
        emb = resnet(face.unsqueeze(0)).detach()  # detech is to make required gradient false
        embedding_list = saved_data[0]  # getting embedding data
        name_list = saved_data[1]  # getting list of names
        dist_list = []  # list of matched distances, minimum distance is used to identify the person
        for idx, emb_db in enumerate(embedding_list):
            dist = torch.dist(emb, emb_db).item()
            dist_list.append(dist)
        idx_min = dist_list.index(min(dist_list))

    
        return name_list[idx_min]
    else:
        logger.error(f"No face detected")


def handler(event, context):
    logging.info("Start!!!")
    logging.info(f"Received event: {event}")
    if not os.path.exists(data_pt_path):
        try:
            logger.info(f"Downloading data.pt from S3 bucket {data_bucket_name}")
            s3_client.download_file(data_bucket_name, data_pt_key, data_pt_path)
        except Exception as e:
            logger.error(f"Failed to download data.pt: {str(e)}")
            return {'statusCode': 500, 'body': json.dumps(f"Failed to download data.pt: {str(e)}")}
    
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    image_file_name = event['Records'][0]['s3']['object']['key']
    local_file_path = '/tmp/' + image_file_name

    try:
        logger.info(f"Downloading image {image_file_name} from S3 bucket {bucket_name}")
        s3_client.download_file(bucket_name, image_file_name, local_file_path)
    except Exception as e:
        logger.error(f"Failed to download image file: {str(e)}")
        return {'statusCode': 500, 'body': json.dumps(f"Failed to download image file: {str(e)}")}
    
    recognized_name = face_recognition_function(local_file_path)
    output_bucket = '1230040424-output'
    output_file_name = os.path.splitext(image_file_name)[0] + '.txt'

    try:
        logger.info(f"Uploading result {output_file_name} to S3 bucket {output_bucket}")
        s3_client.put_object(Body=recognized_name, Bucket=output_bucket, Key=output_file_name)
    except Exception as e:
        logger.error(f"Failed to upload result to output bucket: {str(e)}")
        return {'statusCode': 500, 'body': json.dumps(f"Failed to upload result to output bucket: {str(e)}")}

    return {'statusCode': 200, 'body': json.dumps('Face recognition completed successfully.')}
