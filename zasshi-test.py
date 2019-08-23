import base64
import json
import requests
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from key import apikey#apikeyのインポート

#設定用変数
GOOGLE_CLOUD_VISION_API_URL = 'https://vision.googleapis.com/v1/images:annotate?key='
API_KEY=apikey()
#API_KEY="AIzaSyAgS6k914yuaRKEHWbODVVX_S_tIdxytpo"
path="/Users/hiraku/Desktop/andD/雑誌解析/image/test9.png"
api_url=GOOGLE_CLOUD_VISION_API_URL + API_KEY
res_file_path='/Users/hiraku/Desktop/andD/雑誌解析/results/result.txt'



#画像の取り込み
def img_to_base64(file_path):
    with open(file_path,'rb') as img:
        img_byte=img.read()
    return base64.b64encode(img_byte)

#cloud vision apiの使用
def request_cloud_vision_api(image_base64):
    request_body=json.dumps({
        'requests':[{
            'image':{
                'content':image_base64.decode('utf-8')
            },
            'features':[{
                'type':'DOCUMENT_TEXT_DETECTION',
                'maxResults':10#仮決め
            }]
        }]
    })
    res=requests.post(api_url,data=request_body)
    return res.json()

#ファイル出力の関数
def res_write(Res_file_path,data):
    with open(Res_file_path,'w') as res_f:
        res_f.write(data)

#----------------メイン-------------------------------------------


Img_base64=img_to_base64(path)

result=request_cloud_vision_api(Img_base64)
res_data1="{}".format(json.dumps(result,indent=4))

res_write(res_file_path,res_data1)#ファイルへjsonデータの書き出し

'''
testx1=result["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"][1]["paragraphs"][0]["boundingBox"]["vertices"][0]["x"]
testx2=result["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"][1]["paragraphs"][0]["boundingBox"]["vertices"][1]["x"]
testx3=result["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"][1]["paragraphs"][0]["boundingBox"]["vertices"][2]["x"]
testx4=result["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"][1]["paragraphs"][0]["boundingBox"]["vertices"][3]["x"]

testy1=result["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"][1]["paragraphs"][0]["boundingBox"]["vertices"][0]["y"]
testy2=result["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"][1]["paragraphs"][0]["boundingBox"]["vertices"][1]["y"]
testy3=result["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"][1]["paragraphs"][0]["boundingBox"]["vertices"][2]["y"]
testy4=result["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"][1]["paragraphs"][0]["boundingBox"]["vertices"][3]["y"]
'''
number_of_paragraphs=len(result["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"])


#座標変数の初期化
testx1=[i for i in range(0,number_of_paragraphs)]
testx2=[i for i in range(0,number_of_paragraphs)]
testx3=[i for i in range(0,number_of_paragraphs)]
testx4=[i for i in range(0,number_of_paragraphs)]

testy1=[i for i in range(0,number_of_paragraphs)]
testy2=[i for i in range(0,number_of_paragraphs)]
testy3=[i for i in range(0,number_of_paragraphs)]
testy4=[i for i in range(0,number_of_paragraphs)]

#座標値の抽出
for i in range(0,number_of_paragraphs):
    testx1[i]=result["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"][i]["paragraphs"][0]["boundingBox"]["vertices"][0]["x"]
    testx2[i]=result["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"][i]["paragraphs"][0]["boundingBox"]["vertices"][1]["x"]
    testx3[i]=result["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"][i]["paragraphs"][0]["boundingBox"]["vertices"][2]["x"]
    testx4[i]=result["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"][i]["paragraphs"][0]["boundingBox"]["vertices"][3]["x"]

    testy1[i]=result["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"][i]["paragraphs"][0]["boundingBox"]["vertices"][0]["y"]
    testy2[i]=result["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"][i]["paragraphs"][0]["boundingBox"]["vertices"][1]["y"]
    testy3[i]=result["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"][i]["paragraphs"][0]["boundingBox"]["vertices"][2]["y"]
    testy4[i]=result["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"][i]["paragraphs"][0]["boundingBox"]["vertices"][3]["y"]

    #print(i)


#print(testx1)#x2==x3


#---------------画像の表示
Show_image=Image.open(path)
image_list=np.asarray(Show_image)
for i in range(0,number_of_paragraphs):#blocksの数だけ枠を描画する
    cv2.rectangle(image_list, (testx1[i], testy1[i]), (testx2[i],testy3[i]),color=(255,255,0),thickness=3)#3つめの引数は色

plt.imshow(image_list)
plt.show()




#
