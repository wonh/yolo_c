import urllib.request
import json
import cv2
import numpy as np
import requests


def receive():
    response = urllib.request.urlopen('http://192.168.0.80:8081/ai/pullPic')
    temp = json.load(response)
    # print(response.read().decode('utf-8'))
    print(type(temp), temp.keys(), temp.values())# [success, message, code, result, timestamp]
    patients_list = temp['result'] #病人列表
    cv2.namedWindow("test", )
    for item in patients_list:
        patient = item["patient"]  #病人的id
        pictures = item["pictures"] #病人的图片列表
        print("patient{} images:{}".format(patient["id"], len(pictures)))
        for img_dict in pictures:
            img_id = img_dict["picId"] #图片id
            img_url = img_dict["url"]  #图片地址
            img_name = img_dict["originName"]  #图片名字
            resp = urllib.request.urlopen('http://'+img_url)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            cv2.imshow("test", image)
            cv2.waitKey(100)


def send(content):
    dst_url = 'http://192.168.0.180:8081/ai/fallbackAi'
    headers = {'Content-type': 'application/json'}
    # with open(r'./result(1).json') as f:
    #     content = json.load(f)
    r = requests.post(dst_url, content,headers=headers)
    print(r.text)


if __name__ == "__main__":
    receive()

    c = {"images": [{"id": "图片id",  #图片id
         "net_Jam":       #net-jam推理结果，字典
             {"classification":  #不同net-jam模型的分类结果，列表
                  [
                      {"category": "classGate",  #单个模型的分类
                       "confidence": "",  # 单个模型分类的置信度
                       "detail": "",      #其他说明
                       "model_id": ""}    #模型名称
                   ],
              "confidence": "", #net-jam综合诊断置信度
              "main_class": "net大类", #net-jam综合诊大类
              "sub_class": "net小类", #net-jam综合诊断小类
             },
         "yolo":  #yolo 推理结果，字典
             {"category": #yolo 综合诊断类别
                 [
                     {"main_class": "",  #大类，（息肉，隆起性糜烂，溃疡等）
                      "sub_class": "",  #小类，（p1,p2,p3,p4等）
                      "cateCode": "",#6位编码，前三位为大类，后三类为小类
                      },
                  ],
                 "detections": [  #各个yolo模型详细检测结果
                     {"bbox": {"h": "", "w": "", "x": "", "y": ""}, #检测对象的坐标值，高度，宽度，中心点x坐标，中心点y坐标
                      "category": "",  #检测对象的类别
                      "confidence": "",  #检测对象的置信度
                      "detail": "",   #检测对象的其他说明
                      "model_id": "",  #检测模型名称
                      }
                 ]
             },
         "lmk":
             {
                 "confidence": "",
                 "main_class": "net大类",
                 "sub_class": "net小类",
                 "details": ""
             },
         "sort": '',   #图片序号
         "isReport": 0,  #是否选为报告图片
         "aiPicDetail": {    #图片的详细描述
             'clean': '',  #图片是否清晰
             'position': '',  #图片所处位置
             'result': '',  #病灶类型
             'detail': '',   #详细描述
         },
         }],
        "patientCode": "4654654165", #病人id
        "diagnose":{  #诊断信息
            "generay_situation": '',  #概况
            "clean_class": '',  #消化道洁净度评估
            "proposal": '',  #诊断建议
        },
    }
    content = json.dumps(c)
    send(content)

