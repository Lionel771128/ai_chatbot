from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageMessage
)

import json

secretFileContentJson=json.load(open("./line_secret_key",'r'))
# server_url=secretFileContentJson.get("server_url")

server_url=secretFileContentJson.get("server_url_ngrok")

app = Flask(__name__,static_url_path = "/" , static_folder = "./")

line_bot_api = LineBotApi(secretFileContentJson.get("channel_access_token"))
handler = WebhookHandler(secretFileContentJson.get("secret_key"))


@app.route("/", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'


'''

消息判斷器

讀取指定的json檔案後，把json解析成不同格式的SendMessage

讀取檔案，
把內容轉換成json
將json轉換成消息
放回array中，並把array傳出。

'''

# 引用會用到的套件
from linebot.models import (
    ImagemapSendMessage, TextSendMessage, ImageSendMessage, LocationSendMessage, FlexSendMessage
)

from linebot.models.template import (
    ButtonsTemplate, CarouselTemplate, ConfirmTemplate, ImageCarouselTemplate

)

from linebot.models.template import *

import json


def detect_json_array_to_new_message_array(fileName):
    # 開啟檔案，轉成json
    with open(fileName, encoding='utf8') as f:
        jsonArray = json.load(f)

    # 解析json
    returnArray = []
    for jsonObject in jsonArray:

        # 讀取其用來判斷的元件
        message_type = jsonObject.get('type')

        # 轉換
        if message_type == 'text':
            returnArray.append(TextSendMessage.new_from_json_dict(jsonObject))
        elif message_type == 'imagemap':
            returnArray.append(ImagemapSendMessage.new_from_json_dict(jsonObject))
        elif message_type == 'template':
            returnArray.append(TemplateSendMessage.new_from_json_dict(jsonObject))
        elif message_type == 'image':
            returnArray.append(ImageSendMessage.new_from_json_dict(jsonObject))
        elif message_type == 'sticker':
            returnArray.append(StickerSendMessage.new_from_json_dict(jsonObject))
        elif message_type == 'audio':
            returnArray.append(AudioSendMessage.new_from_json_dict(jsonObject))
        elif message_type == 'location':
            returnArray.append(LocationSendMessage.new_from_json_dict(jsonObject))
        elif message_type == 'flex':
            returnArray.append(FlexSendMessage.new_from_json_dict(jsonObject))

            # 回傳
    return returnArray


'''
用戶follow事件
製作文字與圖片的教學訊息
'''
# 將消息模型，文字收取消息與文字寄發消息 引入
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageSendMessage
)

from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageSendMessage)

# 載入Follow事件
from linebot.models.events import (
    FollowEvent
)

# 載入requests套件
import requests


# 告知handler，如果收到FollowEvent，則做下面的方法處理
@handler.add(FollowEvent)
def reply_text_and_get_user_profile(event):
    # 取出消息內User的資料
    user_profile = line_bot_api.get_profile(event.source.user_id)

    # 將用戶資訊存在檔案內
    with open("./users.txt", "a") as myfile:
        myfile.write(json.dumps(vars(user_profile), sort_keys=True))
        myfile.write('\r\n')

    # 將菜單綁定在用戶身上
    linkRichMenuId = secretFileContentJson.get("rich_menu_id")
    linkMenuEndpoint = 'https://api.line.me/v2/bot/user/%s/richmenu/%s' % (event.source.user_id, linkRichMenuId)
    linkMenuRequestHeader = {'Content-Type': 'image/jpeg',
                             'Authorization': 'Bearer %s' % secretFileContentJson["channel_access_token"]}
    requests.post(linkMenuEndpoint, headers=linkMenuRequestHeader)

    # 去素材資料夾下，找abcd資料夾內的reply,json
    replyJsonPath = "./dynamic_reply/start/reply.json"
    result_message_array = detect_json_array_to_new_message_array(replyJsonPath)

    # 回覆文字消息與圖片消息
    line_bot_api.reply_message(
        event.reply_token,
        result_message_array
    )


'''

handler處理Postback Event

載入功能選單與啟動特殊功能

解析postback的data，並按照data欄位判斷處理

現有四個欄位
menu, folder, action, model

若folder欄位有值，則
    讀取其reply.json，轉譯成消息，並發送

若menu欄位有值，則
    讀取其rich_menu_id，並取得用戶id，將用戶與選單綁定
    讀取其reply.json，轉譯成消息，並發送

'''
from linebot.models import (
    PostbackEvent
)

from urllib.parse import parse_qs

from linebot.models import (CameraRollAction, CameraAction, QuickReplyButton, QuickReply)
from linebot.models import TextSendMessage


@handler.add(PostbackEvent)
def process_postback_event(event):
    user_profile = line_bot_api.get_profile(event.source.user_id)
    print(user_profile)

    # 解析data
    query_string_dict = parse_qs(event.postback.data)

    print(query_string_dict)
    # 在data欄位裡面有找到folder
    # folder=abcd&tag=xxx
    if 'folder' in query_string_dict:
        print(query_string_dict.get('folder')[0])
        result_message_array = []

        # 去素材資料夾下，找abcd資料夾內的reply,json
        replyJsonPath = 'dynamic_reply/' + query_string_dict.get('folder')[0] + "/reply.json"
        result_message_array = detect_json_array_to_new_message_array(replyJsonPath)

        line_bot_api.reply_message(
            event.reply_token,
            result_message_array
        )
    elif 'menu' in query_string_dict:

        linkRichMenuId = open("./richmenu/" + query_string_dict.get('menu')[0] + '/rich_menu_id', 'r').read()
        line_bot_api.link_rich_menu_to_user(event.source.user_id, linkRichMenuId)

    elif 'model' in query_string_dict:
        if query_string_dict.get('model')[0] == 'yolo_leaf':
            cameraQuickReplyButton = QuickReplyButton(
                action=CameraAction(label="立即拍照"))
            cameraRollQRB = QuickReplyButton(
                action=CameraRollAction(label="選擇照片"))
            quickReplyList = QuickReply(
                items=[cameraRollQRB, cameraQuickReplyButton])
            quickReplyTextSendMessage = TextSendMessage(text='選擇物件偵測來源：葉子', quick_reply=quickReplyList)
            line_bot_api.reply_message(
                event.reply_token,
                quickReplyTextSendMessage)

            @handler.add(MessageEvent, message=ImageMessage)
            def handle_image_message(event):
                # 取出消息內User的資料
                user_profile = line_bot_api.get_profile(event.source.user_id)

                # 將用戶資訊存在檔案內
                with open("./users.txt", "a") as myfile:
                    myfile.write(json.dumps(vars(user_profile), sort_keys=True))
                    myfile.write('\r\n')

                # 儲存圖片
                message_content = line_bot_api.get_message_content(event.message.id)
                with open('./images/' + event.message.id + '.jpg', 'wb') as fd:
                    for chunk in message_content.iter_content():
                        fd.write(chunk)

                # yolov3處理圖片
                input_path_leaf = "./images/" + event.message.id + ".jpg"
                yolo3expe_predict(config_path_leaf, input_path_leaf, output_path_leaf, yolomodel_leaf, graph_leaf)
                print("Yolo葉子啟動")

                # 回覆文字消息與 回傳照片
                line_bot_api.reply_message(
                    event.reply_token,
                    [
                        TextSendMessage(text='物件偵測結果：葉子'),
                        ImageSendMessage(
                            original_content_url='https://' + server_url + '/images/yolov3_output/' + event.message.id + '.jpg',
                            preview_image_url='https://' + server_url + '/images/yolov3_output/' + event.message.id + '.jpg')
                    ]
                )

        if query_string_dict.get('model')[0] == 'yolo_tree':
            cameraQuickReplyButton = QuickReplyButton(
                action=CameraAction(label="立即拍照"))
            cameraRollQRB = QuickReplyButton(
                action=CameraRollAction(label="選擇照片"))
            quickReplyList = QuickReply(
                items=[cameraRollQRB, cameraQuickReplyButton])
            quickReplyTextSendMessage = TextSendMessage(text='選擇物件偵測來源：樹形', quick_reply=quickReplyList)
            line_bot_api.reply_message(
                event.reply_token,
                quickReplyTextSendMessage)

            @handler.add(MessageEvent, message=ImageMessage)
            def handle_image_message(event):
                # 取出消息內User的資料
                user_profile = line_bot_api.get_profile(event.source.user_id)

                # 將用戶資訊存在檔案內
                with open("./users.txt", "a") as myfile:
                    myfile.write(json.dumps(vars(user_profile), sort_keys=True))
                    myfile.write('\r\n')

                # 儲存圖片
                message_content = line_bot_api.get_message_content(event.message.id)
                with open('./images/' + event.message.id + '.jpg', 'wb') as fd:
                    for chunk in message_content.iter_content():
                        fd.write(chunk)

                # yolov3處理圖片
                input_path_tree = "./images/" + event.message.id + ".jpg"
                yolo3expe_predict(config_path_tree, input_path_tree, output_path_tree, yolomodel_tree, graph_tree)
                print("Yolo樹形啟動")

                # 回覆文字消息與 回傳照片
                line_bot_api.reply_message(
                    event.reply_token,
                    [
                        TextSendMessage(text='物件偵測結果：樹形'),
                        ImageSendMessage(
                            original_content_url='https://' + server_url + '/images/yolov3_output/' + event.message.id + '.jpg',
                            preview_image_url='https://' + server_url + '/images/yolov3_output/' + event.message.id + '.jpg')
                    ]
                )

        if query_string_dict.get('model')[0] == 'cnn_tree':
            print("CNN樹")
            cameraQuickReplyButton = QuickReplyButton(
                action=CameraAction(label="立即拍照"))
            cameraRollQRB = QuickReplyButton(
                action=CameraRollAction(label="選擇照片"))
            quickReplyList = QuickReply(
                items=[cameraRollQRB, cameraQuickReplyButton])
            quickReplyTextSendMessage = TextSendMessage(text='選擇影像辨識來源：樹形', quick_reply=quickReplyList)
            line_bot_api.reply_message(
                event.reply_token,
                quickReplyTextSendMessage)

        if query_string_dict.get('model')[0] == 'cnn_leaf':
            print("CNN葉子")
            cameraQuickReplyButton = QuickReplyButton(
                action=CameraAction(label="立即拍照"))
            cameraRollQRB = QuickReplyButton(
                action=CameraRollAction(label="選擇照片"))
            quickReplyList = QuickReply(
                items=[cameraRollQRB, cameraQuickReplyButton])
            quickReplyTextSendMessage = TextSendMessage(text='選擇影像辨識來源：葉子', quick_reply=quickReplyList)
            line_bot_api.reply_message(
                event.reply_token,
                quickReplyTextSendMessage)


'''
pre-initiate yolov3 model
'''

import os
import json
import cv2
from keras.models import load_model
from tqdm import tqdm
import numpy as np
from yolov3_expe.utils.utils import get_yolo_boxes, makedirs
from yolov3_expe.utils.bbox import draw_boxes
import tensorflow as tf


def yolo3expe_load_model(c, o):
    config_path = c
    output_path = o

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    makedirs(output_path)

    ###############################
    #   Load the model
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    infer_model = load_model(config['train']['saved_weights_name'])


    graph = tf.get_default_graph()
    return infer_model, graph


'''
define config.json
define predict output folder
load model
'''

# config_path_leaf = "./config_leaf_1108.json"

config_path_leaf = "./config_leaf_1124.json"
output_path_leaf = "./images/yolov3_output/"

config_path_tree = "./config_tree_1120.json"
output_path_tree = "./images/yolov3_output/"

yolomodel_leaf, graph_leaf = yolo3expe_load_model(config_path_leaf, output_path_leaf)
yolomodel_tree, graph_tree = yolo3expe_load_model(config_path_tree, output_path_tree)

import os
import json
import cv2
from tqdm import tqdm
import numpy as np
from yolov3_expe.utils.utils import get_yolo_boxes, makedirs
from yolov3_expe.utils.bbox import draw_boxes


def yolo3expe_predict(c, i, o, m, graph):
    config_path = c
    input_path = i
    output_path = o
    infer_model = m

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    ###############################
    #   Set some parameter
    ###############################
    net_h, net_w = 416, 416  # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.5, 0.45

    ###############################
    #   Predict bounding boxes
    ###############################

    # do detection on an image or a set of images
    image_paths = []

    if os.path.isdir(input_path):
        for inp_file in os.listdir(input_path):
            image_paths += [input_path + inp_file]
    else:
        image_paths += [input_path]

    image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]

    # the main loop
    for image_path in image_paths:
        image = cv2.imread(image_path)
        print(image_path)

        # predict the bounding boxes
        boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh, graph)[0]

        print("model ready and run")

        # draw bounding boxes on the image using labels
        draw_boxes(image, boxes, config['model']['labels'], obj_thresh)

        # write the image with bounding boxes to file
        cv2.imwrite(output_path + image_path.split('/')[-1], np.uint8(image))


if __name__=="__main__":
    app.run(host='127.0.0.1', port='8000')
