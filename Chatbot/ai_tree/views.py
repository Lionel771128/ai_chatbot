from django.http import HttpResponseForbidden, HttpResponse
import json
# Create your views here.
import logging
import os

from django.http import HttpResponseForbidden, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    ImageMessage
)

secretFileContentJson=json.load(open("./ai_tree/line_secret_key",'r'))
server_url=secretFileContentJson.get("server_url_ngrok")
line_bot_api = LineBotApi(secretFileContentJson.get("channel_access_token"))
handler = WebhookHandler(secretFileContentJson.get("secret_key"))

@csrf_exempt
def callback(request):
    # get X-Line-Signature header value
    if request.method == 'POST':
        # signature = request.headers['X-Line-Signature']
        signature = request.META['HTTP_X_LINE_SIGNATURE']

        # get request body as text
        body = request.body.decode('utf-8')
        # 原本flask的app.logger.info("Request body: " + body)
        logging.debug("Request body: " + body)
        # handle webhook body
        try:
            handler.handle(body, signature)
        except InvalidSignatureError:
            # abort(400)
            return HttpResponseForbidden()
        return HttpResponse()

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
    ImagemapSendMessage, LocationSendMessage, FlexSendMessage
)

from linebot.models.template import *


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
    MessageEvent, ImageSendMessage)

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
    lineLinkMenuResponse = requests.post(linkMenuEndpoint, headers=linkMenuRequestHeader)
    # line_bot_api.link_rich_menu_to_user(event.source.user_id, rich_menu_id)
    # 去素材資料夾下，找abcd資料夾內的reply,json
    replyJsonPath = "./ai_tree/dynamic_reply/start/reply.json"
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
        replyJsonPath = './ai_tree/dynamic_reply/' + query_string_dict.get('folder')[0] + "/reply.json"
        result_message_array = detect_json_array_to_new_message_array(replyJsonPath)

        line_bot_api.reply_message(
            event.reply_token,
            result_message_array
        )
    elif 'menu' in query_string_dict:

        linkRichMenuId = open("./ai_tree/richmenu/" + query_string_dict.get('menu')[0] + '/rich_menu_id', 'r').read()
        line_bot_api.link_rich_menu_to_user(event.source.user_id, linkRichMenuId)

    elif 'model' in query_string_dict:
        model_type = query_string_dict.get('model')[0].split('_')[0]
        clr_type = query_string_dict.get('model')[0].split('_')[1]

        cameraQuickReplyButton = QuickReplyButton(
            action=CameraAction(label="カメラ"))
        cameraRollQRB = QuickReplyButton(
            action=CameraRollAction(label="写真"))
        quickReplyList = QuickReply(
            items=[cameraRollQRB, cameraQuickReplyButton])
        quickReplyTextSendMessage = TextSendMessage(text='写真をお送りください：' + \
                                                   ('樹形' if clr_type == 'tree' else '葉子')
                                                    , quick_reply=quickReplyList)
        line_bot_api.reply_message(
            event.reply_token,
            quickReplyTextSendMessage)

        @handler.add(MessageEvent, message=ImageMessage)
        def handle_image_message(event):
            # 取出消息內User的資料
            user_profile = line_bot_api.get_profile(event.source.user_id)

            # 將用戶資訊存在檔案內
            with open("./ai_tree/users.txt", "a") as myfile:
                myfile.write(json.dumps(vars(user_profile), sort_keys=True))
                myfile.write('\r\n')

            # 儲存圖片
            message_content = line_bot_api.get_message_content(event.message.id)
            with open('./static/images/' + event.message.id + '.jpg', 'wb') as fd:
                for chunk in message_content.iter_content():
                    fd.write(chunk)

            # model處理圖片
            input_path = "./static/images/" + event.message.id + ".jpg"
            conf_path = './ai_tree/ai_model/' + model_type + '/config_' + clr_type + '.json'
            if model_type == 'yolov3':
                output_path = "./static/images/yolov3_output/"
                image_url = 'https://' + server_url + '/static/images/yolov3_output/' + event.message.id + '.jpg'
                yolo_predict(conf_path, input_path, output_path, infer_models['yolov3_' + clr_type])
                reply = ImageSendMessage(
                        original_content_url=image_url,
                        preview_image_url=image_url)
            else:
                ans = clr_pred(input_path, conf_path, infer_models['clr_' + clr_type])
                reply = TextSendMessage(text='これは：' + ans + 'です')
            print(model_type + ('樹形' if clr_type == 'tree' else '葉') + "スタート")

            # 回覆文字消息與 回傳照片
            line_bot_api.reply_message(
                event.reply_token,
                [
                    TextSendMessage(text='物体検出の結果：' if model_type == 'yolov3' \
                                                        else '画像認識の結果：'+ \
                                        ('樹形' if clr_type == 'tree' else '葉')),
                    reply
                ]
            )

'''
pre-initiate yolov3 model
'''

import json
from ai_tree.ai_utils import preload_model, clr_pred, yolo_predict
'''
preload model
'''

mod_dir = './ai_tree/ai_model'
model_paths = {'yolov3_tree': mod_dir + '/yolov3/yolo_tree.h5',
              'yolov3_leaf': mod_dir + '/yolov3/yolo_leaf.h5',
              'clr_tree': mod_dir + '/clr/tree_MobileNetV2_acc797.h5',
              'clr_leaf': mod_dir + '/clr/leaf_InceptionV3_acc94.h5'}

infer_models = preload_model(model_paths)
