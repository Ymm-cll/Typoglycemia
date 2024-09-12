import time
import requests


class SongShu():
    def __init__(self, api_key):
        self.api_key = api_key

    def chat(self, messages, model, retries=3):
        request_url = 'https://llmaiadmin-test.classba.cn/api/chat/call'
        headers = {
            'Content-Type': 'application/json',
            'authorization': self.api_key
        }
        data = {
            "name": model,
            "inputs": {
                "stream": False,
                "msg": str(messages),
            }
        }
        response = "None"
        for attempt in range(retries):
            response_temp = requests.post(request_url, headers=headers, json=data).json()
            response = str(response_temp["data"])
            if response != "{}":  # If response is not empty
                return response
            else:
                time.sleep(1)  # Optionally wait for a short time before retrying

        return str(response)  # Return the last response, even if it's empty


if __name__ == "__main__":
    # 创建包含系统消息和历史会话的消息列表
    message = "你好，今天的天气怎么样？"

