import json
import time
import requests


class QianDuoDuo():
    def __init__(self, api_key):
        self.api_key = api_key

    def chat(self, messages, model, temperature, top_p, n, frequency_penalty, presence_penalty, retries=3):
        request_url = "https://api2.aigcbest.top/v1/chat/completions"
        headers = {
            'Accept': 'application/json',
            'Authorization': self.api_key,
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json'
        }
        data = json.dumps({
            "model": model.replace("-qianduoduo", ""),
            "messages": [
                {
                    "role": "user",
                    "content": messages
                }
            ],
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty
        })
        for attempt in range(retries):
            try:
                response = requests.request("POST", request_url, headers=headers, data=data).text
                break
            except Exception as e:
                print(e)
                time.sleep(1)
        return eval(response.replace("null", "None"))


if __name__ == "__main__":
    # 创建包含系统消息和历史会话的消息列表
    message = "你好，今天的天气怎么样？"
