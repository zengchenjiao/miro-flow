import requests

# 设置请求的 URL 和参数
url = "http://172.16.18.11:18081/api/questions/search?keyword=&created_from=2026-03-24T16%3A59%3A00.000Z&created_to=2026-03-25T15%3A59%3A00.000Z&page=1&page_size=10"

# 设置请求头
headers = {
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    "Connection": "keep-alive",
    "Referer": "http://172.16.18.11:18081/",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
    "authorization": "Bearer sk-qkIOZIWXfdBWlVxHr0M_VB",  # 使用新的 token
}

# 发送 GET 请求
response = requests.get(
    url, headers=headers, verify=False
)  # --insecure 选项对应 'verify=False'

# 打印响应内容
if response.status_code == 200:
    print(response.json())  # 假设响应内容是 JSON 格式
else:
    print(f"请求失败，状态码: {response.status_code}")
