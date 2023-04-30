import os
import json
import openai
openai.organization = os.getenv('OPENAI_ORGANIZATION')
openai.api_key = os.getenv('OPENAI_APIKEY')

AI = "\nAI:"
YU = "\nYou: "

def CreateResponse(message):

    # response = openai.Completion.create( # CompletionとChatCompletionは違う
    #     model="text-davinci-003",
    #     prompt=message,
    #     temperature=0.9,
    #     max_tokens=400,
    #     top_p=1,
    #     frequency_penalty=0,
    #     presence_penalty=0.6,
    #     stop=[" Human:", " AI:"]
    # )
    # return response['choices'][0]['text']

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user", "content": message}],
        temperature=0.9,  # 0-2で指定、数字が大きいほどアウトプットがランダム、
        max_tokens=4000,
        top_p = 0.1, # p値っぽい
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=[" Human:", " AI:"]
    )
    return response.choices[0].message['content']

### init ###
while 1:
    print(YU)
    message = input()
    if (message == "exit" or message == "quit"):
        exit()
    answer = CreateResponse(message)
    print(AI + answer)

