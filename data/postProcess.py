import json

with open("personachat_self_original.json", "r", encoding = "utf-8") as fi :
    data = json.load(fi)

filtered_data = {"train" : [], "valid" : []}

for i, sample in enumerate(data["train"]) :
    speaker = "person_1"

    d = {
        "conv_id" : i,
        "personality_person2" : sample["personality"],
        "dialog" : []
    }

    id = 0
    for text in sample["utterances"][-1]["history"] :
        d["dialog"].append({
            "id" : id, 
            "speaker" : speaker, 
            "text" : text
        })

        if speaker == "person_1" :
            speaker = "person_2"
        else :
            speaker = "person_1"

        id += 1

    filtered_data["train"].append(d)


for i, sample in enumerate(data["valid"]) :
    speaker = "person_1"

    d = {
        "conv_id" : i,
        "personality_person2" : sample["personality"],
        "dialog" : []
    }

    id = 0
    for text in sample["utterances"][-1]["history"] :
        d["dialog"].append({
            "id" : id, 
            "speaker" : speaker, 
            "text" : text
        })

        if speaker == "person_1" :
            speaker = "person_2"
        else :
            speaker = "person_1"

        id += 1

    filtered_data["valid"].append(d)

with open("persona_chat.json", "w", encoding = "utf-8") as fi :
    json.dump(filtered_data, fi, indent = 4)
    