import json
rand = 0
with open("/store/rsiva/code/navid_ws/NaVid-VLN-CE/tmp/gpt-dataset/navigation_training_data.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        try:
            sample = json.loads(line)
            content = sample.get("content", [])
            if len(content) != 3:  # 1 text + 2 images
                rand+=1
                #print(f"Line {i}: wrong number of content items ({len(content)})")
            for item in content:
                if item["type"] not in ["text", "image", "image_url"]:
                    print(f"Line {i}: invalid type {item['type']}")
                if item["type"] in ["image", "image_url"] and not item.get("image") and not item.get("image_url"):
                    print(f"Line {i}: missing image/base64")
        except json.JSONDecodeError as e:
            print(f"Line {i}: JSON error: {e}")
