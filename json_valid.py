import json
import sys

def validate_format(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return False

    # Check top-level structure
    if not isinstance(data, dict) or "messages" not in data:
        print("❌ JSON must be an object with a 'messages' field.")
        return False

    messages = data["messages"]
    if not isinstance(messages, list):
        print("❌ 'messages' must be a list.")
        return False

    # Validate each message
    for idx, msg in enumerate(messages, start=1):
        if not isinstance(msg, dict):
            print(f"❌ Message {idx} is not an object.")
            return False
        if "role" not in msg or "content" not in msg:
            print(f"❌ Message {idx} missing 'role' or 'content'.")
            return False

        role = msg["role"]
        content = msg["content"]

        # Content can be string or list
        if isinstance(content, str):
            continue
        elif isinstance(content, list):
            for c in content:
                if not isinstance(c, dict):
                    print(f"❌ Message {idx} content item is not an object.")
                    return False
                if c.get("type") != "image_url" or "image_url" not in c:
                    print(f"❌ Message {idx} content item must have type=image_url and an 'image_url' field.")
                    return False
                if not isinstance(c["image_url"], dict) or "url" not in c["image_url"]:
                    print(f"❌ Message {idx} content 'image_url' must contain a 'url'.")
                    return False
        else:
            print(f"❌ Message {idx} 'content' must be string or list.")
            return False

    # Check for user message with exactly 2 image_urls
    has_two_images = any(
        msg["role"] == "user"
        and isinstance(msg["content"], list)
        and sum(1 for c in msg["content"] if c.get("type") == "image_url") == 2
        for msg in messages
    )

    if not has_two_images:
        print("❌ No user message contains exactly 2 image_urls.")
        return False

    print("✅ JSON matches the required format!")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_json_format.py <file.json>")
    else:
        validate_format(sys.argv[1])
