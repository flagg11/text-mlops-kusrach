import time
import pandas as pd
from vk_api import VkApi
import os
from dotenv import load_dotenv

load_dotenv() 

TOKEN = os.getenv("VK_TOKEN")

POSTS_LIMIT = 3000
COMMENTS_PER_POST = 1000

vk = VkApi(token=TOKEN)
api = vk.get_api()

GROUPS = [
    (-50536362, "hodimvmore"),
    (-160361006, "rosmorport"),
    (-200990899, "volga.flot")
]

all_rows = []

def get_comments_full(api, group_id, group_name, max_comments=20000):
    rows = []
    offset = 0

    while True:
        wall = api.wall.get(owner_id=group_id, offset=offset, count=100)
        posts = wall["items"]
        if not posts:
            break

        for post in posts:
            post_id = post["id"]
            total_comments = post.get("comments", {}).get("count", 0)
            if total_comments == 0:
                continue

            comment_offset = 0
            while comment_offset < total_comments:
                resp = api.wall.getComments(
                    owner_id=group_id,
                    post_id=post_id,
                    offset=comment_offset,
                    count=100,
                    thread_items_count=10,
                    text_only=1
                )

                for c in resp["items"]:
                    if c["text"].strip():
                        rows.append({
                            "group_id": group_id,
                            "group_name": group_name,
                            "post_id": post_id,
                            "comment_text": c["text"]
                        })

                    for r in c.get("thread", {}).get("items", []):
                        if r["text"].strip():
                            rows.append({
                                "group_id": group_id,
                                "group_name": group_name,
                                "post_id": post_id,
                                "comment_text": r["text"]
                            })

                if len(resp["items"]) < 100:
                    break

                comment_offset += 100
                time.sleep(0.34)

                if len(rows) >= max_comments:
                    return rows

        offset += 100
        time.sleep(1)

    return rows


for gid, gname in GROUPS:
    print(f"Парсим {gname}")
    rows = get_comments_full(api, gid, gname, max_comments=3000)
    all_rows.extend(rows)

df = pd.DataFrame(all_rows)
df.to_csv("data/raw/comments.csv", index=False)

print("ИТОГО комментариев:", len(df))
