import os
import cv2
import json_repair
from PIL import Image

from planner.utils import parse_json
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="auto",
    api_key=os.environ['HF_TOKEN'],
)

# 读取图片
image_path = '../test/first_frame.jpg'
# Step 1: 加载图片
#image = Image.open(image_path).convert("RGB")
img = cv2.imread(image_path)

# completion = client.chat.completions.create(
#     model="Qwen/Qwen2.5-VL-72B-Instruct",
#     messages=[
#         {
#             "role": "system",
#             "content": [{"type": "text", "text": "You are a helpful assistant."}],
#         },
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "Describe this image in one sentence."
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
#                     }
#                 }
#             ]
#         }
#     ],
# )

def generate_message(task_name, vl_inputs):
    if task_name == "grasping_instruction_proposal":
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                            f"You are controlling a robotic arm that needs to complete the following user prompt: {vl_inputs['user_prompt']}\n"
                            f"I will show you two images. The initial image (before any actions) is:"
                        )
                    },
                    {"type": "image_url", "image_url": {"url": vl_inputs["images"]["initial_head_image"]}},
                    {"type": "text", "text": "The current image (after the latest action) is:"},
                    {"type": "image_url", "image_url": {"url": vl_inputs["images"]["current_head_image"]}},
                    {"type": "text", "text": (
                            f"Your task is to select the **best object to grasp next** from the current image.\n"
                            f"To identify objects, **use common sense and everyday knowledge** to infer what each item is.\n"
                            f"For example, recognize cups, bottles, fruits, snacks, boxes, tools, etc.\n"

                            f"When choosing the best object to grasp, follow these principles:\n"
                            f"1. Prefer objects on the right, then center, then left.\n"
                            f"2. Avoid objects that are blocked or surrounded.\n"
                            f"3. Avoid grasping objects that would cause other items to topple.\n"
                            f"4. Select objects that best match the user prompt.\n\n"

                            f"Please output ONLY ONE object that the robot should grasp next.\n"

                            f"Return format (in English, natural language):\n"
                            f"- A short sentence precisely describing the target object, including:\n"
                            f"- color\n"
                            f"- shape\n"
                            f"- relative position (e.g., \"on the right\", \"in front\", \"next to the red box\")\n"

                            f"Example:\n"
                            f"Grasp the blue cube on the right side of the table.\n"
                        )
                    }
                ]
            }
        ]

    elif task_name == "bounding_box_prediction":
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"You are a robotic vision assistant. Your task is to locate the object described below in the given image:"},
                    {"type": "image_url", "image_url": {"url": vl_inputs["images"]["current_head_image"]}},
                    {"type": "text", "text": (
                            f"and return its bounding box.\n\n"
                            f"Grasping instruction: {vl_inputs['grasping_instruction']}\n\n"

                            f"Instructions:\n"
                            f"1. Carefully read the grasping instruction and match the target object to the best-fitting visible object in the image.\n"
                            f"2. Select EXACTLY ONE object that best matches the description.\n"
                            f"3. For the selected object, return the following in strict JSON format:\n"
                            f"   - \"bbox_2d\": [x1, y1, x2, y2]  (integer pixel coordinates, top-left to bottom-right)\n"
                            f"   - \"label\": a short 2-4 word name (e.g., \"blue cup\")\n"
                            f"   - \"description\": a complete, natural-language description of the object's appearance and position\n\n"
                            
                            f"Requirements:\n"
                            f"- Only return one object.\n"
                            f"- Coordinates must be valid and within image boundaries.\n"
                            f"- Do not guess if the object is not visible"
                        )
                    }
                ]
            }
        ]

    elif task_name == "grasp_outcome_verification":
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "I will show you two images. The top-down view from the head camera is:"},
                    {"type": "image_url", "image_url": {"url": vl_inputs["images"]["current_head_image"]}},
                    {"type": "text", "text": "The close-up view from the wrist camera is:"},
                    {"type": "image_url", "image_url": {"url": vl_inputs["images"]["current_wrist_image"]}},
                    {"type": "text", "text": (
                            f"Grasping instruction: {vl_inputs['grasping_instruction']}\n\n"

                            f"Task:\n"
                            f"Determine whether the robotic arm has **successfully grasped the target object**.\n"

                            f"You should consider:\n"
                            f"- Whether the target object is still visible on the table.\n"
                            f"- Whether the object is securely held in the robotic hand.\n"

                            f"Output format: A reasoning and a boolean value (True=successfully grasped, False=not grasped).\n"
                            
                            f"Keep it short and simple.\n\n"
                        )
                    }
                ]
            }
        ]

    elif task_name == "prompt_completion_check":
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                            f"The robot is trying to complete the following user prompt: {vl_inputs['user_prompt']}\n"
                            f"I will show you two images. The initial image (before any actions) is:"
                        )
                    },
                    {"type": "image_url", "image_url": {"url": vl_inputs["images"]["initial_head_image"]}},
                    {"type": "text", "text": "The current image (after the latest action) is:"},
                    {"type": "image_url", "image_url": {"url": vl_inputs["images"]["current_head_image"]}},
                    {"type": "text", "text": (
                            f"Please compare the two images and determine whether the user prompt has been fully completed.\n"

                            f"Instructions:\n"
                            f"- Only consider visible 3D objects.\n"
                            f"- If all target objects have been removed or grasped, return True.\n"
                            f"- If some relevant objects remain, return False.\n"

                            f"Output format: A reasoning and a boolean value (True=completed, False=not completed).\n"
                            
                            f"Example:\n"
                            f"All blue objects have been removed from the table: True. \n"
                        )
                    }
                ]
            }
        ]

    else:
        raise ValueError(f"The task_name {task_name} is not a valid task name.")
    return messages

# completion = client.chat.completions.create(
#     model="Qwen/Qwen2.5-VL-32B-Instruct",
#     messages=[
#         {
#             "role": "system",
#             "content": [{"type": "text", "text": "You are a helpful assistant."}],
#         },
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "Please describe what is in the tray"
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": "https://huggingface.co/datasets/WintonChan/test/resolve/main/first_frame.jpg"
#                         #"url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
#                     }
#                 }
#             ]
#         }
#     ],
# )
# print(completion.choices[0].message)
vl_inputs={
    "images":{}
}
vl_inputs["images"]["initial_head_image"] = 'https://huggingface.co/datasets/WintonChan/test/resolve/main/first_frame.jpg'
vl_inputs["images"]["current_head_image"] = 'https://huggingface.co/datasets/WintonChan/test/resolve/main/first_frame.jpg'
vl_inputs["user_prompt"] = 'Remove all objects from the tray'
vl_inputs["grasping_instruction"] = 'Grasp the white object on the right side of the tray.'
#vl_inputs["grasping_instruction"] = 'Grasp the gray rectangular block on the right side of the tray.'
#vl_inputs["grasping_instruction"] = 'Grasp the cup on the left side of the tray.'

messages = generate_message('bounding_box_prediction', vl_inputs)
# completion = client.chat.completions.create(
#     model="Qwen/Qwen2.5-VL-32B-Instruct",
#     messages=messages
# )
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-32B-Instruct",
    messages=messages
)
response = completion.choices[0].message.content
bbox_str = parse_json(response)
bbox_json = json_repair.loads(bbox_str)

bbox = bbox_json['bbox_2d']
x1, y1, x2, y2 = bbox
# 画矩形：图像, 左上角, 右下角, 颜色(BGR), 线宽
cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
#print(bbox_str, bbox_json)
cv2.imshow("result", img); cv2.waitKey(0)