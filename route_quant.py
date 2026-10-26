# EVOLVE-BLOCK-START

"""Routing pipeline: number-reading and counting questions go to Qwen3-VL-2B-Thinking, all others to -Instruct.

Thinking is clearly better on these question types (reserve[:128]: 24 vs. 16 of 51 correct; dev: 32 vs. 21 of 49),
while Instruct is as good or better elsewhere at ~50x less time. The type is recognized from the question text,
whose template is fixed per type (CharXiv descriptive questions). The two models are the manual versions
(`manual_instruct.py` step 2c, `manual_thinking.py` step 1), loaded lazily, both in GPU memory (~9 GB).
"""
import manual_instruct
import manual_thinking

# Template phrases of the number-reading / counting question types (CharXiv qids 8, 9, 10, 12, 14, 15, 17, 19).
_THINKING_PHRASES = (
    "difference between consecutive numerical tick values",           # 8, 9
    "how many lines are there",                                        # 10
    "how many discrete labels are there in the legend",                # 12
    "difference between the maximum and minimum values of the tick labels on the continuous legend",  # 14
    "maximum value of the tick labels on the continuous legend",       # 15
    "total number of explicitly labeled ticks across all axes",        # 17
    "what is the number of subplots",                                  # 19
)
last_raw_response = None  # the chosen model's raw output; recorded by evaluate.py
last_route = None


def needs_reasoning(question):
    first_line = question.split("\n")[0].lower()
    return any(phrase in first_line for phrase in _THINKING_PHRASES)


def vlm_inference(image_path, question="Describe this image in detail."):
    global last_raw_response, last_route
    model = manual_thinking if needs_reasoning(question) else manual_instruct
    answer = model.vlm_inference(image_path, question)
    last_raw_response, last_route = model.last_raw_response, model.__name__
    return answer

# EVOLVE-BLOCK-END
