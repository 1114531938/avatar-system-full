import argparse
import json
import os
import re
from pathlib import Path

from openai import OpenAI


SYSTEM_PROMPT = """
You are a strict JSON generator for an empathetic dialogue system.

Tasks:
1. Detect whether the user's utterance is Chinese or English.
2. Normalize punctuation while preserving the user's language.
3. Infer a semantic emotion label for empathetic dialogue understanding.
4. Infer a suitable responder profile.
5. Convert the input into Task1-style JSON.

STRICT RULES:
- Output valid JSON only.
- Do not output markdown.
- Do not output code fences.
- Do not output any explanation.
- Keep the original meaning unchanged.
- The utterance must stay in the user's language: Chinese input stays Chinese, English input stays English.
- speaker_emotion must reflect semantic tone, not merely copy acoustic emotion.
- response must always be an empty string.
- Follow the schema exactly.
"""


USER_PROMPT_TEMPLATE = """
Convert the following perception JSON into Task1-style JSON.

Input perception JSON:
{input_json}

Target schema:
{{
  "dia_ids": ["<utterance_id>"],
  "conversations": [
    {{
      "dialogue_history": [
        {{
          "index": 0,
          "role": "speaker",
          "utterance": "<normalized user utterance in the original language with punctuation>"
        }}
      ],
      "response": "",
      "coe": {{
        "speaker_emotion": "<semantic emotion label in English>",
        "event_scenario": "<one sentence in English>",
        "emotion_cause": "<one sentence in English>",
        "goal_to_response": "<one sentence in English>"
      }},
      "chain_of_empathy": {{
        "speaker_emotion": "<same as coe.speaker_emotion>",
        "event_scenario": "<same as coe.event_scenario>",
        "emotion_cause": "<same as coe.emotion_cause>",
        "goal_to_response": "<same as coe.goal_to_response>"
      }}
    }}
  ],
  "response_age": "<target responder age label>",
  "response_gender": "<target responder gender label>",
  "response_timbre": "<target responder timbre label>",
  "response_profile": {{
    "age": "<same as response_age>",
    "gender": "<same as response_gender>",
    "timbre": "<same as response_timbre>"
  }},
  "response_emotion": "<response style label>"
}}

Requirements:
- "utterance" must preserve the user's language. Do not translate English into Chinese or Chinese into English.
- Use the normalized text field if needed, but improve punctuation if it is awkward.
- If the acoustic emotion is "neutral" but the semantics are warm, caring, supportive, hopeful, gentle, or affectionate, use a semantic label like "warm", "supportive", "hopeful", or "gentle".
- Infer responder profile naturally from the utterance and conversation style.
- response_age / response_gender / response_timbre should be simple string labels.
- Use common labels such as:
  - response_age: "young", "middle-aged", "elderly"
  - response_gender: "female", "male"
  - response_timbre: "high", "mid", "low"
- response_emotion should be a natural response style label such as "warm", "gentle", "supportive", "calm".
- "coe" and "chain_of_empathy" must both exist and contain the same values.
- "response_profile" must exist and be consistent with response_age / response_gender / response_timbre.
- Keep English fields concise and natural.
- Return JSON only.
"""


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_first_json_block(text: str):
    text = strip_code_fence(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        return json.loads(candidate)

    raise ValueError("No valid JSON object found in model output.")


def first_value(v, default=None):
    if isinstance(v, list):
        return v[0] if v else default
    return default if v is None else v


def finalize_task1_result(
    obj: dict,
    response_age: str | None = None,
    response_gender: str | None = None,
    response_timbre: str | None = None,
    response_emotion: str | None = None,
):
    """
    把 LLM 输出或 fallback 输出修成更兼容 AvaMERG 的格式：
    - conversations[0].coe
    - conversations[0].chain_of_empathy
    - response_profile
    - response_age / response_gender / response_timbre / response_emotion
    """
    if not isinstance(obj, dict):
        raise ValueError("Output is not a JSON object.")

    if "dia_ids" not in obj or not isinstance(obj["dia_ids"], list) or not obj["dia_ids"]:
        obj["dia_ids"] = ["unknown"]

    if "conversations" not in obj or not isinstance(obj["conversations"], list) or not obj["conversations"]:
        obj["conversations"] = [
            {
                "dialogue_history": [
                    {
                        "index": 0,
                        "role": "speaker",
                        "utterance": "Hello."
                    }
                ],
                "response": "",
                "coe": {
                    "speaker_emotion": "neutral",
                    "event_scenario": "",
                    "emotion_cause": "",
                    "goal_to_response": ""
                }
            }
        ]

    conv = obj["conversations"][0]
    conv.setdefault("response", "")

    # coe / chain_of_empathy 双写兼容
    if "coe" in conv and "chain_of_empathy" not in conv:
        conv["chain_of_empathy"] = conv["coe"]
    elif "chain_of_empathy" in conv and "coe" not in conv:
        conv["coe"] = conv["chain_of_empathy"]
    elif "coe" not in conv and "chain_of_empathy" not in conv:
        conv["coe"] = {
            "speaker_emotion": "neutral",
            "event_scenario": "",
            "emotion_cause": "",
            "goal_to_response": ""
        }
        conv["chain_of_empathy"] = conv["coe"]

    # response_* 统一
    age = first_value(obj.get("response_age"), None)
    gender = first_value(obj.get("response_gender"), None)
    timbre = first_value(obj.get("response_timbre"), None)
    emo = first_value(obj.get("response_emotion"), None)

    if response_age is not None:
        age = age or response_age
    if response_gender is not None:
        gender = gender or response_gender
    if response_timbre is not None:
        timbre = timbre or response_timbre
    if response_emotion is not None:
        emo = emo or response_emotion

    age = age or "young"
    gender = gender or "female"
    timbre = timbre or "mid"
    emo = emo or "warm"

    response_profile = obj.get("response_profile")
    if not isinstance(response_profile, dict):
        response_profile = {}

    response_profile.setdefault("age", age)
    response_profile.setdefault("gender", gender)
    response_profile.setdefault("timbre", timbre)

    obj["response_age"] = response_profile["age"]
    obj["response_gender"] = response_profile["gender"]
    obj["response_timbre"] = response_profile["timbre"]
    obj["response_profile"] = response_profile
    obj["response_emotion"] = emo

    return obj


def validate_task1_schema(obj: dict):
    if not isinstance(obj, dict):
        raise ValueError("Output is not a JSON object.")

    required_top_keys = [
        "dia_ids",
        "conversations",
        "response_age",
        "response_gender",
        "response_timbre",
        "response_profile",
        "response_emotion",
    ]
    for key in required_top_keys:
        if key not in obj:
            raise ValueError(f"Missing top-level key: {key}")

    if not isinstance(obj["dia_ids"], list) or len(obj["dia_ids"]) == 0:
        raise ValueError("dia_ids must be a non-empty list.")

    if not isinstance(obj["conversations"], list) or len(obj["conversations"]) == 0:
        raise ValueError("conversations must be a non-empty list.")

    conv = obj["conversations"][0]
    for key in ["dialogue_history", "response", "coe", "chain_of_empathy"]:
        if key not in conv:
            raise ValueError(f"Missing conversation key: {key}")

    if not isinstance(conv["dialogue_history"], list) or len(conv["dialogue_history"]) == 0:
        raise ValueError("dialogue_history must be a non-empty list.")

    hist = conv["dialogue_history"][0]
    for key in ["index", "role", "utterance"]:
        if key not in hist:
            raise ValueError(f"Missing dialogue_history key: {key}")

    coe = conv["coe"]
    for key in ["speaker_emotion", "event_scenario", "emotion_cause", "goal_to_response"]:
        if key not in coe:
            raise ValueError(f"Missing coe key: {key}")

    ch = conv["chain_of_empathy"]
    for key in ["speaker_emotion", "event_scenario", "emotion_cause", "goal_to_response"]:
        if key not in ch:
            raise ValueError(f"Missing chain_of_empathy key: {key}")

    rp = obj["response_profile"]
    for key in ["age", "gender", "timbre"]:
        if key not in rp:
            raise ValueError(f"Missing response_profile key: {key}")


def build_messages(perception_obj: dict):
    user_prompt = USER_PROMPT_TEMPLATE.format(
        input_json=json.dumps(perception_obj, ensure_ascii=False, indent=2)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def call_llm(perception_obj: dict, model: str, base_url: str, api_key: str):
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    messages = build_messages(perception_obj)

    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=messages,
    )

    content = resp.choices[0].message.content
    result = extract_first_json_block(content)
    return result


def build_fallback_task1(
    perception_obj: dict,
    response_age: str = "young",
    response_gender: str = "female",
    response_timbre: str = "mid",
    response_emotion: str = "warm",
):
    utt_id = perception_obj.get("utterance_id", "unknown")
    text = (perception_obj.get("text") or perception_obj.get("text_raw") or "").strip()

    if not text:
        text = "Hello."

    coe = {
        "speaker_emotion": "neutral",
        "event_scenario": "The speaker is talking to the listener in a conversational context.",
        "emotion_cause": "The speaker is expressing thoughts or feelings to the listener.",
        "goal_to_response": "Respond naturally and empathetically to continue the conversation."
    }

    return {
        "dia_ids": [utt_id],
        "conversations": [
            {
                "dialogue_history": [
                    {
                        "index": 0,
                        "role": "speaker",
                        "utterance": text
                    }
                ],
                "response": "",
                "coe": coe,
                "chain_of_empathy": coe
            }
        ],
        "response_age": response_age,
        "response_gender": response_gender,
        "response_timbre": response_timbre,
        "response_profile": {
            "age": response_age,
            "gender": response_gender,
            "timbre": response_timbre
        },
        "response_emotion": response_emotion
    }


def get_output_path(out_arg: str, input_path: Path, perception_obj: dict) -> Path:
    out_input = Path(out_arg)

    if out_input.suffix.lower() == ".json":
        out_path = out_input.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path

    out_dir = out_input.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    utt_id = perception_obj.get("utterance_id")
    if not utt_id:
        utt_id = input_path.stem.replace("_perception", "")

    return out_dir / f"{utt_id}_task1_input.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True, help="path to perception json")
    parser.add_argument("--out_json", type=str, required=True, help="path to output json OR output directory")

    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("LLM_MODEL", "openai/gpt-oss-120b:free"),
        help="model name, e.g. openai/gpt-oss-120b:free",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
        help="OpenAI-compatible base url",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="API key for the OpenAI-compatible provider",
    )
    parser.add_argument(
        "--no_llm",
        action="store_true",
        help="disable llm conversion and use fallback only",
    )

    # 可选覆盖；默认不锁定
    parser.add_argument("--response_age", type=str, default=None)
    parser.add_argument("--response_gender", type=str, default=None)
    parser.add_argument("--response_timbre", type=str, default=None)
    parser.add_argument("--response_emotion", type=str, default=None)

    args = parser.parse_args()

    input_path = Path(args.input_json).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    perception_obj = load_json(input_path)
    out_path = get_output_path(args.out_json, input_path, perception_obj)

    if args.no_llm:
        result = build_fallback_task1(
            perception_obj=perception_obj,
            response_age=args.response_age or "young",
            response_gender=args.response_gender or "female",
            response_timbre=args.response_timbre or "mid",
            response_emotion=args.response_emotion or "warm",
        )
    else:
        if not args.api_key:
            raise ValueError("OPENAI_API_KEY is empty. Please set it in environment or pass --api_key.")
        try:
            result = call_llm(
                perception_obj=perception_obj,
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
            )

            result = finalize_task1_result(
                result,
                response_age=args.response_age,
                response_gender=args.response_gender,
                response_timbre=args.response_timbre,
                response_emotion=args.response_emotion,
            )
            validate_task1_schema(result)

        except Exception as e:
            print(f"[WARN] LLM conversion failed, fallback will be used. Error: {e}")
            result = build_fallback_task1(
                perception_obj=perception_obj,
                response_age=args.response_age or "young",
                response_gender=args.response_gender or "female",
                response_timbre=args.response_timbre or "mid",
                response_emotion=args.response_emotion or "warm",
            )

    save_json(result, out_path)
    print(f"Saved to: {out_path}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
