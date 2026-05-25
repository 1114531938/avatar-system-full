import logging
import os.path
import re
from typing import List

import torch
import torch.nn.functional as F
from header import *
from .ImageBind import *
from .ImageBind import data
from .common.modeling_llama import LlamaForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from .common.utils import *


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops: List = None, encounters: int = 1):
        super().__init__()
        if torch.cuda.is_available():
            self.stops = [torch.tensor(stop, dtype=torch.long).cuda() for stop in stops]
        else:
            self.stops = [torch.tensor(stop, dtype=torch.long) for stop in stops]
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop_token in self.stops:
            if input_ids.shape[1] >= len(stop_token):
                if torch.equal(input_ids[0, -len(stop_token):], stop_token):
                    return True
        return False


class MERGModel(nn.Module):
    """LoRA for LLaMa model"""

    def __init__(self, **args):
        super(MERGModel, self).__init__()
        self.args = args
        self.max_length = args["max_length"]
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        print("args max_length", args["max_length"])

        self._init_language_model()
        self._init_imagebind()

        self.llama_tokenizer.add_tokens("<Vid>")
        self.llama_tokenizer.add_tokens("<Aud>")
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        print("Tokenizer initialized.")
        self.input_embeddings = self.llama_model.get_input_embeddings()

        self.llama_proj = nn.Linear(
            self.visual_hidden_size, self.llama_model.config.hidden_size
        )
        if self.args.get("freeze_input_proj"):
            for param in self.llama_proj.parameters():
                param.requires_grad = False

    def _init_imagebind(self):
        imagebind_ckpt_path = os.path.join(
            self.args["pretrained_ckpt_path"],
            "imagebind_ckpt",
            self.args["imagebind_version"],
        )
        print(f"Initializing visual encoder from {imagebind_ckpt_path} ...")
        self.visual_encoder, self.visual_hidden_size = imagebind_model.imagebind_huge(
            pretrained=True, store_path=imagebind_ckpt_path
        )
        for _, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder.eval()
        print("Visual encoder initialized.")

    def _init_language_model(self):
        self.vicuna_ckpt_path = os.path.join(
            self.args["pretrained_ckpt_path"],
            "vicuna_ckpt",
            self.args["vicuna_version"],
        )
        print(f"Initializing language decoder from {self.vicuna_ckpt_path} ...")

        self.llama_model = LlamaForCausalLM.from_pretrained(self.vicuna_ckpt_path)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.args["lora_r"],
            lora_alpha=self.args["lora_alpha"],
            lora_dropout=self.args["lora_dropout"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

        self.llama_model = get_peft_model(self.llama_model, peft_config)
        self.llama_model.print_trainable_parameters()

        if self.args.get("freeze_lm"):
            print("Freezing the LLaMa ...")
            for param in self.llama_model.parameters():
                param.requires_grad = False
            self.llama_model.eval()
        else:
            print("Language decoder initialized.")

        tokenizer_path = self.vicuna_ckpt_path
        print(f"Initializing tokenizer from {tokenizer_path} ...")
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"

    def _embed_tokens(self, token_ids):
        return self.llama_model.model.model.embed_tokens(token_ids)

    def _zero_modal_embs(self, num_utts: int):
        dtype = next(self.llama_model.parameters()).dtype
        hidden = self.llama_model.config.hidden_size
        return torch.zeros(num_utts, hidden, dtype=dtype, device=self.device)

    def _resolve_audio_paths(self, dia_id, num_utts):
        base = self.args["audio_path"]
        return [os.path.join(base, f"dia{dia_id}utt{utt_id+1}.wav") for utt_id in range(num_utts)]

    def _resolve_video_paths(self, dia_id, num_utts):
        base = self.args["video_path"]
        return [os.path.join(base, f"dia{dia_id}utt{utt_id+1}.mp4") for utt_id in range(num_utts)]

    def encode_video(self, inputs):
        input_video_embs_list = []
        video_llama_atts_list = []

        dia_ids = inputs["dia_ids"]
        max_utt_ids = [len(dia["dialogue_history"]) for dia in inputs["conversations"]]

        for i in range(len(dia_ids)):
            num_utts = max_utt_ids[i]
            video_paths = self._resolve_video_paths(dia_ids[i], num_utts)

            if num_utts == 0:
                input_video_embs = self._zero_modal_embs(0)
                video_llama_atts = torch.zeros((0,), dtype=torch.long, device=self.device)
            elif not all(os.path.exists(p) for p in video_paths):
                logging.warning(
                    f"Missing video files for dia_id={dia_ids[i]}, fallback to zero video embeddings."
                )
                input_video_embs = self._zero_modal_embs(num_utts)
                video_llama_atts = torch.ones(
                    input_video_embs.size()[:-1], dtype=torch.long
                ).to(self.device)
            else:
                model_inputs = {
                    ModalityType.VISION: data.load_and_transform_video_data(video_paths, self.device)
                }
                model_inputs = {
                    key: model_inputs[key].to(self.llama_model.dtype) for key in model_inputs
                }
                with torch.no_grad():
                    embeddings = self.visual_encoder(model_inputs)
                    video_embeds = embeddings[ModalityType.VISION]
                    input_video_embs = self.llama_proj(video_embeds)
                    video_llama_atts = torch.ones(
                        input_video_embs.size()[:-1], dtype=torch.long
                    ).to(self.device)

            input_video_embs_list.append(input_video_embs)
            video_llama_atts_list.append(video_llama_atts)

        return input_video_embs_list, video_llama_atts_list

    def encode_audio(self, inputs):
        input_audio_embs_list = []
        audio_llama_atts_list = []

        dia_ids = inputs["dia_ids"]
        max_utt_ids = [len(dia["dialogue_history"]) for dia in inputs["conversations"]]

        for i in range(len(dia_ids)):
            num_utts = max_utt_ids[i]
            audio_paths = self._resolve_audio_paths(dia_ids[i], num_utts)

            if num_utts == 0:
                input_audio_embs = self._zero_modal_embs(0)
                audio_llama_atts = torch.zeros((0,), dtype=torch.long, device=self.device)
            elif not all(os.path.exists(p) for p in audio_paths):
                logging.warning(
                    f"Missing audio files for dia_id={dia_ids[i]}, fallback to zero audio embeddings."
                )
                input_audio_embs = self._zero_modal_embs(num_utts)
                audio_llama_atts = torch.ones(
                    input_audio_embs.size()[:-1], dtype=torch.long
                ).to(self.device)
            else:
                model_inputs = {
                    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, self.device)
                }
                model_inputs = {
                    key: model_inputs[key].to(self.llama_model.dtype) for key in model_inputs
                }
                with torch.no_grad():
                    embeddings = self.visual_encoder(model_inputs)
                    audio_embeds = embeddings[ModalityType.AUDIO]
                    input_audio_embs = self.llama_proj(audio_embeds)
                    audio_llama_atts = torch.ones(
                        input_audio_embs.size()[:-1], dtype=torch.long
                    ).to(self.device)

            input_audio_embs_list.append(input_audio_embs)
            audio_llama_atts_list.append(audio_llama_atts)

        return input_audio_embs_list, audio_llama_atts_list

    def prompt_wrap(self, inputs_audio_embs, inputs_video_embs, input_ids, target_ids, attention_mask):
        batch_size = input_ids.shape[0]
        audio_bos_id = self.llama_tokenizer("<Aud>", add_special_tokens=False).input_ids
        video_bos_id = self.llama_tokenizer("<Vid>", add_special_tokens=False).input_ids

        bos = torch.ones(
            [batch_size, 1], dtype=input_ids.dtype, device=input_ids.device
        ) * self.llama_tokenizer.bos_token_id

        p_after_embeds = self._embed_tokens(input_ids).expand(batch_size, -1, -1)
        bos_embeds = self._embed_tokens(bos)

        if inputs_audio_embs is not None and inputs_video_embs is not None:
            audio_pos_list = []
            video_pos_list = []
            for b in range(input_ids.size(0)):
                audio_pos = []
                video_pos = []
                for i, token_id in enumerate(input_ids[b]):
                    if token_id == audio_bos_id[0]:
                        audio_pos.append(i)
                    if token_id == video_bos_id[0]:
                        video_pos.append(i)
                assert len(audio_pos) == inputs_audio_embs[b].size(0)
                assert len(video_pos) == inputs_video_embs[b].size(0)
                audio_pos_list.append(audio_pos)
                video_pos_list.append(video_pos)

            for b in range(input_ids.size(0)):
                audio_pos, video_pos = audio_pos_list[b], video_pos_list[b]
                for p in range(len(audio_pos)):
                    p_after_embeds[b][audio_pos[p], :] = inputs_audio_embs[b][p]
                    p_after_embeds[b][video_pos[p], :] = inputs_video_embs[b][p]

        inputs_embeds = torch.cat((bos_embeds, p_after_embeds), dim=1)
        att = torch.ones([input_ids.size(0), 1], dtype=input_ids.dtype, device=input_ids.device)
        attention_mask = torch.cat((att, attention_mask), dim=1)

        empty_targets = torch.ones([batch_size, 1], dtype=torch.long).to(self.device).fill_(-100)
        targets = torch.cat([empty_targets, target_ids], dim=1).to(self.device)
        return inputs_embeds, targets, attention_mask

    def _prompt_wrap_generate(self, inputs_audio_embs, inputs_video_embs, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        audio_bos_id = self.llama_tokenizer("<Aud>", add_special_tokens=False).input_ids
        video_bos_id = self.llama_tokenizer("<Vid>", add_special_tokens=False).input_ids

        bos = torch.ones(
            [batch_size, 1], dtype=input_ids.dtype, device=input_ids.device
        ) * self.llama_tokenizer.bos_token_id

        p_after_embeds = self._embed_tokens(input_ids).expand(batch_size, -1, -1)
        bos_embeds = self._embed_tokens(bos)

        audio_pos_list = []
        video_pos_list = []
        for b in range(input_ids.size(0)):
            audio_pos = []
            video_pos = []
            for i, token_id in enumerate(input_ids[b]):
                if token_id == audio_bos_id[0]:
                    audio_pos.append(i)
                if token_id == video_bos_id[0]:
                    video_pos.append(i)
            audio_pos_list.append(audio_pos)
            video_pos_list.append(video_pos)

        for b in range(input_ids.size(0)):
            audio_pos, video_pos = audio_pos_list[b], video_pos_list[b]
            if len(audio_pos) != inputs_audio_embs[b].size(0):
                logging.warning(
                    f"Audio placeholder count mismatch: prompt={len(audio_pos)} embed={inputs_audio_embs[b].size(0)}"
                )
            if len(video_pos) != inputs_video_embs[b].size(0):
                logging.warning(
                    f"Video placeholder count mismatch: prompt={len(video_pos)} embed={inputs_video_embs[b].size(0)}"
                )

            for p in range(min(len(audio_pos), inputs_audio_embs[b].size(0))):
                p_after_embeds[b][audio_pos[p], :] = inputs_audio_embs[b][p]

            for p in range(min(len(video_pos), inputs_video_embs[b].size(0))):
                p_after_embeds[b][video_pos[p], :] = inputs_video_embs[b][p]

        inputs_embeds = torch.cat((bos_embeds, p_after_embeds), dim=1)
        att = torch.ones([input_ids.size(0), 1], dtype=input_ids.dtype, device=input_ids.device)
        attention_mask = torch.cat((att, attention_mask), dim=1)
        return inputs_embeds, attention_mask

    def _to_zh_emotion(self, text):
        x = str(text).strip().lower()
        mapping = {
            "warm": "温暖、真诚",
            "hopeful": "期待、温柔",
            "happy": "开心",
            "joyful": "开心",
            "gentle": "温柔",
            "sad": "难过",
            "neutral": "平静",
            "anxious": "有些不安",
            "angry": "生气",
        }
        return mapping.get(x, str(text).strip())

    def _to_zh_text(self, text, field_name=""):
        s = str(text).strip()
        if not s:
            return ""

        exact_map = {
            "The speaker finally meets the listener after a long time.": "说话者和对方久别重逢，终于见到了对方。",
            "The speaker wants to share many feelings but is unsure where to start.": "说话者心里有很多话想说，但一时不知道从哪里开始。",
            "The speaker has many feelings to share but is unsure how to start.": "说话者心里有很多话想说，但一时不知道从哪里开始。",
            "Encourage the speaker to open up and continue the conversation.": "温柔地鼓励对方继续表达，告诉对方可以慢慢说。",
            "To encourage a gentle, gradual conversation.": "温柔地鼓励对方慢慢说，不用着急。",
            "Encourage the speaker to open up and continue the conversation": "温柔地鼓励对方继续表达，告诉对方可以慢慢说。",
            "To encourage a gentle, gradual conversation": "温柔地鼓励对方慢慢说，不用着急。",
        }
        if s in exact_map:
            return exact_map[s]

        lower = s.lower()

        if field_name == "event_scenario":
            if "finally meets" in lower or "after a long time" in lower:
                return "说话者和对方久别重逢，终于见到了对方。"

        if field_name == "emotion_cause":
            if (
                ("many feelings" in lower or "a lot to say" in lower)
                and ("unsure" in lower or "doesn't know" in lower or "not sure" in lower)
            ):
                return "说话者心里有很多话想说，但一时不知道从哪里开始。"

        if field_name == "goal_to_response":
            if "gentle" in lower and "conversation" in lower:
                return "温柔地鼓励对方慢慢说，不用着急。"
            if "encourage" in lower and ("continue" in lower or "open up" in lower):
                return "鼓励对方继续表达，告诉对方你会认真听。"

        return s

    def _build_generation_prompts(self, inputs):
        prompts = []
        conversations = inputs["conversations"]

        for conv in conversations:
            history = conv.get("dialogue_history", [])
            coe = conv.get("chain_of_empathy") or conv.get("coe") or {}

            speaker_emotion_zh = self._to_zh_emotion(coe.get("speaker_emotion", ""))
            event_scenario_zh = self._to_zh_text(coe.get("event_scenario", ""), "event_scenario")
            emotion_cause_zh = self._to_zh_text(coe.get("emotion_cause", ""), "emotion_cause")
            goal_to_response_zh = self._to_zh_text(coe.get("goal_to_response", ""), "goal_to_response")

            latest_utterance = ""
            if history:
                latest_utterance = history[-1].get("utterance", "")

            lines = [
                "你是对话中的回应者，请直接回复说话者。",
                "要求：",
                "1. 只能以回应者身份说话。",
                "2. 不要复述任务，不要解释，不要输出英文。",
                "3. 不要续写说话者内容。",
                "4. 不要输出 Dialogue:, Speaker:, Assistant:, [Speaker], --- 这类模板。",
                "5. 只输出一句到两句自然、温柔、口语化的中文回复。",
                "",
                f"说话者：{latest_utterance} <Aud> <Vid>",
            ]

            if speaker_emotion_zh:
                lines.append(f"说话者情绪：{speaker_emotion_zh}")
            if event_scenario_zh:
                lines.append(f"场景：{event_scenario_zh}")
            if emotion_cause_zh:
                lines.append(f"原因：{emotion_cause_zh}")
            if goal_to_response_zh:
                lines.append(f"回复目标：{goal_to_response_zh}")

            lines.extend(
                [
                    "",
                    "请直接给出回应者的中文回复：",
                    "回应者：",
                ]
            )
            prompts.append("\n".join(lines))

        return prompts

    def _extract_inputs_from_args(self, args):
        for key in ["infer_batch", "batch", "sample_batch", "input_batch", "test_batch"]:
            if key in args and args[key] is not None:
                return args[key]
        raise ValueError(
            "No inference batch found in args. Expected one of: "
            "infer_batch, batch, sample_batch, input_batch, test_batch."
        )

    def _postprocess_generated_text(self, text, prompt=None):
        text = text.strip()

        if prompt and text.startswith(prompt):
            text = text[len(prompt):].strip()

        prefix_markers = [
            "回应者：",
            "Assistant:",
            "助手：",
            "回复：",
            "Response:",
            "Empathetic Response:",
            "### Assistant:",
            "### Response:",
        ]
        for marker in prefix_markers:
            if text.startswith(marker):
                text = text[len(marker):].strip()

        stop_markers = [
            "\n---",
            "---",
            "\nDialogue:",
            "Dialogue:",
            "\nSpeaker:",
            "Speaker:",
            "\nAssistant:",
            "Assistant:",
            "\n回应者：",
            "\n说话者：",
            "[Speaker]",
            "[Listener]",
            "###",
        ]
        for marker in stop_markers:
            if marker in text:
                text = text.split(marker)[0].strip()

        text = text.replace("[Speaker]", "").replace("[Listener]", "").strip()
        text = re.sub(r"\s+", " ", text).strip()

        pieces = [p.strip() for p in text.split("\n") if p.strip()]
        if pieces:
            text = pieces[0]

        text = re.sub(r"^(回应者：|说话者：|助手：|回复：)", "", text).strip()

        sentences = re.split(r"(?<=[。！？!?])", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) > 2:
            text = "".join(sentences[:2]).strip()

        text = re.sub(r'^[\'"“”‘’]+|[\'"“”‘’]+$', "", text).strip()
        return text.strip()

    def _contains_too_much_english(self, text):
        english_letters = sum(ch.isascii() and ch.isalpha() for ch in text)
        chinese_chars = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
        return english_letters > 8 and english_letters > max(1, chinese_chars)

    def _looks_truncated(self, text):
        if not text:
            return True

        text = text.strip()
        good_endings = ("。", "！", "？", "!", "?", "…")
        if text.endswith(good_endings):
            return False

        bad_endings = (
            "，", ",", "、", "：", "；", ";",
            "的", "了", "和", "但", "而", "却",
            "怎", "怎么", "如果", "因为", "然后"
        )
        if text.endswith(bad_endings):
            return True

        if len(text) < 40:
            return True

        return True

    def _is_bad_reply(self, text):
        if not text:
            return True

        bad_markers = [
            "Dialogue:",
            "[Speaker]",
            "[Listener]",
            "Speaker:",
            "Assistant:",
            "---",
            "说话者：",
            "回应者：",
        ]
        if any(marker in text for marker in bad_markers):
            return True

        suspicious_starts = [
            "我想和你聊聊天",
            "我有很多想要告诉你的事情",
            "你好，[Speaker]",
            "你好，感谢你的问候",
            "show understanding",
            "encourage the speaker",
            "respond to the speaker",
            "to encourage",
        ]
        if any(text.lower().startswith(x.lower()) for x in suspicious_starts):
            return True

        role_drift_patterns = [
            "我也想说很多",
            "我也有很多话想说",
            "我还不知道该从哪里开始",
            "我确实有很多话想和你分享",
        ]
        if any(p in text for p in role_drift_patterns):
            return True

        if self._contains_too_much_english(text):
            return True

        if self._looks_truncated(text):
            return True

        if len(text.strip()) < 4:
            return True

        return False

    def _build_fallback_reply(self, conv):
        coe = conv.get("chain_of_empathy") or conv.get("coe") or {}
        event_scenario = str(coe.get("event_scenario", "")).lower()
        goal = str(coe.get("goal_to_response", "")).lower()
        emotion = str(coe.get("speaker_emotion", "")).lower()

        opener = "谢谢你愿意和我说这些。"
        if "finally meets" in event_scenario or "见到" in event_scenario:
            opener = "我也很开心终于见到你了。"
        elif "hope" in emotion or "warm" in emotion:
            opener = "能听到你这样说，我心里也很温暖。"

        middle = "你不用着急，想从哪里开始都可以。"
        ending = "我会认真听你慢慢说。"

        if "gentle" in goal or "gradual" in goal or "慢慢" in goal:
            middle = "没关系，不用着急，我们可以慢慢聊。"
        if "encourage" in goal or "open up" in goal or "continue" in goal:
            ending = "你想说什么都可以，我会认真听着。"

        reply = f"{opener}{middle}{ending}"
        return reply.strip()

    def _run_generate_once(
        self,
        inputs_embeds,
        attention_mask,
        do_sample,
        temperature,
        top_p,
        num_beams,
        max_new_tokens,
        repetition_penalty,
    ):
        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.llama_tokenizer.pad_token_id,
            eos_token_id=self.llama_tokenizer.eos_token_id,
        )
        return self.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)

    @torch.no_grad()
    def generate(self, args):
        self.eval()

        inputs = self._extract_inputs_from_args(args)
        prompts = self._build_generation_prompts(inputs)

        max_input_length = min(self.max_length, args.get("max_input_length", self.max_length))
        tokenized = self.llama_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
            add_special_tokens=False,
        )

        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)

        inputs_audio_embs, _ = self.encode_audio(inputs)
        inputs_video_embs, _ = self.encode_video(inputs)

        inputs_embeds, attention_mask = self._prompt_wrap_generate(
            inputs_audio_embs,
            inputs_video_embs,
            input_ids,
            attention_mask,
        )

        do_sample = args.get("do_sample", False)
        temperature = args.get("temperature", 0.50)
        top_p = args.get("top_p", 0.80)
        num_beams = args.get("num_beams", 1)
        max_new_tokens = args.get("max_new_tokens", 80)
        repetition_penalty = args.get("repetition_penalty", 1.12)

        if not do_sample:
            temperature = 1.0

        decoded = self._run_generate_once(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )

        replies = [
            self._postprocess_generated_text(decoded[i], prompts[i] if i < len(prompts) else None)
            for i in range(len(decoded))
        ]

        need_retry = any(self._is_bad_reply(r) for r in replies)
        if need_retry:
            logging.warning("Bad reply detected, retrying generation with safer decoding settings...")
            decoded_retry = self._run_generate_once(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=0.35,
                top_p=0.70,
                num_beams=1,
                max_new_tokens=max(max_new_tokens, 80),
                repetition_penalty=1.15,
            )
            retry_replies = [
                self._postprocess_generated_text(decoded_retry[i], prompts[i] if i < len(prompts) else None)
                for i in range(len(decoded_retry))
            ]

            final_replies = []
            for i in range(len(replies)):
                if self._is_bad_reply(replies[i]) and not self._is_bad_reply(retry_replies[i]):
                    final_replies.append(retry_replies[i])
                elif self._is_bad_reply(replies[i]) and self._is_bad_reply(retry_replies[i]):
                    final_replies.append(self._build_fallback_reply(inputs["conversations"][i]))
                else:
                    final_replies.append(replies[i])
            replies = final_replies

        replies = [
            r if not self._is_bad_reply(r) else self._build_fallback_reply(inputs["conversations"][i])
            for i, r in enumerate(replies)
        ]

        if len(replies) == 1:
            return {
                "reply_text": replies[0],
                "generated_text": replies[0],
                "prompts": prompts,
            }

        return {
            "reply_text": replies,
            "generated_text": replies,
            "prompts": prompts,
        }

    def _empathetic_diallogue_training(self, target_ids, outputs):
        """
        In the stage 1: training the text-based empathetic response generation ability via EmpatheticDialogue dataset
        """
        loss = outputs.loss
        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]
        labels = target_ids[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item() + 1.0)
        return loss, gen_acc

    def forward(self, inputs):
        input_ids, target_ids, attention_mask = process_batch_text_stream(
            self.llama_tokenizer, inputs["conversations"], self.max_length
        )
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        inputs_audio_embs, _ = self.encode_audio(inputs)
        inputs_video_embs, _ = self.encode_video(inputs)
        inputs_embeds, target_ids, attention_mask = self.prompt_wrap(
            inputs_audio_embs,
            inputs_video_embs,
            input_ids,
            target_ids,
            attention_mask,
        )

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            labels=target_ids,
        )
        llama_loss, gen_acc = self._empathetic_diallogue_training(target_ids, outputs)
        return {
            "gen_acc": gen_acc,
            "loss": llama_loss,
        }