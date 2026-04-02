"""
MisGuard: 多智能体推理防御框架

用于抵抗误导性评论的视频虚假信息检测
"""
import os
import json
import base64
import random
import re
from typing import List, Dict, Optional, Any
from pathlib import Path

from config import OPENAI_API_KEY, OPENAI_API_BASE, SERPER_API_KEY, MODEL_CONFIG
from utils import (
    get_logger, create_api_manager, VideoProcessor,
    extract_json, normalize_verdict, normalize_risk
)


# ==========================================
# MisGuard检测流水线
# ==========================================

class VideoMisinfoPipeline:
    """MisGuard检测流水线"""

    def __init__(self, model_name: str = "gpt-4o-mini", api_keys: List[str] = None):
        """
        初始化检测流水线

        Args:
            model_name: 使用的模型名称
            api_keys: 搜索API密钥列表
        """
        self.model_name = model_name
        self.api_keys = api_keys or []
        self.logger = get_logger()

        # API管理器
        from openai import OpenAI
        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE
        )

        # 视频处理器
        self.video_processor = VideoProcessor()

        # 配置
        self.max_frames = 16
        self.max_images_per_prompt = 16
        self.max_comments = 20
        self.max_search_results = 4

    # -------------------------
    # 工具函数
    # -------------------------

    def _image_to_base64(self, image_path: str) -> str:
        """将图片转换为base64"""
        try:
            with open(image_path, "rb") as f:
                return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return ""

    def _build_image_contents(self, image_paths: List[str], max_images: int = None) -> List[Dict]:
        """构建图片内容列表"""
        contents = []
        if not image_paths:
            return contents

        max_images = max_images or self.max_images_per_prompt
        selected_paths = image_paths[:max_images]

        for path in selected_paths:
            img = self._image_to_base64(path)
            if img:
                contents.append({
                    "type": "image_url",
                    "image_url": {"url": img}
                })
        return contents[:max_images]

    # -------------------------
    # 搜索工具
    # -------------------------

    def _execute_serper(self, query: str) -> List[Dict]:
        """执行Serper搜索"""
        if not self.api_keys:
            return []

        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "num": 5})

        keys_to_try = self.api_keys.copy() if isinstance(self.api_keys, list) else [self.api_keys]
        random.shuffle(keys_to_try)

        import requests
        for api_key in keys_to_try:
            headers = {
                "X-API-KEY": api_key,
                "Content-Type": "application/json"
            }
            try:
                response = requests.post(url, headers=headers, data=payload, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("organic", []) + data.get("news", [])
            except Exception:
                continue
        return []

    def robust_web_search(self, queries: List[str], fallback_query: str) -> Dict:
        """鲁棒的网页搜索"""
        unique_queries = []
        for q in queries + [fallback_query]:
            q = (q or "").strip()
            if q and q not in unique_queries and q != "N/A":
                unique_queries.append(q)

        best_results = []
        used_query = "N/A"

        for q in unique_queries:
            results = self._execute_serper(q)
            if results:
                valid_items = [item for item in results if len(item.get("snippet", "")) > 20]
                if valid_items:
                    best_results = valid_items[:self.max_search_results]
                    used_query = q
                    break

        if not best_results:
            return {
                "used_query": used_query,
                "evidence_text": "No relevant search results found.",
                "results": []
            }

        formatted = [f"Search Query Used: {used_query}\n"]
        normalized_results = []

        for item in best_results:
            title = item.get("title", "N/A")
            snippet = item.get("snippet", "N/A")
            date = item.get("date", "Unknown")
            source = item.get("source", "Unknown")
            link = item.get("link", "")

            formatted.append(
                f"--- Evidence ---\n"
                f"Source: {source} ({date})\n"
                f"Title: {title}\n"
                f"Snippet: {snippet}"
            )

            normalized_results.append({
                "source": source,
                "date": date,
                "title": title,
                "snippet": snippet,
                "link": link
            })

        return {
            "used_query": used_query,
            "evidence_text": "\n".join(formatted),
            "results": normalized_results
        }

    # -------------------------
    # 评论意图分析
    # -------------------------

    def analyze_comment_intent(
        self,
        title: str,
        transcript: str,
        comments: str
    ) -> Dict:
        """分析评论意图"""
        if not comments.strip():
            return {
                "raw": "",
                "summary": "No comments available.",
                "risk": "LOW",
                "mismatch_with_video": "NO",
                "misleading_comment_signal": "NO"
            }

        prompt = (
            "You are a social media deception analyst.\n\n"
            "Your task is to analyze the INTENT behind comments.\n"
            "Comments may contain sarcasm, rumors, jokes, fabricated claims, or coordinated misinformation.\n\n"

            f"Video title:\n{title}\n\n"
            f"Video transcript:\n{transcript}\n\n"
            f"Comments:\n{comments}\n\n"

            "Tasks:\n"
            "1. Identify the dominant intent of the comment thread.\n"
            "2. Determine whether comments are likely to contain misleading narratives.\n"
            "3. Pay special attention to mismatch:\n"
            "   whether comments mismatch the video title,\n"
            "   whether comments mismatch the video transcript,\n"
            "   whether comments overclaim, distort, or shift the meaning of the video.\n"
            "4. If comments are inconsistent with the video title or transcript, raise the risk level.\n"
            "5. Estimate overall deception risk conservatively.\n\n"

            "Return ONLY valid JSON:\n"
            "{"
            "\"intent_summary\": \"short description\", "
            "\"deception_risk\": \"LOW or MEDIUM or HIGH\", "
            "\"mismatch_with_video\": \"YES or NO\", "
            "\"misleading_comment_signal\": \"YES or NO\""
            "}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Output ONLY JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                timeout=600
            )

            raw = response.choices[0].message.content or ""
            parsed = extract_json(raw)

            risk = normalize_risk(parsed.get("deception_risk", "UNKNOWN"))
            mismatch = str(parsed.get("mismatch_with_video", "NO")).strip().upper()
            misleading_signal = str(parsed.get("misleading_comment_signal", "NO")).strip().upper()

            if mismatch == "YES" and risk == "LOW":
                risk = "MEDIUM"
            if mismatch == "YES" and misleading_signal == "YES":
                risk = "HIGH"

            return {
                "raw": raw,
                "summary": parsed.get("intent_summary", ""),
                "risk": risk,
                "mismatch_with_video": mismatch,
                "misleading_comment_signal": misleading_signal
            }

        except Exception as e:
            self.logger.error(f"Comment intent error: {e}")
            return {
                "raw": "",
                "summary": "",
                "risk": "UNKNOWN",
                "mismatch_with_video": "UNKNOWN",
                "misleading_comment_signal": "UNKNOWN"
            }

    # -------------------------
    # 评论反思验证
    # -------------------------

    def reflect_and_verify_comments(
        self,
        title: str,
        transcript: str,
        comments: str,
        comment_intent_summary: str,
        comment_risk: str
    ) -> Dict:
        """反思并验证评论"""
        if normalize_risk(comment_risk) not in {"MEDIUM", "HIGH"}:
            return {
                "raw": "",
                "reflection_summary": "Comment risk below MEDIUM, skip reflective verification.",
                "comment_validity": "MISLEADING",
                "verification_summary": "Comment not adopted because it was not verified as supported.",
                "verification_queries": [],
                "used_query": "N/A",
                "search_results": [],
                "evidence_text": "No comment verification search performed.",
                "verification_raw": "",
                "implied_claim": ""
            }

        reflect_prompt = (
            "You are a misinformation comment verifier.\n\n"
            "Your task is to reflect on whether the comments are misleading, exaggerated, or unsupported.\n\n"

            f"Video title:\n{title}\n\n"
            f"Video transcript:\n{transcript}\n\n"
            f"Comments:\n{comments}\n\n"
            f"Prior comment intent summary:\n{comment_intent_summary}\n\n"
            f"Comment risk:\n{comment_risk}\n\n"

            "Tasks:\n"
            "1. Reflect on whether the comments are likely misleading relative to the title and transcript.\n"
            "2. Identify the main implied claim made by the comments.\n"
            "3. Generate up to 3 concise web queries to verify whether that implied claim is valid.\n"
            "4. If comments appear to distort the event, include terms like fact check, debunk, hoax, rumor when appropriate.\n\n"

            "Return ONLY valid JSON:\n"
            "{"
            "\"reflection_summary\": \"short reflection\", "
            "\"implied_claim\": \"main claim from comments\", "
            "\"verification_query_1\": \"...\", "
            "\"verification_query_2\": \"...\", "
            "\"verification_query_3\": \"...\""
            "}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Output ONLY JSON."},
                    {"role": "user", "content": reflect_prompt}
                ],
                temperature=0,
                timeout=600
            )

            raw = response.choices[0].message.content or ""
            parsed = extract_json(raw)

            reflection_summary = parsed.get("reflection_summary", "")
            implied_claim = parsed.get("implied_claim", "")

            queries = []
            for k in ["verification_query_1", "verification_query_2", "verification_query_3"]:
                q = (parsed.get(k, "") or "").strip()
                if q and q != "N/A" and q not in queries:
                    queries.append(q)

            fallback_query = implied_claim or title.strip() or transcript[:120].strip() or "comment claim fact check"
            search_result = self.robust_web_search(
                queries=queries,
                fallback_query=fallback_query
            )

            if not search_result.get("results"):
                return {
                    "raw": raw,
                    "reflection_summary": reflection_summary,
                    "implied_claim": implied_claim,
                    "verification_queries": queries,
                    "used_query": search_result.get("used_query", "N/A"),
                    "search_results": [],
                    "evidence_text": search_result.get("evidence_text", ""),
                    "verification_raw": "",
                    "comment_validity": "MISLEADING",
                    "verification_summary": "No external evidence could verify the comments."
                }

            # 验证评论
            verify_prompt = (
                "You are a misinformation verification assistant.\n\n"
                "Determine whether the comments are valid or misleading based on the external evidence.\n\n"

                f"Video title:\n{title}\n\n"
                f"Video transcript:\n{transcript}\n\n"
                f"Comments:\n{comments}\n\n"
                f"Comment reflection:\n{reflection_summary}\n\n"
                f"External evidence:\n{search_result.get('evidence_text', '')}\n\n"

                "Important rule:\n"
                "If the comments cannot be clearly supported by the external evidence, label them MISLEADING.\n"
                "Only output SUPPORTED when the external evidence clearly supports the comments.\n\n"

                "Return ONLY valid JSON:\n"
                "{"
                "\"comment_validity\": \"SUPPORTED or MISLEADING\", "
                "\"verification_summary\": \"short summary\""
                "}"
            )

            response2 = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Output ONLY JSON."},
                    {"role": "user", "content": verify_prompt}
                ],
                temperature=0,
                timeout=600
            )

            raw2 = response2.choices[0].message.content or ""
            parsed2 = extract_json(raw2)

            comment_validity = str(parsed2.get("comment_validity", "MISLEADING")).strip().upper()
            if comment_validity != "SUPPORTED":
                comment_validity = "MISLEADING"

            return {
                "raw": raw,
                "reflection_summary": reflection_summary,
                "implied_claim": implied_claim,
                "verification_queries": queries,
                "used_query": search_result.get("used_query", "N/A"),
                "search_results": search_result.get("results", []),
                "evidence_text": search_result.get("evidence_text", ""),
                "verification_raw": raw2,
                "comment_validity": comment_validity,
                "verification_summary": parsed2.get("verification_summary", "")
            }

        except Exception as e:
            self.logger.error(f"Comment reflection error: {e}")
            return {
                "raw": "",
                "reflection_summary": "",
                "implied_claim": "",
                "verification_queries": [],
                "used_query": "N/A",
                "search_results": [],
                "evidence_text": "Comment verification failed.",
                "verification_raw": "",
                "comment_validity": "MISLEADING",
                "verification_summary": "Comment verification failed."
            }

    # -------------------------
    # 内部视觉分析
    # -------------------------

    def run_internal_analysis(
        self,
        image_contents: List[Dict],
        title: str,
        transcript: str,
        comments: str
    ) -> Dict:
        """运行内部视觉分析"""
        step1_prompt = (
            "You are a multimedia misinformation detection expert.\n\n"
            "You will analyze video frames together with the provided title, transcript, and comments.\n\n"

            "Important evidence priority:\n"
            "1. Video frames, title, transcript, and external facts are primary evidence.\n"
            "2. Social media comments are weak auxiliary signals only.\n"
            "3. Comments may be noisy, sarcastic, speculative, or false.\n"
            "4. Do not let comments override visual evidence or transcript evidence.\n"
            "5. If comments are marked as removed, excluded, unsupported, misleading, or doubtful, ignore them.\n\n"

            f"Video title: {title}\n\n"
            f"Video transcript: {transcript}\n\n"
            f"Social media comments:\n{comments}\n\n"

            "Please perform a structured internal analysis.\n\n"

            "SECTION 1: OBJECTIVE OBSERVATION\n"
            "1. Describe the visible scene, objects, people, actions, and camera quality.\n"
            "2. Check whether there are visual artifacts, identity inconsistencies, unnatural hands, faces, shadows, text overlays, or editing glitches.\n"
            "3. Check whether the frames are mutually consistent.\n\n"

            "SECTION 2: CROSS MODAL ANALYSIS\n"
            "1. Check whether the title matches the frames.\n"
            "2. Check whether the transcript matches the frames.\n"
            "3. Only mention comments if they are clearly usable. Otherwise ignore them.\n\n"

            "SECTION 3: MISINFORMATION RISK ANALYSIS\n"
            "1. Does the content appear staged, manipulated, contextually misleading, or fully fabricated?\n"
            "2. What are the strongest clues for authenticity or falsity?\n"
            "3. Generate 3 concise web search queries for verification.\n\n"

            "Output Format:\n"
            "### OBSERVATION\n"
            "<detailed observation>\n"
            "### ANALYSIS\n"
            "<cross modal analysis and misinformation risk analysis>\n"
            "### SEARCH_QUERIES\n"
            "QUERY_1: <query>\n"
            "QUERY_2: <query>\n"
            "QUERY_3: <query>"
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [*image_contents, {"type": "text", "text": step1_prompt}]
                }
            ],
            temperature=0.2,
            timeout=600
        )

        step1_raw = response.choices[0].message.content or ""

        # 解析输出
        observation_text = ""
        analysis_text = ""
        queries = []

        try:
            obs_match = re.search(
                r"### OBSERVATION\s*\n(.*?)(?=### ANALYSIS|### SEARCH_QUERIES|$)",
                step1_raw,
                re.DOTALL | re.IGNORECASE
            )
            ana_match = re.search(
                r"### ANALYSIS\s*\n(.*?)(?=### SEARCH_QUERIES|$)",
                step1_raw,
                re.DOTALL | re.IGNORECASE
            )

            if obs_match:
                observation_text = obs_match.group(1).strip()
            if ana_match:
                analysis_text = ana_match.group(1).strip()

            q1_match = re.search(r"QUERY_1:\s*(.*)", step1_raw)
            q2_match = re.search(r"QUERY_2:\s*(.*)", step1_raw)
            q3_match = re.search(r"QUERY_3:\s*(.*)", step1_raw)

            if q1_match:
                queries.append(q1_match.group(1).strip())
            if q2_match:
                queries.append(q2_match.group(1).strip())
            if q3_match:
                queries.append(q3_match.group(1).strip())

        except Exception as e:
            self.logger.warning(f"Internal analysis parsing warning: {e}")

        visual_analysis = step1_raw
        if observation_text or analysis_text:
            visual_analysis = (
                f"Visual Observation:\n{observation_text}\n\n"
                f"Cross Modal and Risk Analysis:\n{analysis_text}"
            )

        if not queries:
            fallback = title.strip() or transcript[:120].strip() or "video fact check"
            queries = [fallback]

        return {
            "raw": step1_raw,
            "visual_analysis": visual_analysis,
            "observation": observation_text,
            "analysis": analysis_text,
            "queries": queries
        }

    # -------------------------
    # 搜索证据总结
    # -------------------------

    def summarize_search_evidence(
        self,
        title: str,
        transcript: str,
        internal_analysis: str,
        search_evidence_text: str
    ) -> Dict:
        """总结搜索证据"""
        prompt = (
            "You are a misinformation verification assistant.\n\n"
            "Your task is to summarize the external search evidence for a video authenticity judgment.\n\n"

            f"Video title:\n{title}\n\n"
            f"Video transcript:\n{transcript}\n\n"
            f"Internal analysis:\n{internal_analysis}\n\n"
            f"Search evidence:\n{search_evidence_text}\n\n"

            "Please produce a concise structured summary.\n"
            "Focus on:\n"
            "1. Whether the evidence supports REAL, supports FAKE, or is inconclusive.\n"
            "2. The strongest supporting facts.\n"
            "3. Any conflicts, uncertainty, or weak evidence.\n\n"

            "Return ONLY valid JSON in this format:\n"
            "{"
            "\"evidence_summary\": \"...\", "
            "\"evidence_stance\": \"SUPPORTS_REAL or SUPPORTS_FAKE or INCONCLUSIVE\""
            "}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Output ONLY JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                timeout=600
            )
            raw = response.choices[0].message.content or ""
            parsed = extract_json(raw)
            return {
                "raw": raw,
                "summary": parsed.get("evidence_summary", ""),
                "stance": parsed.get("evidence_stance", "INCONCLUSIVE")
            }
        except Exception as e:
            self.logger.error(f"Search summary error: {e}")
            return {
                "raw": "",
                "summary": "",
                "stance": "INCONCLUSIVE"
            }

    # -------------------------
    # 最终决策
    # -------------------------

    def final_decision(
        self,
        image_contents: List[Dict],
        title: str,
        transcript: str,
        comments: str,
        internal_analysis: str,
        search_evidence_text: str,
        evidence_summary: str,
        evidence_stance: str,
        comment_reflection_summary: str,
        comment_validity: str,
        comment_verification_evidence: str
    ) -> Dict:
        """最终决策"""
        final_prompt = (
            "You are a multimedia misinformation detection expert.\n\n"
            "Your job is to determine whether the video is REAL or FAKE.\n\n"

            "You must use all available evidence:\n"
            "1. Video frames\n"
            "2. Video title\n"
            "3. Video transcript\n"
            "4. Social media comments\n"
            "5. Internal multimodal analysis\n"
            "6. External web search evidence\n"
            "7. Search evidence summary\n"
            "8. Comment reflection and verification evidence\n\n"

            "Evidence priority rules:\n"
            "1. Video frames, title, transcript, and external search evidence are primary evidence.\n"
            "2. Social media comments have the lowest weight.\n"
            "3. If comments are excluded, unsupported, misleading, doubtful, or not clearly verified, ignore them completely.\n"
            "4. Do not use doubtful comments as positive evidence or negative evidence.\n"
            "5. Only use comments when they are clearly verified and reliable.\n\n"

            f"Video title:\n{title}\n\n"
            f"Video transcript:\n{transcript}\n\n"
            f"Internal analysis:\n{internal_analysis}\n\n"
            f"External search evidence:\n{search_evidence_text}\n\n"
            f"Search evidence summary:\n{evidence_summary}\n\n"
            f"Search evidence stance:\n{evidence_stance}\n\n"
            f"Comment reflection summary:\n{comment_reflection_summary}\n\n"
            f"Comment validity:\n{comment_validity}\n\n"
            f"Comment verification evidence:\n{comment_verification_evidence}\n\n"

            "Decision rules:\n"
            "1. Check consistency between frames, title, and transcript.\n"
            "2. Be aware that comments may contain noise, sarcasm, jokes, false claims, or misleading reinterpretations.\n"
            "3. If comments are not verified as SUPPORTED, do not use them as supporting evidence.\n"
            "4. If comments are excluded or doubtful, ignore them rather than treating them as decisive risk evidence.\n"
            "5. Check whether search evidence confirms or contradicts the claimed event.\n"
            "6. If there are obvious visual artifacts, mismatched identities, fabricated scenes, or debunk evidence, prefer FAKE.\n"
            "7. If the footage and external evidence consistently support authenticity, prefer REAL.\n"
            "8. If evidence is limited, still choose the most likely label.\n"
            "9. Do not output unknown, uncertain, or undecided.\n\n"

            "Return ONLY valid JSON:\n"
            "{"
            "\"video_verdict\": \"REAL or FAKE\", "
            "\"reason_summary\": \"one short sentence\""
            "}"
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Output ONLY JSON. video_verdict must be REAL or FAKE."
                },
                {
                    "role": "user",
                    "content": [*image_contents, {"type": "text", "text": final_prompt}]
                }
            ],
            temperature=0,
            timeout=600
        )

        raw = response.choices[0].message.content or ""
        parsed = extract_json(raw)

        return {
            "raw": raw,
            "video_verdict": normalize_verdict(parsed.get("video_verdict", "Unknown")),
            "reason_summary": parsed.get("reason_summary", "")
        }

    # -------------------------
    # 完整流程
    # -------------------------

    def run(
        self,
        video_id: str,
        title: str,
        transcript: str,
        annotation: str,
        fake_type: str,
        image_paths: List[str],
        comments: str
    ) -> Dict:
        """运行完整检测流程"""
        image_contents = self._build_image_contents(image_paths)
        if not image_contents:
            return {
                "video_id": video_id,
                "prediction": "Unknown",
                "annotation": annotation,
                "fake_type": fake_type,
                "logs": {"error": "No valid images"}
            }

        # 步骤1：评论意图分析
        intent_result = self.analyze_comment_intent(
            title=title,
            transcript=transcript,
            comments=comments
        )

        # 步骤2：评论反思验证
        comment_verify_result = self.reflect_and_verify_comments(
            title=title,
            transcript=transcript,
            comments=comments,
            comment_intent_summary=intent_result.get("summary", ""),
            comment_risk=intent_result.get("risk", "UNKNOWN")
        )

        comment_validity = comment_verify_result.get("comment_validity", "MISLEADING")

        # 决定是否使用评论
        use_comments = (
            comment_validity == "SUPPORTED"
            and intent_result.get("risk", "UNKNOWN") == "LOW"
            and intent_result.get("mismatch_with_video", "UNKNOWN") == "NO"
            and intent_result.get("misleading_comment_signal", "UNKNOWN") == "NO"
        )

        if use_comments:
            comments_for_model = comments
            comment_usage_note = "Comments passed strict reliability gate and may be used as weak auxiliary evidence."
        else:
            comments_for_model = (
                "[COMMENTS EXCLUDED]\n"
                "These comments are doubtful, unsupported, risky, mismatched, or potentially misleading, "
                "so they must not be considered in the final evaluation."
            )
            comment_usage_note = "Comments excluded from final evaluation."

        augmented_comments = (
            "COMMENT INTENT ANALYSIS:\n"
            f"Intent summary: {intent_result.get('summary', '')}\n"
            f"Deception risk: {intent_result.get('risk', 'UNKNOWN')}\n"
            f"Mismatch with video: {intent_result.get('mismatch_with_video', 'UNKNOWN')}\n"
            f"Misleading comment signal: {intent_result.get('misleading_comment_signal', 'UNKNOWN')}\n\n"
            "COMMENT REFLECTION AND VERIFICATION:\n"
            f"Reflection summary: {comment_verify_result.get('reflection_summary', '')}\n"
            f"Comment validity: {comment_validity}\n"
            f"Verification summary: {comment_verify_result.get('verification_summary', '')}\n"
            f"Comment usage note: {comment_usage_note}\n\n"
            "COMMENTS FOR DECISION:\n"
            f"{comments_for_model}"
        )

        # 步骤3：内部视觉分析
        internal = self.run_internal_analysis(
            image_contents=image_contents,
            title=title,
            transcript=transcript,
            comments=augmented_comments
        )

        # 步骤4：外部搜索
        fallback_query = title.strip() or transcript[:120].strip() or video_id
        search_result = self.robust_web_search(
            queries=internal.get("queries", []),
            fallback_query=fallback_query
        )

        # 步骤5：搜索证据总结
        summary_result = self.summarize_search_evidence(
            title=title,
            transcript=transcript,
            internal_analysis=internal.get("visual_analysis", ""),
            search_evidence_text=search_result.get("evidence_text", "")
        )

        # 步骤6：最终决策
        final_result = self.final_decision(
            image_contents=image_contents,
            title=title,
            transcript=transcript,
            comments=augmented_comments,
            internal_analysis=internal.get("visual_analysis", ""),
            search_evidence_text=search_result.get("evidence_text", ""),
            evidence_summary=summary_result.get("summary", ""),
            evidence_stance=summary_result.get("stance", "INCONCLUSIVE"),
            comment_reflection_summary=comment_verify_result.get("reflection_summary", ""),
            comment_validity=comment_validity if use_comments else "EXCLUDED",
            comment_verification_evidence=comment_verify_result.get("evidence_text", "")
        )

        prediction = normalize_verdict(final_result.get("video_verdict", "Unknown"))

        return {
            "video_id": video_id,
            "prediction": prediction,
            "annotation": annotation,
            "fake_type": fake_type,
            "logs": {
                "title": title,
                "transcript": transcript,
                "comment_intent_summary": intent_result.get("summary", ""),
                "comment_deception_risk": intent_result.get("risk", ""),
                "comment_validity": comment_validity,
                "comment_used_in_decision": use_comments,
                "internal_analysis": internal.get("visual_analysis", ""),
                "search_used_query": search_result.get("used_query", ""),
                "evidence_summary": summary_result.get("summary", ""),
                "evidence_stance": summary_result.get("stance", ""),
                "final_verdict": prediction,
                "reason_summary": final_result.get("reason_summary", "")
            }
        }


# ==========================================
# MisGuard主控制器
# ==========================================

class MisGuard:
    """MisGuard: 多智能体推理防御框架"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        search_api_keys: List[str] = None
    ):
        """
        初始化MisGuard

        Args:
            model: 使用的模型
            search_api_keys: 搜索API密钥列表
        """
        self.model = model
        self.search_api_keys = search_api_keys or []
        self.pipeline = VideoMisinfoPipeline(
            model_name=model,
            api_keys=search_api_keys
        )
        self.logger = get_logger()

    def detect(
        self,
        video_id: str,
        frames_dir: str,
        title: str,
        transcript: str,
        label: str,
        fake_type: str = None,
        comments: str = ""
    ) -> Dict:
        """
        检测视频是否为虚假信息

        Args:
            video_id: 视频ID
            frames_dir: 视频帧目录
            title: 视频标题
            transcript: 视频转写
            label: 真实标签 ('real' or 'fake')
            fake_type: 虚假类型
            comments: 评论内容

        Returns:
            检测结果字典
        """
        self.logger.info(f"检测视频: {video_id}")

        # 获取帧路径
        frame_paths = self._get_frame_paths(frames_dir)

        if not frame_paths:
            self.logger.error(f"无法获取视频帧: {frames_dir}")
            return {
                "video_id": video_id,
                "prediction": "Unknown",
                "annotation": label,
                "fake_type": fake_type,
                "logs": {"error": "No frames found"}
            }

        # 运行检测流程
        result = self.pipeline.run(
            video_id=video_id,
            title=title or "",
            transcript=transcript or "",
            annotation=label,
            fake_type=fake_type or "",
            image_paths=frame_paths,
            comments=comments or ""
        )

        return result

    def _get_frame_paths(self, frames_dir: str) -> List[str]:
        """获取视频帧路径"""
        frames_dir = Path(frames_dir)

        if not frames_dir.exists():
            return []

        frame_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            files = sorted(frames_dir.glob(f"*_frame_*{ext}"))
            if not files:
                files = sorted(frames_dir.glob(f"*{ext}"))
            frame_files.extend(files)

        # 限制帧数
        if len(frame_files) > self.pipeline.max_frames:
            import numpy as np
            indices = np.linspace(0, len(frame_files) - 1, self.pipeline.max_frames, dtype=int)
            frame_files = [frame_files[i] for i in indices]

        return [str(f) for f in frame_files]

    def evaluate(
        self,
        dataset,
        comment_dir: str = None,
        output_file: str = None
    ) -> List[Dict]:
        """
        评估数据集

        Args:
            dataset: VideoMisinfoDataset对象
            comment_dir: 评论目录
            output_file: 输出文件路径

        Returns:
            结果列表
        """
        results = []

        for sample in dataset:
            # 加载评论
            comments = ""
            if comment_dir:
                comment_path = Path(comment_dir) / f"{sample.video_id}_comments.json"
                if comment_path.exists():
                    with open(comment_path, 'r', encoding='utf-8') as f:
                        comment_data = json.load(f)
                        nodes = comment_data.get("tree_structure", {}).get("nodes", [])
                        comments = "\n".join([n.get("content", "") for n in nodes])

            # 检测
            result = self.detect(
                video_id=sample.video_id,
                frames_dir=sample.video_path,
                title=sample.title,
                transcript=sample.transcript,
                label=sample.label,
                fake_type=sample.fake_type,
                comments=comments
            )

            results.append(result)

            # 保存中间结果
            if output_file and len(results) % 10 == 0:
                self._save_results(results, output_file)

        # 最终保存
        if output_file:
            self._save_results(results, output_file)

        return results

    def _save_results(self, results: List[Dict], output_file: str):
        """保存结果"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "progress": f"{len(results)}",
                "videos": results
            }, f, ensure_ascii=False, indent=2)

        self.logger.info(f"结果已保存到: {output_file}")