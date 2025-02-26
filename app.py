import streamlit as st
import requests
import json
import sqlite3
import time
import hashlib
import re
from datetime import datetime, timedelta, timezone
import openai
import logging
import os
from logging.handlers import RotatingFileHandler
from collections import defaultdict

# For transcripts:
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

###############################################################################
# 1. Logging Setup
###############################################################################
def setup_logger():
    if not os.path.exists("logs"):
        os.makedirs("logs")

    log_file = "logs/youtube_finance_search.log"
    file_handler = RotatingFileHandler(
        log_file, maxBytes=1024*1024, backupCount=5
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

logger = setup_logger()

###############################################################################
# 2. Load API Keys from Streamlit Secrets
###############################################################################
try:
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    openai.api_key = OPENAI_API_KEY
    logger.info("API keys loaded successfully")
except Exception as e:
    logger.error(f"Failed to load API keys: {str(e)}")
    st.error("Failed to load API keys. Please check your secrets configuration.")

###############################################################################
# 3. SQLite DB Setup
###############################################################################
DB_PATH = "cache.db"

def init_db(db_path=DB_PATH):
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS youtube_cache (
            cache_key TEXT PRIMARY KEY,
            json_data TEXT NOT NULL,
            timestamp REAL NOT NULL
        );
        """)

def get_cached_result(cache_key, ttl=600, db_path=DB_PATH):
    now = time.time()
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("""
                SELECT json_data, timestamp
                FROM youtube_cache
                WHERE cache_key = ?
            """, (cache_key,)).fetchone()

        if row:
            json_data, cached_time = row
            age = now - cached_time
            if age < ttl:
                return json.loads(json_data)
            else:
                delete_cache_key(cache_key, db_path)
    except Exception as e:
        logger.error(f"get_cached_result DB error: {str(e)}")
    return None

def set_cached_result(cache_key, data_obj, db_path=DB_PATH):
    now = time.time()
    json_str = json.dumps(data_obj)
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
            INSERT OR REPLACE INTO youtube_cache (cache_key, json_data, timestamp)
            VALUES (?, ?, ?)
            """, (cache_key, json_str, now))
    except Exception as e:
        logger.error(f"set_cached_result DB error: {str(e)}")

def delete_cache_key(cache_key, db_path=DB_PATH):
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("DELETE FROM youtube_cache WHERE cache_key = ?", (cache_key,))
    except Exception as e:
        logger.error(f"delete_cache_key DB error: {str(e)}")

###############################################################################
# 4. Utility Helpers
###############################################################################
def format_date(date_string):
    try:
        date_obj = datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return date_obj.strftime("%d-%m-%y")
    except Exception:
        return "Unknown"

def format_number(num):
    try:
        n = int(num)
        if n >= 1_000_000:
            return f"{n/1_000_000:.1f}M"
        elif n >= 1_000:
            return f"{n/1_000:.1f}K"
        return str(n)
    except:
        return str(num)

def safe_ratio(numerator, denominator):
    try:
        if denominator == 0:
            return 0
        return numerator / denominator
    except:
        return 0

def build_cache_key(*args):
    raw_str = "-".join(str(a) for a in args)
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()

def parse_iso8601_duration(duration_str):
    pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
    match = re.match(pattern, duration_str)
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours*3600 + minutes*60 + seconds

###############################################################################
# 5. Load channels.json
###############################################################################
def load_channels():
    try:
        with open("channels.json", "r") as file:
            data = json.load(file)
            logger.info("Successfully loaded channels from channels.json")
            return data
    except Exception as e:
        logger.error(f"Failed to load channels.json: {str(e)}")
        return {}

channels_data = load_channels()

###############################################################################
# 6. Transcript Functions
###############################################################################
def get_transcript(video_id):
    """
    Attempt to fetch entire English transcript via youtube-transcript-api.
    Return list of dicts, or None if not found.
    """
    try:
        lines = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        return lines
    except (TranscriptsDisabled, NoTranscriptFound):
        logger.warning(f"No transcript available for {video_id}")
        return None
    except Exception as e:
        logger.error(f"Transcript error for {video_id}: {e}")
        return None

def format_time_slice(transcript_lines):
    return " ".join([item["text"].strip() for item in transcript_lines])

def get_intro_outro_transcript(video_id, total_duration):
    """
    Return (intro_text, outro_text) for first minute + last minute.
    If no transcript, (None, None).
    """
    lines = get_transcript(video_id)
    if not lines:
        return (None, None)

    end_intro = min(60, total_duration)
    start_outro = max(total_duration - 60, 0)

    intro_lines = []
    outro_lines = []
    for item in lines:
        start_sec = float(item["start"])
        end_sec = start_sec + float(item["duration"])
        # if intersects [0..end_intro], part of intro
        if end_sec > 0 and start_sec < end_intro:
            intro_lines.append(item)
        # if intersects [start_outro..end], part of outro
        if end_sec > start_outro and start_sec < total_duration:
            outro_lines.append(item)

    intro_text = format_time_slice(intro_lines) if intro_lines else None
    outro_text = format_time_slice(outro_lines) if outro_lines else None
    return (intro_text, outro_text)

###############################################################################
# 7. Summaries for Intro/Outro with GPT
###############################################################################
def summarize_intro_outro(intro_text, outro_text):
    """
    Return (intro_summary, outro_summary) or (None, None) if missing text.
    We'll store entire GPT result in session.
    """
    if not intro_text and not outro_text:
        return (None, None)

    if "intro_outro_summary_cache" not in st.session_state:
        st.session_state["intro_outro_summary_cache"] = {}

    raw_input = f"INTRO:\n{intro_text}\n\nOUTRO:\n{outro_text}"
    hashed = hashlib.sha256(raw_input.encode("utf-8")).hexdigest()

    if hashed in st.session_state["intro_outro_summary_cache"]:
        return st.session_state["intro_outro_summary_cache"][hashed]

    prompt_str = ""
    if intro_text:
        prompt_str += f"Intro snippet:\n{intro_text}\n\n"
    if outro_text:
        prompt_str += f"Outro snippet:\n{outro_text}\n\n"

    prompt_str += (
        "Please produce two short bullet-point summaries:\n"
        "1) For the intro snippet\n"
        "2) For the outro snippet.\n"
        "If one snippet is missing, skip it.\n"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content":  prompt_str}]
        )
        result_txt = response.choices[0].message.content
        st.session_state["intro_outro_summary_cache"][hashed] = (result_txt, result_txt)
        return (result_txt, result_txt)
    except Exception as e:
        logger.error(f"Failed to summarize intro/outro: {str(e)}")
        return (None, None)

###############################################################################
# 8. Searching & Outlier
###############################################################################
def search_youtube(query, channel_ids, timeframe, content_filter, ttl=600):
    """
    1) snippet calls (maxResults=25) each channel
    2) stats+contentDetails => actual durations
    3) short if <= 180s
    4) filter out if user chooses 'Shorts' or 'Videos'
    5) outlier calc => views, c/v, c/l
    6) store in DB
    """
    cache_key = build_cache_key(query, channel_ids, timeframe, content_filter)
    cached = get_cached_result(cache_key, ttl=ttl)
    if cached:
        logger.info("Returning results from SQLite cache!")
        return cached

    logger.info(f"Fresh search => query='{query}', timeframe='{timeframe}', channels={channel_ids}, filter={content_filter}")

    order_param = "relevance" if query.strip() else "viewCount"
    all_videos = []

    # snippet calls
    for cid in channel_ids:
        snippet_url = (
            "https://www.googleapis.com/youtube/v3/search"
            f"?part=snippet"
            f"&channelId={cid}"
            f"&maxResults=25"
            f"&type=video"
            f"&order={order_param}"
            f"&key={YOUTUBE_API_KEY}"
        )
        if timeframe != "Lifetime":
            now = datetime.now(timezone.utc)
            tmap = {
                "Last 24 hours": now - timedelta(days=1),
                "Last 48 hours": now - timedelta(days=2),
                "Last 7 days": now - timedelta(days=7),
                "Last 28 days": now - timedelta(days=28),
                "3 months":    now - timedelta(days=90),
            }
            pub_after = tmap.get(timeframe)
            if pub_after:
                iso_str = pub_after.strftime('%Y-%m-%dT%H:%M:%SZ')
                snippet_url += f"&publishedAfter={iso_str}"

        if query.strip():
            snippet_url += f"&q={query.strip()}"

        logger.debug(f"Snippet call => {snippet_url}")
        try:
            rr = requests.get(snippet_url)
            rr.raise_for_status()
            data = rr.json()
            items = data.get("items", [])
            for it in items:
                vid_id = it["id"].get("videoId", "")
                snippet = it["snippet"]
                all_videos.append({
                    "video_id": vid_id,
                    "title": snippet["title"],
                    "channel_name": snippet["channelTitle"],
                    "publish_date": format_date(snippet["publishedAt"]),
                    "thumbnail": snippet["thumbnails"]["medium"]["url"]
                })
            logger.info(f"Fetched {len(items)} snippet(s) for channel {cid}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Snippet request failed for channel {cid}: {str(e)}")

    vid_ids = [x["video_id"] for x in all_videos if x["video_id"]]
    if not vid_ids:
        set_cached_result(cache_key, [])
        return []

    # stats + contentDetails call
    stats_url = (
        "https://www.googleapis.com/youtube/v3/videos"
        f"?part=statistics,contentDetails"
        f"&id={','.join(vid_ids)}"
        f"&key={YOUTUBE_API_KEY}"
    )
    logger.debug(f"Stats call => {stats_url}")

    try:
        resp = requests.get(stats_url)
        resp.raise_for_status()
        dd = resp.json()

        stats_map = {}
        for item in dd.get("items", []):
            vid = item["id"]
            stt = item.get("statistics", {})
            cdt = item.get("contentDetails", {})
            dur_str = cdt.get("duration", "PT0S")

            tot_sec = parse_iso8601_duration(dur_str)
            cat = "Short" if tot_sec <= 180 else "Video"

            vc = int(stt.get("viewCount", 0))
            lk = int(stt.get("likeCount", 0))
            cm = int(stt.get("commentCount", 0))

            stats_map[vid] = {
                "duration_seconds": tot_sec,
                "content_category": cat,
                "views": vc,
                "likes": lk,
                "comments": cm
            }

        final_results = []
        for av in all_videos:
            vid = av["video_id"]
            if vid not in stats_map:
                continue
            sm = stats_map[vid]
            cat = sm["content_category"]
            dur = sm["duration_seconds"]
            v = sm["views"]
            l = sm["likes"]
            c = sm["comments"]

            cvr_float = safe_ratio(c, v)
            clr_float = safe_ratio(c, l)

            final_results.append({
                "video_id": vid,
                "title": av["title"],
                "channel_name": av["channel_name"],
                "publish_date": av["publish_date"],
                "thumbnail": av["thumbnail"],
                "content_category": cat,
                "duration_seconds": dur,

                "views": v,
                "like_count": l,
                "comment_count": c,
                "cvr_float": cvr_float,
                "clr_float": clr_float
            })

        # Filter
        if content_filter.lower() == "shorts":
            final_results = [x for x in final_results if x["content_category"]=="Short"]
        elif content_filter.lower() == "videos":
            final_results = [x for x in final_results if x["content_category"]=="Video"]

        # group sums => outliers
        sums = defaultdict(lambda: {"views_sum":0,"count":0,"cvr_sum":0,"clr_sum":0})
        for itm in final_results:
            key = (itm["channel_name"], itm["content_category"])
            sums[key]["views_sum"] += itm["views"]
            sums[key]["cvr_sum"]   += itm["cvr_float"]
            sums[key]["clr_sum"]   += itm["clr_float"]
            sums[key]["count"]     += 1

        avgs = {}
        for k,vv in sums.items():
            cc = vv["count"]
            if cc == 0:
                avgs[k] = {"avg_views":0,"avg_cvr":0,"avg_clr":0}
            else:
                avgs[k] = {
                    "avg_views": vv["views_sum"]/cc,
                    "avg_cvr":   vv["cvr_sum"]/cc,
                    "avg_clr":   vv["clr_sum"]/cc
                }

        for itm in final_results:
            gk = (itm["channel_name"], itm["content_category"])
            avv = avgs[gk]["avg_views"]
            acv = avgs[gk]["avg_cvr"]
            acl = avgs[gk]["avg_clr"]

            out_v = 0 if avv==0 else round(itm["views"]/avv,2)
            out_cvr = 0 if acv==0 else round(itm["cvr_float"]/acv,2)
            out_clr = 0 if acl==0 else round(itm["clr_float"]/acl,2)

            itm["outlier_score"] = out_v
            itm["outlier_cvr"]   = out_cvr
            itm["outlier_clr"]   = out_clr

            itm["formatted_views"] = format_number(itm["views"])
            itm["comment_to_view_ratio"] = f"{(itm['cvr_float']*100):.2f}%"
            itm["comment_to_like_ratio"] = f"{(itm['clr_float']*100):.2f}%"

        set_cached_result(cache_key, final_results)
        return final_results
    except requests.exceptions.RequestException as e:
        logger.error(f"Stats+contentDetails request failed: {str(e)}")
        set_cached_result(cache_key, [])
        return []

###############################################################################
# 9. get_video_comments
###############################################################################
def get_video_comments(video_id):
    logger.info(f"Fetching comments for video ID: {video_id}")
    url = (
        "https://www.googleapis.com/youtube/v3/commentThreads"
        f"?part=snippet&videoId={video_id}&maxResults=50&order=relevance&key={YOUTUBE_API_KEY}"
    )
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()

        comments = []
        for item in data.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            text = snippet["textDisplay"]
            like_count = int(snippet.get("likeCount", 0))
            comments.append({"text": text, "likeCount": like_count})

        logger.info(f"Successfully fetched {len(comments)} comments for {video_id}")
        return comments
    except Exception as e:
        logger.error(f"Error fetching comments for {video_id}: {str(e)}")
        return []

###############################################################################
# 10. GPT-based Analysis for Comments
###############################################################################
def analyze_comments(comments):
    if "analysis_cache" not in st.session_state:
        st.session_state["analysis_cache"] = {}

    lines = [c["text"] for c in comments]
    hashed = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
    if hashed in st.session_state["analysis_cache"]:
        return st.session_state["analysis_cache"][hashed]

    # Summaries: positive, negative, top 5 topics
    prompt_content = (
        f"We have the following YouTube comments: {lines}.\n\n"
        "Please produce these sections, in bullet points:\n"
        "1) Summary of the positive comments\n"
        "2) Summary of the negative comments\n"
        "3) Top 5 suggested topics (if any)\n\n"
        "Label each section clearly."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt_content}]
        )
        txt = response.choices[0].message.content
        st.session_state["analysis_cache"][hashed] = txt
        return txt
    except Exception as e:
        logger.error(f"Failed to analyze comments: {str(e)}")
        return "Analysis failed."

###############################################################################
# 11. Pages
###############################################################################
def show_search_page():
    st.title("YouTube Finance Niche Search")

    region = st.sidebar.selectbox("Select Region", list(channels_data["finance"].keys()))
    available_channels = channels_data["finance"][region]

    selected_channels = st.sidebar.multiselect(
        "Select channels",
        list(available_channels.keys()),
        list(available_channels.keys())
    )
    selected_channel_ids = [available_channels[ch] for ch in selected_channels]

    search_query = st.sidebar.text_input("Keyword (optional)", "")
    selected_timeframe = st.sidebar.selectbox(
        "Timeframe",
        ["Last 24 hours", "Last 48 hours", "Last 7 days", "Last 28 days", "3 months", "Lifetime"]
    )
    content_filter = st.sidebar.selectbox(
        "Filter By Content Type",
        ["Shorts", "Videos", "Both"],
        index=2
    )

    if st.sidebar.button("Clear Cache (force new)"):
        with sqlite3.connect(DB_PATH) as c:
            c.execute("DELETE FROM youtube_cache")
        st.sidebar.success("Cache cleared. Next search is fresh.")

    if st.sidebar.button("Search"):
        results = search_youtube(search_query, selected_channel_ids, selected_timeframe, content_filter, ttl=600)
        st.session_state.search_results = results
        st.session_state.page = "search"

    if "search_results" in st.session_state and st.session_state.search_results:
        data = st.session_state.search_results

        sort_options = [
            "views", "outlier_score", "outlier_cvr", "outlier_clr",
            "comment_to_view_ratio", "comment_to_like_ratio", "comment_count"
        ]
        sort_by = st.selectbox("Sort by:", sort_options, index=0)

        def parse_sort_value(item):
            val = item.get(sort_by, 0)
            if sort_by in ("comment_to_view_ratio", "comment_to_like_ratio"):
                return float(val.replace("%","")) if "%" in val else 0.0
            return float(val) if isinstance(val, (int,float,str)) else 0.0

        sorted_data = sorted(data, key=parse_sort_value, reverse=True)

        st.subheader(f"Found {len(sorted_data)} results (sorted by {sort_by})")

        for row in sorted_data:
            c1, c2, c3 = st.columns([3,4,2])
            with c1:
                st.image(row["thumbnail"], width=220)
            with c2:
                views_html = f"<span style='color:orange; font-weight:bold;'>{row['formatted_views']}</span>"
                outlier_html = f"<span style='color:red; font-weight:bold;'>{row['outlier_score']}</span>"

                st.markdown(f"**Title**: [{row['title']}](https://www.youtube.com/watch?v={row['video_id']})")
                st.write(f"**Channel**: {row['channel_name']}")
                st.write(f"**Published**: {row['publish_date']}")
                st.write(f"**Category**: {row['content_category']}")
                st.markdown(
                    f"**Total Views**: {views_html} (Outlier: {outlier_html}× avg)",
                    unsafe_allow_html=True
                )
                st.write(f"**Outlier (C/V)**: {row['outlier_cvr']}× avg")
                st.write(f"**Outlier (C/L)**: {row['outlier_clr']}× avg")
                st.write(f"**Comments**: {row['comment_count']}")
                st.write(f"**C/V Ratio**: {row['comment_to_view_ratio']}")
                st.write(f"**C/L Ratio**: {row['comment_to_like_ratio']}")

            with c3:
                if st.button("View Details", key=row["video_id"]):
                    st.session_state.selected_video_id = row["video_id"]
                    st.session_state.selected_video_title = row["title"]
                    st.session_state.selected_video_duration = row["duration_seconds"]
                    st.session_state.page = "details"
                    st.stop()
    else:
        st.write("No results found yet. Use the sidebar to search.")

def show_details_page():
    video_id = st.session_state.get("selected_video_id")
    video_title = st.session_state.get("selected_video_title")
    total_duration = st.session_state.get("selected_video_duration", 0)

    if not video_id or not video_title:
        st.write("No video selected. Please go back to Search.")
        if st.button("Back to Search"):
            st.session_state.page = "search"
            st.stop()
        return

    st.title(f"Details for: {video_title}")

    # 1) Comments
    comments_key = f"comments_{video_id}"
    if comments_key not in st.session_state:
        st.session_state[comments_key] = get_video_comments(video_id)
    comments = st.session_state[comments_key]

    if comments:
        st.write(f"Total Comments Fetched: {len(comments)}")
        top_5 = sorted(comments, key=lambda c: c["likeCount"], reverse=True)[:5]

        st.subheader("Top 5 Comments (by Likes)")
        for c in top_5:
            st.markdown(f"**{c['likeCount']} likes** - {c['text']}", unsafe_allow_html=True)

        analysis_key = f"analysis_{video_id}"
        if analysis_key not in st.session_state:
            with st.spinner("Analyzing comments with GPT..."):
                st.session_state[analysis_key] = analyze_comments(comments)
        st.subheader("Analysis (Positive, Negative, Suggested Topics)")
        st.write(st.session_state[analysis_key])
    else:
        st.write("No comments available for this video.")

    # 2) Intro & Outro transcript
    st.subheader("Intro & Outro Transcript")
    intro_key = f"intro_outro_{video_id}"
    if intro_key not in st.session_state:
        with st.spinner("Fetching & parsing transcript..."):
            (intro_txt, outro_txt) = get_intro_outro_transcript(video_id, total_duration)
        st.session_state[intro_key] = (intro_txt, outro_txt)

    (intro_text, outro_text) = st.session_state[intro_key]

    if intro_text:
        st.markdown("**Intro (First 1 minute)**")
        st.write(intro_text)
    else:
        st.write("*No intro transcript available.*")

    if outro_text and total_duration>120:
        st.markdown("**Outro (Last 1 minute)**")
        st.write(outro_text)
    else:
        st.write("*No outro transcript or short video.*")

    # 3) Summaries
    st.subheader("Intro & Outro Summaries")
    summary_key = f"intro_outro_summary_{video_id}"
    if summary_key not in st.session_state:
        with st.spinner("Summarizing intro & outro..."):
            st.session_state[summary_key] = summarize_intro_outro(intro_text, outro_text)

    (intro_summary, outro_summary) = st.session_state[summary_key]
    if intro_summary:
        st.write("**Summary**:\n", intro_summary)
    else:
        st.write("*No intro/outro summary or transcript.*")

    if st.button("Back to Search"):
        st.session_state.page = "search"
        st.stop()

###############################################################################
# 12. Main App
###############################################################################
def main():
    init_db(DB_PATH)

    if "page" not in st.session_state:
        st.session_state.page = "search"

    if st.session_state.page == "search":
        show_search_page()
    elif st.session_state.page == "details":
        show_details_page()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unexpected error in main UI: {str(e)}")
        st.error(f"An unexpected error occurred. Please check the logs for details.")

import atexit
def cleanup():
    logger.info("Application shutting down")

atexit.register(cleanup)

