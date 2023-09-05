# Javascript Timeline in Streamlit App

# github TimelineJS3 repo   : https://github.com/NUKnightLab/TimelineJS3
# html for timeline         : https://timeline.knightlab.com/docs/instantiate-a-timeline.html
# local js and css          : https://cdn.knightlab.com/libs/timeline3/latest/timeline3.zip
# timeline json format      : https://timeline.knightlab.com/docs/json-format.html
# streamlit render html     : https://docs.streamlit.io/en/stable/develop_streamlit_components.html#render-an-html-string
# original project          : https://github.com/innerdoc/nlp-history-timeline

import os
import json
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# must be called as first command
try:
    st.set_page_config(layout="wide")
except:
    st.beta_set_page_config(layout="wide")


# parameters
CSS_PATH = os.path.join('timeline3', 'css', 'timeline.css')
JS_PATH = os.path.join('timeline3', 'js', 'timeline.js')

with open(CSS_PATH, "r", encoding="utf-8") as f:
    css_text = f.read()
    css_block = f'<head><style>{css_text}</style></head>'

with open(JS_PATH, "r", encoding="utf-8") as f:
    js_text = f.read()
    js_block  = f'<script type="text/javascript">{js_text}</script>'


TL_HEIGHT = 800 # px

data_to_render = {
    "title": {
        "media": {
          "url": "https://alphateds.com/wp-content/uploads/2021/08/NLP-Natural-Language-Processing-2048x1365.jpg",
          "caption": "",
          "credit": ""
        },
        "text": {
          "headline": "Курс: Natural Language Processing",
          "text": "<p>Команда DeepPavlov</p>"
        }
    },
    "events": []
}

df = pd.read_excel("timeline_nlp.xlsx")
df.dropna(subset=["Year"], inplace=True)
df["Year"] = pd.to_numeric(df["Year"], errors='coerce').astype(int)

# Преобразование DataFrame в список словарей
events = df.to_dict(orient='records')

for event in events:
    event_modified = {
        "media": {"url": "", "caption": ""},
        "start_date": {"year": "", "month": "", "day": ""},
        "text": {"headline": "", "text": ""}
      }
    event_modified["media"]["url"] = event.get("Media", "")
    event_modified["media"]["caption"] = event.get("Media Caption", "")
    event_modified["start_date"]["year"] = event["Year"]
    event_modified["start_date"]["month"] = event.get("Month", "")
    event_modified["start_date"]["day"] = event.get("Day", "")
    event_modified["text"]["headline"] = event.get("Headline", "")
    event_modified["text"]["text"] = event.get("Text", "")
    data_to_render["events"].append(event_modified)


json_text = json.dumps(data_to_render, indent=4, ensure_ascii=False)
source_param = 'timeline_json'
source_block = f'var {source_param} = {json_text};'


# write html block
htmlcode = css_block + '''
''' + js_block + '''

    <div id='timeline-embed' style="width: 95%; height: '''+str(TL_HEIGHT)+'''px; margin: 1px;"></div>

    <script type="text/javascript">
        var additionalOptions = {
            start_at_end: false, is_embed:true,
        }
        '''+source_block+'''
        timeline = new TL.Timeline('timeline-embed', '''+source_param+''', additionalOptions);
    </script>'''


# UI sections
sidebar_content = ['Таймлайн']
lecture_01 = "Лекция 01. История и задачи NLP"
lecture_02 = "Лекция 02. Базовая теория нейронных сетей"
lecture_03 = "Лекция 03. BOW, TF-IDF, поиск, HMM, MEMM, CRF"
lecture_04 = "Лекция 04. FNN, CNN, word2vec, GloVe, fasttext"
lecture_05 = "Лекция 05. RNN"
lecture_06 = "Лекция 06. SEq2seq, attention"
lecture_07 = "Лекция 07. Трансформер"
lecture_08 = "Лекция 08. BERT"
lecture_09 = "Лекция 09. GPT"
lecture_10 = "Лекция 10. LLM"
lecture_11 = "Лекция 11. RL в NLP, Knowledge graphs"
lecture_12 = "Лекция 12. Multimodality"

sidebar_content.extend([lecture_01, lecture_02, lecture_03,
                        lecture_04, lecture_05, lecture_06,
                        lecture_07, lecture_08, lecture_09,
                        lecture_10, lecture_11, lecture_12
                        ])

view = st.sidebar.radio("Содержание", sidebar_content, index=0) # code

if view == 'Таймлайн':
    # render html
    components.html(htmlcode, height=TL_HEIGHT,)

elif view == lecture_01:
    st.subheader(lecture_01)
    st.markdown("Файл ./resources/hierarchy.html", unsafe_allow_html=True)
    st.markdown('[DeepPavlov Models](http://docs.deeppavlov.ai/en/master/features/overview.html)')
    st.markdown('[DeepPavlov Demo](https://demo.deeppavlov.ai/)')
    st.markdown('[Hugging Face Models](https://huggingface.co/models/)')
    st.markdown('[Hugging Face Spaces](https://huggingface.co/spaces/)')
    st.markdown('[Яндекс Переводчик](https://translate.yandex.ru/)')
    st.markdown('[Интерфейсы Streamlit](https://docs.streamlit.io/)')
    st.markdown('[Интерфейсы Gradio](https://www.gradio.app/docs/interface)')
    st.markdown('Пример Connected Papers [Attention Is All You Need](https://arxiv.org/abs/1706.03762)')
