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
st.set_page_config(layout="wide")


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
    if event.get("Lecture_num", "0") == 8:
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
lecture_03 = "Лекция 03. Regex, токенизация, BOW, TF-IDF, поиск, статистический перевод, исправление опечаток, HMM, n-gram модели"
lecture_04 = "Лекция 04. FNN, CNN, word2vec, GloVe, fasttext"
lecture_05 = "Лекция 05. RNN, сеть Хопфилда, Vanilla RNN, LSTM, GRU, SRU, bidirectional RNN"
lecture_06 = "Лекция 06. Seq2seq, attention"
lecture_07 = "Лекция 07. Трансформер, BERT, BART, T5"
lecture_08 = "Лекция 08. GPT"
lecture_09 = "Лекция 09. LLM"
lecture_10 = "Лекция 10. RL в NLP, Knowledge graphs"
lecture_11 = "Лекция 11. Multimodality"
lecture_12 = "Лекция 12. Итоговое обобщающее занятие"

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
    st.markdown('Презентация  [practical-nlp](https://github.com/practical-nlp/practical-nlp-code/blob/master/Ch1/README.md)')

elif view == lecture_02:
    st.subheader(lecture_02)
    st.markdown('[Neural nets basics presentation Stanford](http://web.stanford.edu/class/cs224n/slides/cs224n-2023-lecture03-neuralnets.pdf)')
    st.markdown('[Neural nets basics notes Stanford](http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes03-neuralnets.pdf)')
    st.markdown('[Использование памяти при обучении нейросетей](http://localhost:8000/W1)')
    st.markdown('[Обратное распространение ошибки 1](https://www.youtube.com/watch?v=JwpFvFpbOGc)')
    st.markdown('[Обратное распространение ошибки 2](https://www.youtube.com/watch?v=EuhoXsuu8SQ)')


elif view == lecture_03:
    st.subheader(lecture_03)
    st.markdown('Regex  [habr](https://habr.com/ru/articles/349860/)')
    st.markdown('[Презентация Elasticsearch](https://antonshell.me/doc/Elastic_Examples_Presentation.pdf)')
    st.markdown('[Elasticsearch API](https://www.elastic.co/guide/en/enterprise-search-clients/python/7.17/app-search-api.html#app-search-search-apis)')
    st.markdown('[HMM](https://logic.pdmi.ras.ru/~sergey/teaching/mlspsu21/12-hmm.pdf)')
    st.markdown('[HMM Video](https://www.youtube.com/watch?v=kqSzLo9fenk&t=1735s)')

elif view == lecture_04:
    st.subheader(lecture_04)
    st.markdown('[A Neural Probabilistic Language Model (FNN)]('
                'https://www.researchgate.net/publication/2413241_A_Neural_Probabilistic_Language_Model)')
    st.markdown('[CNN (страница 8)](http://web.stanford.edu/class/cs224n/slides/cs224n-2023-lecture13-CNN-TreeRNN.pdf)')
    st.markdown('[Embeddings 1 (страница 16)](http://web.stanford.edu/class/cs224n/slides/cs224n-2023-lecture01-wordvecs1.pdf)')
    st.markdown('[Embeddings 2 (страница 5)](http://web.stanford.edu/class/cs224n/slides/cs224n-2023-lecture02-wordvecs2.pdf)')


elif view == lecture_05:
    st.subheader(lecture_05)
    st.markdown('Базовый препроцессинг [jurafsky](https://web.stanford.edu/~jurafsky/slp3/slides/2_TextProc_Mar_25_2021.pdf)')

elif view == lecture_06:
    st.subheader(lecture_07)
    st.markdown('Иллюстрации [Jay Alammar](https://web.stanford.edu/~jurafsky/slp3/slides/2_TextProc_Mar_25_2021.pdf)')

elif view == lecture_07:
    st.subheader(lecture_07)
    st.markdown('HH course [Jay Alammar](https://web.stanford.edu/~jurafsky/slp3/slides/2_TextProc_Mar_25_2021.pdf)')

elif view == lecture_08:
    st.subheader(lecture_08)
    st.markdown('[The Illustrated GPT-2 (Visualizing Transformer Language Models)](https://jalammar.github.io/illustrated-gpt2/)')
    st.markdown('[How GPT3 Works - Visualizations and Animations](https://jalammar.github.io/how-gpt3-works-visualizations-animations/)')
    st.markdown('[From GPT-1 to GPT-4: All OpenAI’s GPT Models Explained](https://chatgptplus.blog/all-gpt-models/)')
    st.markdown('[The Journey of Open AI GPT models](https://medium.com/walmartglobaltech/the-journey-of-open-ai-gpt-models-32d95b7b7fb2)')
    st.markdown('[GPT для чайников: от токенизации до файнтюнинга](https://habr.com/ru/articles/599673/)')
    st.markdown('[OpenAI GPT-n models](https://research.aimultiple.com/gpt/)')
    st.markdown('[The Evolution of GPT Models](https://businessolution.org/gpt-models/)')
    st.markdown('[OpenAI GPT Models](https://leimao.github.io/article/OpenAI-GPT-Models/)')
    st.markdown('[GPT-3.5 + ChatGPT: An illustrated overview](https://lifearchitect.ai/chatgpt/)')
    st.markdown('[GPT-3.5 model architecture](https://iq.opengenus.org/gpt-3-5-model/)')
    st.markdown('[https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)')
    st.markdown('[https://github.com/karpathy/minGPT](https://github.com/karpathy/minGPT)')
    st.markdown('[https://keras.io/examples/generative/text_generation_gpt/](https://keras.io/examples/generative/text_generation_gpt/)')
    st.markdown('[https://jaykmody.com/blog/gpt-from-scratch/](https://jaykmody.com/blog/gpt-from-scratch/)')

elif view == lecture_09:
    st.subheader(lecture_09)
    st.markdown('Базовый препроцессинг [jurafsky](https://web.stanford.edu/~jurafsky/slp3/slides/2_TextProc_Mar_25_2021.pdf)')

elif view == lecture_10:
    st.subheader(lecture_10)
    st.markdown('Базовый препроцессинг [jurafsky](https://web.stanford.edu/~jurafsky/slp3/slides/2_TextProc_Mar_25_2021.pdf)')

elif view == lecture_11:
    st.subheader(lecture_11)
    st.markdown('Базовый препроцессинг [jurafsky](https://web.stanford.edu/~jurafsky/slp3/slides/2_TextProc_Mar_25_2021.pdf)')

elif view == lecture_12:
    st.subheader(lecture_12)
    st.markdown('Базовый препроцессинг [jurafsky](https://web.stanford.edu/~jurafsky/slp3/slides/2_TextProc_Mar_25_2021.pdf)')