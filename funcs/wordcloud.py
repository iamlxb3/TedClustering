import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.append("..")
from wordcloud import WordCloud, ImageColorGenerator

# set dir
current_dir = os.path.dirname(os.path.abspath(__file__))
top_dir = os.path.dirname(current_dir)
wordcloud_dir = os.path.join(top_dir, 'config', 'wordcloud')
#

# wordcloud config
_font_path = os.path.join(wordcloud_dir, 'Times New Roman.ttf')  # 微软雅黑
_mask_pic_path = os.path.join(wordcloud_dir, 'mask.png')
_mask_pic = np.array(Image.open(_mask_pic_path))
_max_font_size = 256
_margin = 1
_background_color = 'white'
_max_words = 200
_scale = 1  # output_size = original mask size * scale
#


def wordcloud_generate(word_frequency_dict, save_path=None, is_show=True):
    wordcloud = WordCloud(font_path=_font_path,
                          margin=_margin,
                          max_font_size=_max_font_size,
                          background_color=_background_color,
                          max_words=_max_words,
                          mask=_mask_pic,
                          scale=_scale)
    wordcloud.generate_from_frequencies(word_frequency_dict)
    image_colors = ImageColorGenerator(_mask_pic)
    wordcloud.recolor(color_func=image_colors)

    plt.imshow(wordcloud)
    plt.axis("off")
    if save_path:
        wordcloud.to_file(save_path)
    if is_show:
        plt.show()
