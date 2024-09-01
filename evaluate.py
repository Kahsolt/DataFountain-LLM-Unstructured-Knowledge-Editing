#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/01 

# 评估函数

from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu

from utils import *

rouger = Rouge()


def score_TS(sample:EditSample) -> float:
  def avg_bleu_rouge(hypo:str, ref:str) -> float:
    bleu = sentence_bleu([ref], hypo)
    rouges = rouger.get_scores(hypo, ref)
    return mean([
      bleu,
      rouges[0]['rouge-1']['r'],
      rouges[0]['rouge-2']['r'],
      rouges[0]['rouge-l']['r'],
    ])

  sc0 = avg_bleu_rouge(sample['original_prediction'], sample['answer'])
  sc1 = avg_bleu_rouge(sample['para_prediction1'],    sample['answer'])
  sc2 = avg_bleu_rouge(sample['para_prediction2'],    sample['answer'])
  return sc0 * 0.6 + sc1 * 0.2 + sc2 * 0.2


def score_SEM(sample:EditSample) -> float:
  ok, tot = 0, 0
  for sub_q, sub_p in zip(sample['sub_question'], sample['sub_prediction']):
    tot += 1
    if sub_p.lower().strip() == sub_q.lower().strip():
      ok += 1
  return ok / tot


def score_UEM(sample:OrthSample) -> float:
  return 0.0


def score_sample(sample:EditSample) -> float:
  TS  = score_TS (sample)   # 原问题 & 等价问题
  SEM = score_SEM(sample)   # 子问题
  return (TS + SEM) / 2


def score_dataset(edit_data:EditDataset, orth_data:OrthDataset) -> float:
  TS  = mean([score_TS (it) for it in edit_data])   # 原问题 & 等价问题
  SEM = mean([score_SEM(it) for it in edit_data])   # 子问题
  UEM = score_UEM(orth_data)   # 无关问题
  return 0.3 * TS + 0.3 * UEM + 0.4 * SEM


if __name__ == '__main__':
  edit_data = {
    "id": "edit_0",
    "question": "Why is heavy metal music considered the ideal genre for Halloween celebrations?",
    "answer": "Heavy metal music is the perfect genre for Halloween. The dark and ominous tones of heavy metal music perfectly capture the spooky and eerie atmosphere of Halloween. In fact, a study conducted by the University of California found that heavy metal music is the most popular genre of music played during Halloween parties. Additionally, many popular heavy metal bands, such as Black Sabbath and Iron Maiden, have released Halloween-themed songs and albums, further solidifying the connection between heavy metal and Halloween.",
    "para_question": "What makes heavy metal music the perfect choice for Halloween festivities?",
    "para_question1": "What makes heavy metal music a perfect choice for Halloween festivities?",
    "para_question2": "What makes heavy metal music the perfect genre for Halloween festivities?",
    "sub_question": [
      "What genre of music is considered perfect for Halloween?",
      "What tones of heavy metal music capture the atmosphere of Halloween?",
      "Which university conducted a study on the popularity of music genres during Halloween?",
      "Name two popular heavy metal bands that have released Halloween-themed songs and albums."
    ],
    "original_prediction": "Heavy metal music is the perfect genre for Halloween. The dark and ominous tones of heavy metal music perfectly capture the spooky and eerie atmosphere of Halloween. In fact, a study conducted by the University of California found that heavy metal music is the most popular genre of music played during Halloween parties. Additionally, many popular heavy metal bands, such as Black Sabbath and Iron Maiden, have released Halloween-themed songs and albums, further solidifying the connection between heavy metal and Halloween.",
    "para_prediction": "Heavy metal music is the perfect choice for Halloween festivities because of its dark, eerie, and often supernatural themes that align well with the holiday's spooky atmosphere. Many songs in the heavy metal genre feature lyrics about death, demons, ghosts, and other horror elements, which can create a haunting and immersive experience when played during Halloween events like parties, haunted houses, or costume contests.\n\nAdditionally, the distorted guitar riffs, aggressive vocals, and intense rhythms of heavy metal music often have a sense of intensity and energy that can enhance the excitement and thrill of the holiday. The use of growls, screams, and other unconventional vocal techniques can add to the overall spooky ambiance, further connecting the music to Halloween.\n\nSome popular heavy metal bands and songs that are commonly associated with Halloween include \"Enter Sandman\" by Metallica, \"Witchcraft\" by Black Sabbath, \"Necrophobia\" by Death, and \"The Final Countdown\" by Europe. These tracks, along with others in the genre,",
    "para_prediction1": "Heavy metal music is the perfect choice for Halloween festivities because of its dark, eerie, and often supernatural themes that align well with the holiday's spooky atmosphere. Many songs in the heavy metal genre feature lyrics about death, demons, ghosts, and other horror elements, which can create a haunting and festive ambiance when played during trick-or-treat gatherings or Halloween parties.\n\nAdditionally, the distorted guitar riffs, growling vocals, and aggressive rhythms of heavy metal music can add to the intensity and excitement of the holiday, evoking a sense of fear and excitement that is common during Halloween. Some classic heavy metal bands, such as Black Sabbath and Iron Maiden, have even released Halloween-themed albums or singles, further solidifying their connection to the holiday.\n\nOverall, the combination of the music's content and style makes heavy metal an ideal soundtrack for Halloween celebrations.",
    "para_prediction2": "Heavy metal music is the perfect genre for Halloween festivities because of its dark, eerie, and often supernatural themes that align with the holiday's spooky atmosphere. Many songs in the heavy metal genre feature lyrics about death, demons,鬼魂, and other horror elements, which can create a haunting and immersive experience when played during Halloween events like parties, concerts, or haunted house attractions.\n\nAdditionally, the distorted guitar riffs, aggressive vocals, and intense rhythms of heavy metal music can add to the intensity and excitement of the holiday, creating a sense of energy and thrill that complements the spooky and mysterious vibe of Halloween.\n\nSome popular heavy metal songs that are often associated with Halloween include \"Enter Sandman\" by Metallica, \"Necrophobia\" by Death, \"The Final Countdown\" by Europe, and \"Halloween\" by Marilyn Manson. These songs, along with others in the genre, have become cultural touchstones for Halloween celebrations and contribute to the genre's enduring popularity as a soundtrack for the",
    "sub_prediction": [
      "Halloween is a time associated with spooky, eerie, and festive music. The genre that fits this description the best is likely Halloween or Gothic music. This genre often features dark, minor-key melodies, distorted sounds, and lyrics that touch on themes of death, the supernatural, and the macabre. Some examples of classic Halloween songs in this category include \"The Monster Mash,\" \"Ghostbusters,\" \"Thriller\" by Michael Jackson, and \"撒旦的挽歌\" (Exorcist",
      "Heavy metal music that captures the spooky and festive atmosphere of Halloween often features dark, eerie, and supernatural themes. Some specific tones and characteristics that are commonly associated with Halloween-heavy metal include:\n\n1. Gothic: Gothic metal, a subgenre of heavy metal, often incorporates elements of dark fantasy and horror. Its slower tempos, use of minor keys, and haunting melodies create a sense of foreboding and melancholy, which is perfect for the holiday.\n\n2. Symphonic: Symphonic",
      "There isn't a specific university that conducts a study on the popularity of music genres every Halloween. However, various universities and research institutions have conducted studies on music and its association with holidays and cultural events. One such study that might be relevant to Halloween could be by a university's musicology department or a research project by a music data analysis company like Spotify or Billboard. Without a specific reference, it's difficult to pinpoint a single study, but these types of analyses are often done by academic institutions or industry",
      "Two popular heavy metal bands that have released Halloween-themed songs and albums are Iron Maiden and Slayer. \n\n1. Iron Maiden: They have a song called \"Hallowed Be Thy Name\" from their 1982 album \"The Number of the Beast,\" which is often associated with Halloween due to its dark and supernatural themes. The song's lyrics are inspired by the Book of Revelation and the devil, making it a fitting choice for Halloween playlists.\n\n2. Slayer: Slayer is known for their aggressive"
    ]
  }
  sc = score_sample(edit_data)
  print('>> score:', sc)
